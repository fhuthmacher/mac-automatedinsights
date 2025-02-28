import json
import sys
import logging
import pandas as pd
import boto3
import os
import tempfile
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel
import time

class Parameters(BaseModel):
    AthenaDatabase: Optional[str] = None

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# get the environment variables
if 'S3_BUCKET_NAME' not in globals():
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
    logger.info(f"S3_BUCKET_NAME: {S3_BUCKET_NAME}")

if 'MODEL_ID' not in globals():
    MODEL_ID = os.getenv('MODEL_ID')
    logger.info(f"MODEL_ID: {MODEL_ID}")

if 'ATHENA_QUERY_EXECUTION_LOCATION' not in globals():
    ATHENA_QUERY_EXECUTION_LOCATION = f's3://{S3_BUCKET_NAME}/athena_results/'
    logger.info(f"ATHENA_QUERY_EXECUTION_LOCATION: {ATHENA_QUERY_EXECUTION_LOCATION}")

# check if session python variable exists
if 'SESSION_PROFILE' in globals():
    logger.info(f"Session profile found: {SESSION_PROFILE}")
    session = boto3.Session(profile_name=SESSION_PROFILE, region_name=REGION)
else:
    logger.info('No session profile found, using default session')
    session = boto3.Session()

s3_client = session.client('s3')
athena_client = session.client('athena')



logger.info('start athena result location configuration')
try:
    response = athena_client.get_work_group(WorkGroup='primary')
    ConfigurationUpdates={}
    ConfigurationUpdates['EnforceWorkGroupConfiguration']= True
    ResultConfigurationUpdates= {}
    athena_location = "s3://"+ S3_BUCKET_NAME +"/athena_results/"
    ResultConfigurationUpdates['OutputLocation']=athena_location
    EngineVersion = response['WorkGroup']['Configuration']['EngineVersion']
    ConfigurationUpdates['ResultConfigurationUpdates']=ResultConfigurationUpdates
    ConfigurationUpdates['PublishCloudWatchMetricsEnabled']= response['WorkGroup']['Configuration']['PublishCloudWatchMetricsEnabled']
    ConfigurationUpdates['EngineVersion']=EngineVersion
    ConfigurationUpdates['RequesterPaysEnabled']= response['WorkGroup']['Configuration']['RequesterPaysEnabled']
    response2 = athena_client.update_work_group(WorkGroup='primary',ConfigurationUpdates=ConfigurationUpdates,State='ENABLED')
    logger.info(f"athena output location updated to s3://{S3_BUCKET_NAME}/athena_results/")  
except Exception as e:
    logger.error(str(e))

class BusinessAnalystTools:
    """Collection of tools for the Business Analyst Agent to use"""
    
    def __init__(self, session, s3_bucket_name: str, athena_database: str):
        logger.info(f"Initializing BusinessAnalystTools with bucket: {s3_bucket_name}, database: {athena_database}")
        self.s3_client = session.client('s3')
        self.athena_client = session.client('athena')
        self.s3_bucket_name = s3_bucket_name
        self.athena_database = athena_database
        
    def get_database_schema(self) -> str:
        """Retrieve the SQL database schema from S3"""
        schema_prefix = 'metadata/sql_table_definition'
        logger.info(f"Retrieving database schema from s3://{self.s3_bucket_name}/{schema_prefix}")
        
        sql_database_schema = []
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket_name, 
                Prefix=schema_prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No schema files found in s3://{self.s3_bucket_name}/{schema_prefix}")
                return "[]"
            
            logger.info(f"Found {len(response['Contents'])} schema files")
            
            for item in response['Contents']:
                if item['Key'].endswith('/'):
                    continue
                    
                logger.info(f"Reading schema file: {item['Key']}")
                try:
                    content = self.s3_client.get_object(
                        Bucket=self.s3_bucket_name, 
                        Key=item['Key']
                    )['Body'].read().decode('utf-8')
                    sql_database_schema.append(content)
                    logger.debug(f"Successfully read schema from {item['Key']}")
                except Exception as e:
                    logger.error(f"Error reading schema file {item['Key']}: {str(e)}")
            
            logger.info(f"Successfully retrieved {len(sql_database_schema)} schema definitions")
            return json.dumps(sql_database_schema)
            
        except Exception as e:
            logger.error(f"Error in get_database_schema: {str(e)}", exc_info=True)
            return "[]"

    def get_use_cases(self) -> List[Dict]:
        """Retrieve available AI/ML use cases from S3"""
        use_cases_path = 'metadata/use_cases/use_case_details.jsonl'
        logger.info(f"Retrieving use cases from s3://{self.s3_bucket_name}/{use_cases_path}")
        
        try:
            with tempfile.NamedTemporaryFile() as tmp:
                self.s3_client.download_file(
                    self.s3_bucket_name,
                    use_cases_path,
                    tmp.name
                )
                df = pd.read_json(tmp.name, lines=True)
                use_cases = df.to_dict('records')
                logger.info(f"Successfully retrieved {len(use_cases)} use cases")
                logger.debug(f"Use cases: {json.dumps(use_cases, indent=2)}")
                return use_cases
        except Exception as e:
            logger.error(f"Error in get_use_cases: {str(e)}", exc_info=True)
            return []

    def execute_query(self, query: str) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
        """Execute an Athena query and return results with detailed error info"""
        logger.info(f"Executing Athena query in database {self.athena_database}")
        logger.debug(f"Query: {query}")
        
        try:
            response = self.athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': self.athena_database,
                    'Catalog': 'AwsDataCatalog'
                },
                ResultConfiguration={
                    'OutputLocation': f's3://{self.s3_bucket_name}/athena_results/'
                }
            )
            
            query_id = response['QueryExecutionId']
            logger.info(f"Query execution started with ID: {query_id}")
            
            # Wait for completion
            while True:
                status = self.athena_client.get_query_execution(QueryExecutionId=query_id)
                state = status['QueryExecution']['Status']['State']
                logger.debug(f"Query state: {state}")
                
                if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                time.sleep(1)
            
            # Get detailed error information if query failed
            error_info = None
            if state != 'SUCCEEDED':
                status_details = status['QueryExecution']['Status']
                error_info = {
                    'state': state,
                    'reason': status_details.get('StateChangeReason', 'Unknown error'),
                    'athena_error': status_details.get('AthenaError', {}),
                    'query_id': query_id
                }
                error_msg = (
                    f"Query failed with state {state}.\n"
                    f"Reason: {error_info['reason']}\n"
                    f"Query ID: {query_id}"
                )
                if 'AthenaError' in status_details:
                    error_msg += f"\nAthena Error: {status_details['AthenaError']}"
                logger.error(error_msg)
                return state, None, error_msg
            
            # Query succeeded
            logger.info(f"Query {query_id} completed successfully")
            results = self.athena_client.get_query_results(QueryExecutionId=query_id)
            df = self._convert_results_to_df(results)
            logger.info(f"Query returned {len(df)} rows and {len(df.columns)} columns")
            return 'SUCCEEDED', df, None
                
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return 'ERROR', None, error_msg

    def save_dataset(self, df: pd.DataFrame, use_case_name: str) -> str:
        """Save a dataset to S3 and return its location"""
        logger.info(f"Saving dataset for use case: {use_case_name}")
        logger.debug(f"Dataset shape: {df.shape}")
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
                df.to_csv(tmp.name, index=False)
                s3_path = f'ml_datasets/{use_case_name}_dataset.csv'
                
                logger.info(f"Uploading dataset to s3://{self.s3_bucket_name}/{s3_path}")
                self.s3_client.upload_file(
                    tmp.name,
                    self.s3_bucket_name,
                    s3_path
                )
                
                location = f's3://{self.s3_bucket_name}/{s3_path}'
                logger.info(f"Dataset successfully saved to {location}")
                return location
                
        except Exception as e:
            logger.error(f"Error in save_dataset: {str(e)}", exc_info=True)
            return ''

    def _convert_results_to_df(self, query_results: Dict) -> pd.DataFrame:
        """Convert Athena query results to a pandas DataFrame"""
        try:
            columns = [col['Name'] for col in query_results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
            logger.debug(f"Converting query results with columns: {columns}")
            
            data = []
            for row in query_results['ResultSet']['Rows'][1:]:  # Skip header row
                data.append([item.get('VarCharValue', '') for item in row['Data']])
            
            df = pd.DataFrame(data, columns=columns)
            logger.debug(f"Converted results to DataFrame with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error in _convert_results_to_df: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def execute_and_save_query(self, query: str, use_case_name: str = None) -> Tuple[str, Optional[Dict]]:
        """Execute query, save full results, and return samples"""
        logger.info(f"Executing and saving query for {use_case_name if use_case_name else 'analysis'}")
        logger.debug(f"Query: {query}")
        
        try:
            status, df, error_msg = self.execute_query(query)
            
            if status == 'SUCCEEDED' and df is not None:
                result_info = {
                    "status": status,
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "columns": list(df.columns),
                    "sample_data": df.head(5).to_dict('records')  # Only return 5 sample rows
                }
                
                # Save the full dataset if we have a use case name
                if use_case_name and not df.empty:
                    dataset_location = self.save_dataset(df, use_case_name)
                    result_info["dataset_location"] = dataset_location
                    logger.info(f"Saved full dataset ({len(df)} rows) to {dataset_location}")
                
                return status, result_info
            else:
                return status, {
                    "status": status,
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"Error in execute_and_save_query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return 'ERROR', {
                "status": 'ERROR',
                "error": error_msg
            }

def truncate_response(data: Any, max_size: int = 20000) -> Any:
    """Truncate response data to stay within size limits"""
    if isinstance(data, dict):
        serialized = json.dumps(data)
        if len(serialized) <= max_size:
            return data
            
        # For dictionary responses, try to preserve structure while reducing content
        truncated = data.copy()
        if 'data' in truncated and isinstance(truncated['data'], list):
            # Calculate approximate size per record
            record_count = len(truncated['data'])
            if record_count > 0:
                avg_record_size = len(json.dumps(truncated['data'])) / record_count
                # Calculate how many records we can keep
                safe_record_count = int((max_size * 0.8) / avg_record_size)  # 80% of max size
                truncated['data'] = truncated['data'][:safe_record_count]
                truncated['truncated'] = True
                truncated['total_records'] = record_count
                truncated['showing_records'] = safe_record_count
                return truncated
                
    elif isinstance(data, list):
        serialized = json.dumps(data)
        if len(serialized) <= max_size:
            return data
            
        # For list responses, truncate the list
        original_length = len(data)
        # Calculate approximate size per item
        if original_length > 0:
            avg_item_size = len(serialized) / original_length
            safe_item_count = int((max_size * 0.8) / avg_item_size)  # 80% of max size
            return {
                'data': data[:safe_item_count],
                'truncated': True,
                'total_items': original_length,
                'showing_items': safe_item_count
            }
    
    return data

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Extract parameters from requestBody if present
        parameters_dict = {}
        if 'requestBody' in event and 'content' in event['requestBody']:
            content = event['requestBody']['content']
            if 'application/json' in content and 'properties' in content['application/json']:
                for prop in content['application/json']['properties']:
                    parameters_dict[prop['name']] = prop['value']
        
        # Create Parameters object with the extracted values
        parameters = Parameters(**parameters_dict)

        # Initialize tools
        tools = BusinessAnalystTools(
            session=session,
            s3_bucket_name=S3_BUCKET_NAME,
            athena_database=parameters.AthenaDatabase
        )
        
        # Extract APIPath from event
        api_path = event.get('apiPath', '').strip('/')
        
        logger.info(f"API Path: {api_path}")
        logger.info(f"Parameters: {parameters}")

        response_data = None

        if api_path == 'GetDatabaseSchema':
            # Get database schema
            schema = tools.get_database_schema()
            response_data = json.loads(schema)  # Convert string to JSON array
            
        elif api_path == 'GetUseCases':
            # Get available use cases
            use_cases = tools.get_use_cases()
            response_data = use_cases
            
        elif api_path == 'ExecuteQuery':
            # Execute Athena query
            query = parameters_dict.get('Query')
            use_case_name = parameters_dict.get('UseCaseName')  # Optional parameter
            
            if not query:
                raise ValueError("Query parameter is required")
                
            status, result_info = tools.execute_and_save_query(query, use_case_name)
            if status != 'SUCCEEDED':
                # Instead of raising ValueError, return the error info directly
                logger.info(f"Query failed with info: {result_info}")
                response_data = result_info
            else:
                response_data = result_info
            
        elif api_path == 'SaveDataset':
            # This endpoint becomes optional since datasets are saved automatically
            # but keep it for explicit saves
            use_case_name = parameters_dict.get('UseCaseName')
            data = parameters_dict.get('Data')
            
            if not use_case_name or not data:
                raise ValueError("UseCaseName and Data parameters are required")
                
            df = pd.DataFrame(data)
            location = tools.save_dataset(df, use_case_name)
            response_data = {"location": location}
            
        else:
            raise ValueError(f"Unknown API path: {api_path}")
                
        # Check and truncate response size if needed
        response_size = sys.getsizeof(json.dumps(response_data))
        if response_size > 20000:  # 20KB limit
            logger.warning(f"Response size {response_size} exceeds limit. Truncating content...")
            response_data = truncate_response(response_data)
                
        response_body = {
            'application/json': {
                'body': response_data
            }
        }
        
        # Set response code based on status if it exists
        response_code = 200
        if isinstance(response_data, dict) and response_data.get('status') in ['FAILED', 'CANCELLED', 'ERROR']:
            response_code = 400

        action_response = {
            'actionGroup': event['actionGroup'],
            'apiPath': event['apiPath'],
            'httpMethod': event['httpMethod'],
            'httpStatusCode': response_code,
            'responseBody': response_body
        }

        return {'messageVersion': '1.0', 'response': action_response}
            
    except ValueError as e:
        # Handle bad request errors (400)
        logger.error(f"Validation error: {str(e)}")
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event.get('actionGroup'),
                "apiPath": event.get('apiPath'),
                "httpMethod": event.get('httpMethod', 'POST'),
                "httpStatusCode": 400,
                "responseBody": {
                    "application/json": {
                        "body": {
                            "error": str(e)
                        }
                    }
                }
            }
        }
    except Exception as e:
        # Handle internal server errors (500)
        logger.error(f"Internal error: {str(e)}", exc_info=True)
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event.get('actionGroup'),
                "apiPath": event.get('apiPath'),
                "httpMethod": event.get('httpMethod', 'POST'),
                "httpStatusCode": 500,
                "responseBody": {
                    "application/json": {
                        "body": {
                            "error": f"Internal server error: {str(e)}"
                        }
                    }
                }
            }
        }
