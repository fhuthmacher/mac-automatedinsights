import json
import sys
import logging
import pandas as pd
import boto3
import os
import zipfile
from urllib.parse import urlparse
import time
from io import StringIO
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import re

class RequestType(str, Enum):
    GET_INFORMATION_FOR_SEMANTIC_TYPE_DETECTION = "/GetInformationForSemanticTypeDetection"
    SAVE_SQL_TABLE_DEFINITION = "/SaveSQLTableDefinition"
    CREATE_ATHENA_TABLE = "/CreateAthenaTable"
    QUERY_DATA = "/QueryData"
    GET_DATABASE_SCHEMA = "/GetDatabaseSchema"
    GET_ERM = "/GetERM"
    SAVE_ERM = "/SaveERM"



class APIResponse(BaseModel):
    message: str
    results: Dict[str, Any]

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

if 'REGION' not in globals():
    REGION = os.getenv('REGION')
    logger.info(f"REGION: {REGION}")

if 'ATHENA_QUERY_EXECUTION_LOCATION' not in globals():
    ATHENA_QUERY_EXECUTION_LOCATION = f's3://{S3_BUCKET_NAME}/athena_results/'
    logger.info(f"ATHENA_QUERY_EXECUTION_LOCATION: {ATHENA_QUERY_EXECUTION_LOCATION}")

session = boto3.Session(region_name=REGION)

s3_client = session.client('s3')
athena_client = session.client('athena')

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



def parse_json(json_string):
    if not json_string:  # Handle None or empty string
        logger.warning("Received empty JSON string")
        return {
            'semantic_column_name': 'unknown',
            'column_description': 'No response from LLM',
            'data_type': 'unknown',
            'usecases': []
        }
    try:
        # First try to clean up any leading/trailing whitespace
        json_string = json_string.strip()
        
        # Remove any text before the first '{'
        if '{' in json_string:
            json_string = json_string[json_string.find('{'):]
            
        # Remove any text after the last '}'
        if '}' in json_string:
            json_string = json_string[:json_string.rfind('}')+1]
            
        # Try to parse the cleaned JSON string
        parsed = json.loads(json_string)
        
        # Convert old semantic_type key to semantic_column_name if needed
        if 'semantic_type' in parsed and 'semantic_column_name' not in parsed:
            parsed['semantic_column_name'] = parsed.pop('semantic_type')
            
        return parsed
        
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        logger.error(f"Problematic JSON string: {json_string}")
        return {
            'semantic_column_name': 'unknown',
            'column_description': 'No response from LLM',
            'data_type': 'unknown',
            'usecases': []
        }

def execute_athena_query(database, query):
    logger.info("Executing Athena query...")
    # Start query execution
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': database
        },
        ResultConfiguration={
            'OutputLocation': ATHENA_QUERY_EXECUTION_LOCATION
        }
    )

    # Get query execution ID
    query_execution_id = response['QueryExecutionId']
    print(f"Query Execution ID: {query_execution_id}")

    # Wait for the query to complete
    response_wait = athena_client.get_query_execution(QueryExecutionId=query_execution_id)

    while response_wait['QueryExecution']['Status']['State'] in ['QUEUED', 'RUNNING']:
        print("Query is still running...")
        response_wait = athena_client.get_query_execution(QueryExecutionId=query_execution_id)

    print(f'response_wait {response_wait}')

    # Check if the query completed successfully
    if response_wait['QueryExecution']['Status']['State'] == 'SUCCEEDED':
        print("Query succeeded!")

        # Get query results
        query_results = athena_client.get_query_results(QueryExecutionId=query_execution_id)

        # Extract and return the result data
        code = 'SUCCEEDED'
        return code, extract_result_data(query_results)

    else:
        print("Query failed!")
        code = response_wait['QueryExecution']['Status']['State']
        message = response_wait['QueryExecution']['Status']['StateChangeReason']
    
        return code, message

def extract_result_data(query_results):
    # Return a cleaned response to the agent
    result_data = []

    # Extract column names
    column_info = query_results['ResultSet']['ResultSetMetadata']['ColumnInfo']
    column_names = [column['Name'] for column in column_info]

    # Extract data rows
    for row in query_results['ResultSet']['Rows']:
        # Handle different data types in the response
        data = []
        for item in row['Data']:
            # Each item may contain different field based on the data type
            value = None
            if 'VarCharValue' in item:
                value = item['VarCharValue']
            elif 'NumberValue' in item:
                value = str(item['NumberValue'])  # Convert to string for consistency
            else:
                value = str(item)  # Fallback to string representation
            data.append(value)
            
        result_data.append(dict(zip(column_names, data)))

    return result_data


def create_sql_table_definition(sql_table_definition, s3_file_location) -> APIResponse:
    logger.info("Creating SQL table definition...")
    
    try:
        # Create a writable directory
        temp_dir = "/tmp/sql_table_definition"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Set the current working directory to the temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)

        # save sql_table_definition to s3 based on s3_file_location
        s3_url = urlparse(s3_file_location)
        bucket = s3_url.netloc
        key = s3_url.path.lstrip('/')  # Remove leading slash
        # extract the filename from the key
        filename = key.split('/')[-1]

        with open(filename, 'w') as f:
            f.write(sql_table_definition)

        s3_client.upload_file(
            filename,
            Bucket=bucket,
            Key=key
        )
        
        return APIResponse(
            message="SQL table definition successful",
            results={
                'sql_table_definition': sql_table_definition,
                'sql_table_definition_file_location': s3_file_location
            }
        )
    
    except Exception as e:
        logger.error(f"Error in create_sql_table_definition function: {str(e)}")
        raise


def get_database_schema() -> str:
        """Retrieve the SQL database schema from S3"""
        schema_prefix = 'metadata/sql_table_definition'
        logger.info(f"Retrieving database schema from s3://{S3_BUCKET_NAME}/{schema_prefix}")
        
        sql_database_schema = []
        try:
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET_NAME, 
                Prefix=schema_prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No schema files found in s3://{S3_BUCKET_NAME}/{schema_prefix}")
                return "[]"
            
            logger.info(f"Found {len(response['Contents'])} schema files")
            
            for item in response['Contents']:
                if item['Key'].endswith('/'):
                    continue
                    
                logger.info(f"Reading schema file: {item['Key']}")
                try:
                    content = s3_client.get_object(
                        Bucket=S3_BUCKET_NAME, 
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

def get_erm_schema() -> str:
        """Retrieve the Entity Relationship Model (ERM) from S3"""
        erm_prefix = 'metadata/er_diagram'
        logger.info(f"Retrieving ERM from s3://{S3_BUCKET_NAME}/{erm_prefix}")
        
        erm_schemas = []
        try:
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET_NAME, 
                Prefix=erm_prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No ERM files found in s3://{S3_BUCKET_NAME}/{erm_prefix}")
                return "[]"
            
            logger.info(f"Found {len(response['Contents'])} ERM files")
            
            for item in response['Contents']:
                if item['Key'].endswith('/'):
                    continue
                    
                logger.info(f"Reading ERM file: {item['Key']}")
                try:
                    content = s3_client.get_object(
                        Bucket=S3_BUCKET_NAME, 
                        Key=item['Key']
                    )['Body'].read().decode('utf-8')
                    # Verify it's valid JSON before adding
                    json_content = json.loads(content)
                    erm_schemas.append(json_content)
                    logger.debug(f"Successfully read ERM from {item['Key']}")
                except json.JSONDecodeError:
                    logger.error(f"File {item['Key']} contains invalid JSON")
                except Exception as e:
                    logger.error(f"Error reading ERM file {item['Key']}: {str(e)}")
            
            logger.info(f"Successfully retrieved {len(erm_schemas)} ERM schemas")
            return json.dumps(erm_schemas)
            
        except Exception as e:
            logger.error(f"Error in get_erm_schema: {str(e)}", exc_info=True)
            return "[]"

def save_erm_schema(erm_data) -> APIResponse:
    """Save the Entity Relationship Model (ERM) to S3
    
    Args:
        erm_data: The ERM data to save
        
    Returns:
        APIResponse: Response with status and saved ERM location
    """
    try:
        # Create a writable directory
        temp_dir = "/tmp/erm"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Set the current working directory to the temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        

        filename = f'erm.json'
        filepath = os.path.join(temp_dir, filename)
        
        # Save the ERM data to a file
        with open(filepath, 'w') as f:
            if isinstance(erm_data, str):
                f.write(erm_data)
            else:
                json.dump(erm_data, f, indent=2)
        
        # Upload the file to S3
        s3_key = f'metadata/er_diagram/{filename}'
        s3_client.upload_file(
            filepath,
            Bucket=S3_BUCKET_NAME,
            Key=s3_key
        )
        
        s3_location = f's3://{S3_BUCKET_NAME}/{s3_key}'
        logger.info(f"Saved ERM to {s3_location}")
        
        # Change back to original directory
        os.chdir(original_dir)
        
        return APIResponse(
            message="ERM saved successfully",
            results={
                'erm_file_location': s3_location
            }
        )
    except Exception as e:
        logger.error(f"Error in save_erm_schema: {str(e)}", exc_info=True)
        raise


def query_athena_table(athena_database, sql_query) -> APIResponse:
    logger.info("Querying Athena table...")
    try:

        # execute sql query
        status_code, response_data = execute_athena_query(athena_database, sql_query)
        logger.info(f"Athena query execution response: {response_data}")
        logger.info(f"status_code: {status_code}")
        if status_code == 'SUCCEEDED':
            return APIResponse(
                message=f"Query execution {status_code}",
                results={
                    'status': status_code,
                    'query': sql_query,
                    'data': response_data if isinstance(response_data, dict) else {'results': response_data}
                }
            )
        else:
            return APIResponse(
                message=f"Query execution failed with status: {status_code}",
                results={
                    'status': status_code,
                    'query': sql_query,
                    'error': response_data if isinstance(response_data, str) else str(response_data)
                }
            )

    except Exception as e:
        logger.error(f"Error in query_athena_table function: {str(e)}")
        raise

def check_athena_table_exists(database, table_name):
    """
    Check if a table exists in Athena
    
    Args:
        database (str): The name of the database
        table_name (str): The name of the table to check
        
    Returns:
        bool: True if table exists, False otherwise
    """
    logger.info(f"Checking if table {table_name} exists in database {database}")
    try:
        # Try to get table metadata
        response = athena_client.get_table_metadata(
            CatalogName='AwsDataCatalog',
            DatabaseName=database,
            TableName=table_name
        )
        logger.info(f"get_table_metadata response: {response}")
        logger.info(f"Table {table_name} exists in database {database}")
        return True
    except athena_client.exceptions.MetadataException:
        logger.info(f"Table {table_name} does not exist in database {database}")
        return False
    except Exception as e:
        logger.error(f"Error checking if table exists: {str(e)}")
        return False
    

def parse_request_parameters(event):
    """Parse request parameters from the Lambda event"""
    parameters = {}
    
    # Extract parameters from requestBody if present
    if event.get('requestBody') and event['requestBody'].get('content'):
        content = event['requestBody']['content']
        if 'application/json' in content and 'properties' in content['application/json']:
            for prop in content['application/json']['properties']:
                parameters[prop['name']] = prop['value']
    
    return parameters

def lambda_handler(event, context):
    try:
        # Create base temp directories at the start
        os.makedirs("/tmp/data", exist_ok=True)
        os.makedirs("/tmp/metadata", exist_ok=True)
        
        logger.info(f"Received event: {json.dumps(event)}")
        
        parameters = parse_request_parameters(event)
        logger.info(f"parameters: {parameters}")

        request_type = event.get('apiPath')
        logger.info(f"request_type: {request_type}")
        
        
        if parameters.get('DataLocation') is not None and parameters.get('DataLocation') != "":
            logger.info(f"Downloading data from {parameters.get('DataLocation')}")

            # Ensure data directory exists
            os.makedirs("/tmp/data", exist_ok=True)

            # Clear the /tmp/data directory before processing new file
            for file in os.listdir("/tmp/data"):
                os.remove(os.path.join("/tmp/data", file))

            # split data_location into bucket and key
            s3_url = urlparse(parameters.get('DataLocation'))
            data_bucket = s3_url.netloc
            logger.info(f"data_bucket: {data_bucket}")
            data_key = s3_url.path.lstrip('/')  # Remove leading slash
            logger.info(f"data_key: {data_key}")

            # Get just the filename from the path
            filename = os.path.basename(data_key)
            logger.info(f"filename: {filename}")
            # remove the extension from the filename
            file_or_table_name = os.path.splitext(filename)[0]
            logger.info(f"file_or_table_name: {file_or_table_name}")

            local_file_path = os.path.join("/tmp/data", filename)
            logger.info(f"local_file_path: {local_file_path}")
            # download the data from s3
            s3_client.download_file(
                Bucket=data_bucket,
                Key=data_key,
                Filename=local_file_path
            )

            # check if the data is zipped then unzip it
            if data_key.endswith(".zip"):
                logger.info(f"Unzipping data")
                with zipfile.ZipFile(local_file_path, "r") as zip_ref:
                    zip_ref.extractall("/tmp/data")
                logger.info(f"Unzipped data to /tmp/data")
            
            logger.info(f"Files in /tmp/data: {os.listdir('/tmp/data')}")

            # Load data from the first compatible file found
            data_loaded = False
            for file in os.listdir("/tmp/data"):
                logger.info(f"file: {file}")
                file_path = os.path.join("/tmp/data", file)
                logger.info(f"Processing file: {file_path}")
                ext = os.path.splitext(file)[1].lower()
                logger.info(f"File extension: {ext}")
                # set filename
                try:
                    if ext == '.csv':
                        df = pd.read_csv(file_path)
                        # remove the extension from the filename
                        file_or_table_name = os.path.splitext(file)[0]
                        
                    elif ext == '.json':
                        df = pd.read_json(file_path)
                        # remove the extension from the filename
                        file_or_table_name = os.path.splitext(file)[0]
                    elif ext == '.parquet':
                        df = pd.read_parquet(file_path)
                        # remove the extension from the filename
                        file_or_table_name = os.path.splitext(file)[0]
                    
                    if not df.empty:
                        logger.info(f"Successfully loaded data from {file_path}")
                        data_loaded = True
                        logger.info(f"file_or_table_name: {file_or_table_name}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {str(e)}")
                    continue

            if not data_loaded:
                raise ValueError("Could not load data from any files in the specified location")
            

        response_data = None
        s3_base_path = f's3://{S3_BUCKET_NAME}/'

        if request_type == RequestType.GET_DATABASE_SCHEMA:
            # Get database schema
            schema = get_database_schema()
            response_data = json.loads(schema)  # Convert string to JSON array
            response_body = {
                'application/json': {
                    'body': response_data
                }
            }
            response_data = APIResponse(
                message="Database schema retrieved successfully",
                results=response_body
            )


        if request_type == RequestType.GET_INFORMATION_FOR_SEMANTIC_TYPE_DETECTION:
            # Export sample data to make available for the ReAct agent
            sample_df = df.head(100)  # Use only a sample
            
            # Return information needed for the ReAct agent to perform semantic detection
            response_data = APIResponse(
                message="Data sample prepared for semantic type detection",
                results={
                    'column_names': list(df.columns),
                    'data_sample': sample_df.head(10).to_dict('records')
                }
            )
        
        if request_type == RequestType.SAVE_SQL_TABLE_DEFINITION:
            # Get the SQL table definition
            sql_table_definition = parameters.get('SQL_Table_Definition')
            table_name = parameters.get('TableName')
            
            if not table_name:
                # Try to extract table name from SQL definition
                table_match = re.search(r'CREATE\s+TABLE\s+(\w+)', sql_table_definition, re.IGNORECASE)
                if table_match:
                    table_name = table_match.group(1)
                else:
                    # Fallback to a default name if we can't extract it
                    table_name = f"table_{int(time.time())}"
                logger.info(f"Extracted or generated table name: {table_name}")
            
            if sql_table_definition:
                # Save SQL definition to S3
                s3_file_location = f'{s3_base_path}metadata/sql_table_definition/{table_name}_sql_table_definition.sql'
                s3_file_location = s3_file_location.lower()
                response_data = create_sql_table_definition(sql_table_definition, s3_file_location)
                
            else:
                response_data = APIResponse(
                    message="Missing SQL table definition",
                    results={
                        'error': "Missing SQL table definition"
                    }
                )

        if request_type == RequestType.CREATE_ATHENA_TABLE:
            table_name = parameters.get('TableName')
            athena_database = parameters.get('AthenaDatabase')
            athena_table_create_definition = parameters.get('Athena_Table_Create_SQL_statement')
            
            # For backward compatibility, check other possible parameter names
            if athena_table_create_definition is None:
                athena_table_create_definition = parameters.get('TableDefinition')
                if athena_table_create_definition is None:
                    athena_table_create_definition = parameters.get('Table_Definition')
            
            if athena_table_create_definition is None:
                logger.error("Missing table definition parameter")
                raise ValueError("Missing required parameter: Athena_Table_Create_SQL_statement")
                
            data_location = parameters.get('DataLocation')
            if not data_location:
                logger.error("Missing DataLocation parameter")
                raise ValueError("Missing required parameter: DataLocation")
                
            s3_target_file_location = f'{s3_base_path}raw/{table_name}/'
            
            # Get the filename from the data_location
            filename = os.path.basename(data_location)
            
            logger.info(f"Original data location: {data_location}")
            s3_target_file_location = s3_target_file_location.lower()
            logger.info(f"Target location: {s3_target_file_location}")
            
            # Update the LOCATION with the s3_target_file_location if it exists in the SQL
            if 'LOCATION' in athena_table_create_definition:
                # Extract the current location
                location_match = re.search(r"LOCATION\s+'([^']+)'", athena_table_create_definition)
                if location_match:
                    current_location = location_match.group(1)
                    logger.info(f"Current location in SQL: {current_location}")
                    # Replace with the new location
                    athena_table_create_definition = athena_table_create_definition.replace(
                        f"LOCATION '{current_location}'", 
                        f"LOCATION '{s3_target_file_location}'"
                    )
                    logger.info(f"Updated SQL with new location: {s3_target_file_location}")
            
            # Prepare and upload the data
            prepare_and_upload_data(df, athena_table_create_definition, s3_target_file_location)
            
            # Execute the provided table definition in Athena
            status_code, response_data = execute_athena_query(athena_database, athena_table_create_definition)
            
            response_data = APIResponse(
                message=f"Athena table creation {status_code}",
                results={
                    'status': status_code,
                    'query_results': response_data if isinstance(response_data, dict) else {'data': response_data}
                }
            )

        if request_type == RequestType.QUERY_DATA:
            sql_query = parameters.get('SQLQuery')
            athena_database = parameters.get('AthenaDatabase')
            response_data = query_athena_table(athena_database, sql_query)

        if request_type == RequestType.GET_ERM:
            # Get the ERM schema
            erm_schema = get_erm_schema()
            response_data = json.loads(erm_schema)  # Convert string to JSON array
            response_body = {
                'application/json': {
                    'body': response_data
                }
            }
            response_data = APIResponse(
                message="ERM schema retrieved successfully",
                results=response_body
            )
        
        if request_type == RequestType.SAVE_ERM:
            # Extract ERM data from request body
            erm_data = parameters.get('ERMData')
            
            if erm_data:
                response_data = save_erm_schema(erm_data)
            elif event.get('requestBody') and event['requestBody'].get('content'):
                content = event['requestBody']['content']
                if 'application/json' in content and 'body' in content['application/json']:
                    erm_data = content['application/json']['body']
                    response_data = save_erm_schema(erm_data)
                else:
                    response_data = APIResponse(
                        message="Missing ERM data in request body",
                        results={
                            'error': "Missing ERM data in request body"
                        }
                    )
            else:
                response_data = APIResponse(
                    message="Missing ERM data",
                    results={
                        'error': "Missing ERM data parameter or request body"
                    }
                )
                
        # Format successful response
        response_body = {
            'application/json': {
                'body': {
                    'message': str(response_data.message),
                    'results': str(response_data.results)
                }
            }
        }
        response_size = sys.getsizeof(json.dumps(response_body))
        MAX_RESPONSE_SIZE = 22000  # 22KB limit
        if response_size > MAX_RESPONSE_SIZE:
            logger.error(f"Response size {response_size} exceeds limit. Truncating content...")

        response_code = 200
        action_response = {
            'actionGroup': event['actionGroup'],
            'apiPath': event['apiPath'],
            'httpMethod': event['httpMethod'],
            'httpStatusCode': response_code,
            'responseBody': response_body
        }

        api_response = {'messageVersion': '1.0', 'response': action_response}
        logger.info(f"action_response: {api_response}")
        return api_response
            
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Stack trace: {sys.exc_info()}")
            
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event['actionGroup'],
                "apiPath": event['apiPath'],
                "httpMethod": event['httpMethod'],
                "httpStatusCode": 400 if isinstance(e, ValueError) else 500,
                "responseBody": {
                    'application/json': {
                        'body': {
                            "error": str(e),
                            "errorCode": "400" if isinstance(e, ValueError) else "500"
                        }
                    }
                }
            }
        }

def prepare_and_upload_data(df, sql_table_definition, s3_target_file_location):
    """
    Updates DataFrame columns to match SQL table definition and uploads to S3
    
    Args:
        df (pandas.DataFrame): The DataFrame to process
        sql_table_definition (str): SQL CREATE TABLE statement
        s3_target_file_location (str): S3 location to upload the processed file
        
    Returns:
        str: The S3 location where the file was uploaded
    """
    try:
        
        logger.info(f"Preparing data for upload to {s3_target_file_location}")
        
        # Extract column names from SQL table definition
        # This regex looks for column definitions in a CREATE TABLE statement
        # Updated to handle both regular and EXTERNAL tables
        column_pattern = r'CREATE\s+(EXTERNAL\s+)?TABLE\s+\w+\s*\((.*?)\).*?(?:LOCATION|$)'
        match = re.search(column_pattern, sql_table_definition, re.DOTALL | re.IGNORECASE)
        
        if not match:
            logger.error("Could not extract column definitions from SQL statement")
            logger.error(f"SQL statement: {sql_table_definition}")
            raise ValueError("Invalid SQL table definition format")
            
        # Group 2 contains the column definitions if EXTERNAL is present, otherwise group 1
        column_section = match.group(2)
        
        # Extract individual column names
        columns = []
        for line in column_section.split(','):
            # Extract the column name (first word in the line)
            column_match = re.search(r'^\s*(\w+)', line.strip())
            if column_match:
                columns.append(column_match.group(1))
        
        logger.info(f"Extracted columns from SQL definition: {columns}")
        
        if len(columns) == 0:
            logger.error("No columns extracted from SQL definition")
            raise ValueError("No columns found in SQL table definition")
            
        # Check if number of columns matches
        if len(columns) != len(df.columns):
            logger.warning(f"Column count mismatch: SQL has {len(columns)}, DataFrame has {len(df.columns)}")
            # We'll proceed anyway and rename the columns we have
        
        # Create a mapping of old column names to new column names
        # Use min length to avoid index errors if counts don't match
        column_mapping = {}
        for i in range(min(len(df.columns), len(columns))):
            column_mapping[df.columns[i]] = columns[i]
        
        # Rename the DataFrame columns
        df = df.rename(columns=column_mapping)
        logger.info(f"Renamed DataFrame columns to: {list(df.columns)}")
        
        # Parse S3 URL
        s3_url = urlparse(s3_target_file_location)
        bucket = s3_url.netloc
        key = s3_url.path.lstrip('/')
        
        # Create a temporary file
        temp_dir = "/tmp/processed_data"
        os.makedirs(temp_dir, exist_ok=True)
        
        # If s3_target_file_location is a directory (ends with /), append filename
        if s3_target_file_location.endswith('/'):
            filename = f"{os.path.basename(key.rstrip('/'))}_{int(time.time())}.csv"
            key = f"{key}{filename}"
            local_file_path = os.path.join(temp_dir, filename)
        else:
            local_file_path = os.path.join(temp_dir, os.path.basename(key))
        
        # Save DataFrame to CSV
        df.to_csv(local_file_path, index=False)
        logger.info(f"Saved processed data to {local_file_path}")
        
        # Upload to S3
        s3_client.upload_file(
            local_file_path,
            Bucket=bucket,
            Key=key
        )
        logger.info(f"Uploaded processed data to s3://{bucket}/{key}")
        
        return f"s3://{bucket}/{key}"
        
    except Exception as e:
        logger.error(f"Error in prepare_and_upload_data: {str(e)}")
        raise
