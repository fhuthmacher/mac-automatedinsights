import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
import os
import logging
import json
from enum import Enum
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, TypedDict, Optional
from urllib.parse import urlparse
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
import urllib3
from pydantic import BaseModel
from utils.bedrock import BedrockLLMWrapper
from boto3.dynamodb.conditions import Key
import sys

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

TRACE_TABLE_NAME = os.getenv('TRACE_TABLE_NAME')
USER_FEEDBACK_TABLE_NAME = os.getenv('USER_FEEDBACK_TABLE_NAME')
MODEL_ID = os.getenv('MODEL_ID')

LESSONS_LEARNED_PROMPT_TEMPLATE = '''Analyze the following information:

user feedback:
{USER_FEEDBACK}

traces from past runs:
{PAST_RUNS}

1. Review the user feedback and traces from past runs.

2. Extract the lessons learned with regards to agent orchestration and function/tool calling based on the past runs and user feedback and return them in a list of strings.

Sample output:
- when creating a SQL query that requires functions, ensure you have the correct Athena function names, e.g. DATE_DIFF instead of datediff
- when calling the data scientist agent, ensure you provide the ml dataset location and target column name, otherwise don't call the data scientist agent
'''



class SupervisorTools:
    """Collection of tools for the Supervisor Agent to use"""
    
    def __init__(self):
        logger.info(f"Initializing SupervisorTools")
        self.dynamodb = boto3.client('dynamodb')
        self.bedrock_llm = BedrockLLMWrapper(model_id=MODEL_ID, 
                            max_token_count=2000,
                            temperature=0
                        )
    
    def get_user_feedback(self, user_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve user feedback from DynamoDB"""
        try:
            response = self.dynamodb.get_item(
                TableName=USER_FEEDBACK_TABLE_NAME,
                Key={'user_id': {'S': user_id}, 'conversation_id': {'S': conversation_id}}
            )
            return response.get('Item', {}).get('feedback', [])
        except ClientError as e:
            logger.error(f"Error retrieving user feedback: {e}")
            return []
    
    
    def get_all_user_feedback(self) -> List[Dict[str, Any]]:
        """Retrieve all user feedback from DynamoDB"""
        try:
            response = self.dynamodb.scan(TableName=USER_FEEDBACK_TABLE_NAME)
            return response.get('Items', [])
        except ClientError as e:
            logger.error(f"Error retrieving all user feedback: {e}")
            return []

    def get_all_traces_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all traces for a user from DynamoDB"""
        try:
            response = self.dynamodb.scan(TableName=TRACE_TABLE_NAME,
                                         FilterExpression=Key('user_id').eq(user_id))
            return response.get('Items', [])
        except ClientError as e:
            logger.error(f"Error retrieving all traces for user {user_id}: {e}")
            return []
    
    def get_all_conversation_traces(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve all traces for a conversation from DynamoDB"""
        try:
            response = self.dynamodb.scan(TableName=TRACE_TABLE_NAME,
                                         FilterExpression=Key('conversation_id').eq(conversation_id))
            return response.get('Items', [])
        except ClientError as e:
            logger.error(f"Error retrieving all traces for conversation {conversation_id}: {e}")
            return []
    
    def get_all_traces(self) -> List[Dict[str, Any]]:
        """Retrieve all traces from DynamoDB"""
        try:
            response = self.dynamodb.scan(TableName=TRACE_TABLE_NAME)
            return response.get('Items', [])
        except ClientError as e:
            logger.error(f"Error retrieving all traces: {e}")
            return []

    def get_lessons_learned_from_past_runs(self):
        """Get lessons learned from past runs and user feedback"""
        try:
            user_feedback = self.get_all_user_feedback()
            past_runs = self.get_all_traces()
            # convert to a string
            past_runs_string = '\n'.join([str(run) for run in past_runs])
            user_feedback_string = '\n'.join([str(feedback) for feedback in user_feedback])

            prompt = LESSONS_LEARNED_PROMPT_TEMPLATE.format(USER_FEEDBACK=user_feedback_string, PAST_RUNS=past_runs_string)
            response = self.bedrock_llm.generate(prompt)
            return response[0]
        except Exception as e:
            logger.error(f"Error getting lessons learned from past runs: {e}", exc_info=True)
            return f'Error encountered while getting lessons learned from past runs and user feedback: {e}'



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
        
        # # Extract parameters from requestBody if present
        # parameters_dict = {}
        # if 'requestBody' in event and 'content' in event['requestBody']:
        #     content = event['requestBody']['content']
        #     if 'application/json' in content and 'properties' in content['application/json']:
        #         for prop in content['application/json']['properties']:
        #             parameters_dict[prop['name']] = prop['value']
        

        # Initialize tools
        tools = SupervisorTools(
        )
        
        # Extract APIPath from event
        api_path = event.get('apiPath', '').strip('/')
        
        logger.info(f"API Path: {api_path}")

        response_data = None
        
        if api_path == 'GetLessonsLearnedFromPastRuns':
            lessons_learned = tools.get_lessons_learned_from_past_runs()
            response_data = lessons_learned
            
        else:
            raise ValueError(f"Invalid API path: {api_path}")
        
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
