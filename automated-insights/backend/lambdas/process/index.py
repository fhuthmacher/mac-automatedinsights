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
import random


logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Message(TypedDict):
    sender: str
    message: str

class EventType(TypedDict):
    userId: str
    conversationId: str
    query: str

class EventResult(TypedDict):
    sender: str
    message: str

class MessageSystemStatus(str, Enum):
    NEW = 'NEW'
    PENDING = 'PENDING'
    PROCESSING = 'PROCESSING'
    COMPLETE = 'COMPLETE'
    ERROR = 'ERROR'

# Define the GraphQL mutation
send_message_chunk_mutation = """
mutation Mutation($userId: ID!, $conversationId: ID!, $status: ConversationStatus!, $chunkType: String!, $chunk: String!) {
  systemSendMessageChunk(input: {userId: $userId, conversationId: $conversationId, status: $status, chunkType: $chunkType, chunk: $chunk}) {
        status
        userId
        conversationId
        chunkType
        chunk
  }
}
"""

# read TABLE_NAME from environment variable
TABLE_NAME = os.getenv('TABLE_NAME')
TRACE_TABLE_NAME = os.getenv('TRACE_TABLE_NAME')
USER_FEEDBACK_TABLE_NAME = os.getenv('USER_FEEDBACK_TABLE_NAME')
USER_UPLOAD_BUCKET_NAME = os.getenv('USER_UPLOAD_BUCKET_NAME')
BUCKET_NAME = os.getenv('BUCKET_NAME')
ATHENA_DATABASE_NAME = os.getenv('ATHENA_DATABASE_NAME')
FLOW_ID = os.getenv('FLOW_ID')
FLOW_ALIAS_ID = os.getenv('FLOW_ALIAS_ID')
AGENT_ID = os.getenv('AGENT_ID')
AGENT_ALIAS_ID = os.getenv('AGENT_ALIAS_ID')
GRAPHQL_URL = os.getenv('GRAPHQL_URL', '')
REGION = os.getenv('AWS_REGION', 'us-east-1')


# Initialize DynamoDB client
dynamodb = boto3.client('dynamodb')

# Initialize Bedrock Agent Runtime client
bedrock_agent_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    config=Config(
        connect_timeout=900,
        read_timeout=900,  # 15 minutes
        retries={
            'max_attempts': 3,
            'mode': 'adaptive'
        }
    )
)

config = Config(
            retries = {
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )

s3_client = boto3.client('s3')

session = boto3.Session()


def send_request(query: str, variables: dict) -> dict:
    """Send a request to AppSync using IAM auth"""
    if not GRAPHQL_URL:
        raise ValueError('GRAPHQL_URL is missing. Aborting operation.')

    session = boto3.Session()
    credentials = session.get_credentials()
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }
    
    payload = {
        'query': query,
        'variables': variables
    }

    request = AWSRequest(
        method='POST',
        url=GRAPHQL_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers
    )

    SigV4Auth(credentials, 'appsync', REGION).add_auth(request)
    
    final_headers = dict(request.headers)
    
    http = urllib3.PoolManager()
    try:
        response = http.request(
            'POST',
            GRAPHQL_URL,
            headers=final_headers,
            body=request.body
        )
        
        if response.status != 200:
            raise Exception(f"Request failed with status {response.status}")
            
        return json.loads(response.data.decode('utf-8'))
    except Exception as e:
        raise Exception(f"Failed to send request: {str(e)}")

def send_chunk(user_id: str, conversation_id: str, status: str = 'PROCESSING', chunk_type: str = 'text', chunk: str = '', citations: list = None) -> dict:
    """Send a message chunk to the conversation"""
    if citations is None:
        citations = []

    variables = {
        'userId': user_id,
        'conversationId': conversation_id,
        'status': status,
        'chunkType': chunk_type,
        'chunk': chunk,
        'citations': citations
    }

    return send_request(send_message_chunk_mutation, variables)



def update_conversation_status(
    user_id: str,
    conversation_id: str,
    status: MessageSystemStatus,
    table_name: str
) -> None:
    now = datetime.utcnow().isoformat()
    
    params = {
        'TableName': table_name,
        'Key': {
            'pk': {'S': f'USER#{user_id}'},
            'sk': {'S': f'CONVERSATION#{conversation_id}'}
        },
        'UpdateExpression': 'SET #status = :status, #updatedAt = :updatedAt',
        'ExpressionAttributeNames': {
            '#status': 'status',
            '#updatedAt': 'updatedAt'
        },
        'ExpressionAttributeValues': {
            ':status': {'S': status},
            ':updatedAt': {'S': now}
        }
    }

    dynamodb.update_item(**params)


def format_citation(citation: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to format a single citation for DynamoDB"""
    return {
        'M': {
            'content': {'S': citation['content']} if citation.get('content') else {'NULL': True},
            'imageContent': {'S': citation['imageContent']} if citation.get('imageContent') else {'NULL': True},
            'imageUrl': {'S': citation['imageUrl']} if citation.get('imageUrl') else {'NULL': True},
            'location': {
                'M': {
                    'type': {'S': citation['location']['type']} if citation.get('location', {}).get('type') else {'NULL': True},
                    'uri': {'S': citation['location']['uri']} if citation.get('location', {}).get('uri') else {'NULL': True},
                    'url': {'S': citation['location']['url']} if citation.get('location', {}).get('url') else {'NULL': True}
                }
            } if citation.get('location') else {'NULL': True},
            'metadata': {
                'M': {
                    key: {'S': str(value)} 
                    for key, value in citation['metadata'].items()
                }
            } if citation.get('metadata') else {'NULL': True},
            'span': {
                'M': {
                    'start': {'N': str(citation['span']['start'])},
                    'end': {'N': str(citation['span']['end'])}
                }
            } if citation.get('span') else {'NULL': True}
        }
    }


def add_message(
    id: str,
    conversation_id: str,
    message: str,
    sender: str,
    table_name: str,
    citations: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Add a message to a conversation in DynamoDB
    
    Args:
        id: User ID
        conversation_id: Conversation ID
        message: Message content
        sender: Message sender
        table_name: DynamoDB table name
        citations: Optional list of citations
        
    Returns:
        Dict containing the updated attributes
    """
    now = datetime.utcnow().isoformat()
    citations = citations or []
    
    # Format citations for DynamoDB
    formatted_citations = [format_citation(citation) for citation in citations]

    params = {
        'TableName': table_name,
        'Key': {
            'pk': {'S': f'USER#{id}'},
            'sk': {'S': f'CONVERSATION#{conversation_id}'}
        },
        'UpdateExpression': 'SET #messages = list_append(if_not_exists(#messages, :empty_list), :message), #updatedAt = :updatedAt',
        'ExpressionAttributeNames': {
            '#messages': 'messages',
            '#updatedAt': 'updatedAt'
        },
        'ExpressionAttributeValues': {
            ':message': {'L': [{
                'M': {
                    'sender': {'S': sender},
                    'message': {'S': message},
                    'citations': {'L': formatted_citations},
                    'createdAt': {'S': now}
                }
            }]},
            ':empty_list': {'L': []},
            ':updatedAt': {'S': now}
        },
        'ReturnValues': 'ALL_NEW'
    }

    response = dynamodb.update_item(**params)
    return response.get('Attributes', {})



def save_trace_events(conversation_id: str, user_id: str, trace_events: list, table_name: str) -> None:
    """
    Save trace events to a DynamoDB table.

    Args:
        conversation_id: Unique identifier for the conversation.
        user_id: Unique identifier for the user.
        trace_events: List of trace events to save.
        table_name: Name of the DynamoDB table.
    """
    try:
        for event in trace_events:
            event_order = event.get("event_order")
            
            if event_order is None:
                logger.error("Event order is None, skipping event")
                continue

            # Prepare the item to be saved
            item = {
                'pk': {'S': conversation_id},  # Assuming 'pk' is the partition key
                'sk': {'S': f'{user_id}#{event_order}'},  # Incorporate event_order into sk
                'user_id': {'S': user_id},
                'event_order': {'N': str(event_order)},
                'event_data': {'S': json.dumps(event)}
            }
            
            # Save the item to DynamoDB
            dynamodb.put_item(TableName=table_name, Item=item)
        
        logger.info(f"Successfully saved {len(trace_events)} trace events to {table_name}.")
    
    except ClientError as e:
        logger.error(f"Failed to save trace events: {e.response['Error']['Message']}")
        raise

def save_user_feedback(conversation_id: str, user_id: str, feedbackType: str, comment: str, context: str, table_name: str) -> None:
    """
    Save user feedback to a DynamoDB table.

    Args:
        conversation_id: Unique identifier for the conversation.
        user_id: Unique identifier for the user.
        feedbackType: Type of feedback.
        comment: Comment from the user.
        context: Context of the feedback.
        table_name: Name of the DynamoDB table.
    """
    try:
        # Prepare the item to be saved
        item = {
            'pk': {'S': conversation_id},
            'sk': {'S': user_id},
            'feedbackType': {'S': feedbackType},
            'comment': {'S': comment},
            'context': {'S': context}
        }
            
        # Save the item to DynamoDB
        dynamodb.put_item(TableName=table_name, Item=item)
        
        logger.info(f"Successfully saved user feedback to {table_name}.")
        return "Success"
    except ClientError as e:
        logger.error(f"Failed to save user feedback: {e.response['Error']['Message']}")
        return "Error"

def retry_with_backoff(func, max_retries=5, initial_backoff=1, max_backoff=32):
    """Retry a function with exponential backoff."""
    retries = 0
    backoff = initial_backoff
    
    while retries < max_retries:
        try:
            return func()
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "DependencyFailedException" or error_code == "ThrottlingException":
                wait_time = backoff + random.uniform(0, 1)
                logger.info(f"Request failed with {error_code}, retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                retries += 1
                backoff = min(backoff * 2, max_backoff)
            else:
                # If it's not a retryable error, re-raise it
                raise
    
    # If we've exhausted our retries, raise the last exception
    raise Exception(f"Failed after {max_retries} retries")

def process_single_event(event: EventType) -> EventResult:
    """Process a single event through Bedrock Agent Flow
    
    Args:
        event: Event containing user query and context
        
    Returns:
        EventResult containing the assistant's response
    """



    # Initialize variables
    feedbackType = None
    comment = None
    useCase = None
    files = None
    action = None
    prompt = 'Do nothing'

    query = event.get('query')
    conversation_id = event.get('conversationId')
    logger.info(f"Conversation ID: {conversation_id}")
    user_id = event.get('userId')
    logger.info(f"User ID: {user_id}")

    # First check if the query is in the "Processing files:" format
    if isinstance(query, str) and query.startswith('Processing files:'):
        files = [f.strip() for f in query.replace('Processing files:', '').split(',') if f.strip()]
        action = 'upload'
        logger.info(f"Detected file upload request with files: {files}")
    # If not in the "Processing files:" format, try to parse as JSON
    elif isinstance(query, str):
        try:
            logger.info(f"Parsing Query: {query}")
            json_query = json.loads(query)
            logger.info(f"Parsed JSON Query: {json_query}")
            feedbackType = json_query.get('feedbackType', '')
            comment = json_query.get('comment', '')
            useCase = json_query.get('useCase', '')
            files = json_query.get('files', '')
            action = json_query.get('action', '')
        except json.JSONDecodeError as e:
            logger.info(f"Failed to parse query as JSON: {str(e)}")
            # Handle plain text queries by setting a default action
            if "discover" in query.lower() or "database" in query.lower() or "use case" in query.lower():
                action = 'discover'
                logger.info(f"Detected discover request from plain text: {query}")
            else:
                # Default to using the agent with the query as the prompt
                prompt = query
                logger.info(f"Using plain text query as prompt: {prompt}")


    if action == 'feedback':
        # save query to dynamodb table
        result = save_user_feedback(
            conversation_id,
            user_id,
            feedbackType,
            comment,
            useCase,
            USER_FEEDBACK_TABLE_NAME
        )
        return {
            'message': result,
            'citations': []
        }
    
    else:
        # Set default prompt if not already set
        if prompt == 'Do nothing' and action is None:
            prompt = query

        if action == 'discover':
            prompt = f'''Inputs:
                    1. Here is the Athena database name:  "{ATHENA_DATABASE_NAME}"
                    Output:
                    Return the final response in XML format and nothing else.'''
            
        elif action == 'get_entity_relationship':
            # retrieve json from s3
            s3_client = boto3.client('s3')
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key='metadata/er_diagram/erm.json')
            entity_relationship = json.loads(response['Body'].read().decode('utf-8'))
            
            # Convert dictionary to JSON string before returning
            return {
                'message': json.dumps(entity_relationship),
                'citations': []
            }
        
        elif action == 'upload' and files is not None:
            # No need to check for "Processing files:" format here since we already handled it above
            logger.info(f"Processing file(s): {files}")
            
            # Iterate through all files and concatenate them into a single array that will be used for the prompt
            all_data_locations = []
            for file_name in files:
                data_location = f"s3://{USER_UPLOAD_BUCKET_NAME}/public/{file_name}"
                all_data_locations.append(data_location)
                
            # convert all_data_locations to a string
            all_data_locations_string = ', '.join(all_data_locations)
                
            prompt = f'''Inputs:
                1. Here is the Athena database name:  "{ATHENA_DATABASE_NAME}"
                2. And here is the list of all files that need to be processed: {all_data_locations_string}
                
                Output:
                Return the final response in XML format and nothing else.'''
        
                
        logger.info(f"Using prompt: {prompt}")
        try:
            if FLOW_ID != 'flow-id' and FLOW_ALIAS_ID != 'flow-alias-id':
                logger.info(f"Invoking flow: {FLOW_ID} with alias: {FLOW_ALIAS_ID}")
        
                # Call Bedrock Flow with the query
                response = bedrock_agent_runtime.invoke_flow(
                    flowIdentifier=FLOW_ID,
                    flowAliasIdentifier=FLOW_ALIAS_ID,
                    inputs=[
                        {
                            "content": { 
                                "document": query
                            },
                            "nodeName": "FlowInputNode",
                            "nodeOutputName": "document"
                        }
                    ]
                )

                # Process the response stream
                result = {
                    'message': '',
                }

                for event in response.get("responseStream"):
                    if 'flowOutputEvent' in event:
                        content = event['flowOutputEvent']
                        node_name = content.get('nodeName', '')
                        
                        if 'content' in content and 'document' in content['content']:
                            document_content = content['content']['document']
                            
                            # Route content based on node name
                            if node_name == 'Predictions':
                                result['Predictions'] = document_content
                            elif node_name == 'UseCases':
                                result['UseCases'] = document_content
                            elif node_name == 'FeatureImportance':
                                result['FeatureImportance'] = document_content

                # Combine all results into a formatted message
                formatted_message = ""
                if result['Predictions']:
                    formatted_message += f"Predictions:\n{result['Predictions']}\n\n"
                if result['UseCases']:
                    formatted_message += f"UseCases:\n{result['UseCases']}\n\n"
                if result['FeatureImportance']:
                    formatted_message += f"FeatureImportance:\n{result['FeatureImportance']}"

                return {
                    'message': formatted_message.strip(),
                    'citations': []
                }
            
            if AGENT_ID != 'agent-id' and AGENT_ALIAS_ID != 'agent-alias-id':
                try:
                    logger.info(f"Invoking agent: {AGENT_ID} with alias: {AGENT_ALIAS_ID}")
                    
                    def invoke_agent_call():
                        return bedrock_agent_runtime.invoke_agent(
                            agentId=AGENT_ID,
                            agentAliasId=AGENT_ALIAS_ID,
                            inputText=prompt,
                            sessionId=conversation_id,
                            enableTrace=True,
                        )
                    
                    # Use retry logic for the agent invocation
                    response = retry_with_backoff(invoke_agent_call)

                    # Process and concatenate the response from the agent
                    completion = ""
                    trace_events = []
                    index = 1
                    try:
                        for event in response.get("completion", []):
                            chunk = event.get("chunk", {})
                            if chunk and "bytes" in chunk:
                                completion += chunk["bytes"].decode()
                            
                            trace = event.get("trace", {})
                            if trace:
                                trace["event_order"] = index
                                trace_events.append(trace)
                                index += 1

                        save_trace_events(conversation_id, user_id, trace_events, TRACE_TABLE_NAME)


                    
                        logger.info(f"Completion: {completion}")
                        return {
                            'message': completion,
                            'citations': []
                        }
                    
                    except Exception as chunk_error:
                        logger.error(f"Error processing chunks: {str(chunk_error)}")
                        error_message = str(chunk_error).lower()
                        
                        # Handle specific error cases
                        if "throttlingexception" in error_message:
                            return {
                                'message': "The service is currently experiencing high demand. Please wait a moment and try again.",
                                'citations': []
                            }
                        if "timeout" in error_message or "badgatewayexception" in error_message:
                            return {
                                'message': "The request took too long to process. Please try breaking your request into smaller parts or try again later.",
                                'citations': []
                            }
                        
                        # Return partial completion if we have any
                        if completion:
                            logger.warning("Returning partial completion due to chunk processing error")
                            return {
                                'message': completion + "\n\nNote: Response may be incomplete due to processing error.",
                                'citations': []
                            }
                        raise chunk_error

                    except Exception as agent_error:
                        logger.error(f"Agent invocation error: {str(agent_error)}")
                        error_message = str(agent_error).lower()
                        
                        if "throttlingexception" in error_message:
                            return {
                                'message': "The service is currently experiencing high demand. Please wait a moment and try again.",
                                'citations': []
                            }
                        if "timeout" in error_message:
                            return {
                                'message': "The request took too long to process. Please try with a shorter input or try again later.",
                                'citations': []
                            }
                        raise agent_error

                except Exception as e:
                    logger.error(f"Error in process_single_event: {str(e)}", exc_info=True)
                    error_message = str(e)
                    if "timeout" in error_message.lower():
                        return {
                            'message': "The request took too long to process. Please try with a shorter input or try again later.",
                            'citations': []
                        }
                    raise Exception(f"Failed to process event: {error_message}")

        except Exception as e:
            logger.error(f"Error in process_single_event: {str(e)}", exc_info=True)
            error_message = str(e)
            if "timeout" in error_message.lower():
                return {
                    'message': "The request took too long to process. Please try with a shorter input or try again later.",
                    'citations': []
                }
            raise Exception(f"Failed to process event: {error_message}")

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    logger.info(f"Received context: {json.dumps(context.__dict__, default=str, indent=2)}")
    logger.info(f"Received event: {json.dumps(event, default=str, indent=2)}")
    # Extract event data
    identity = event.get('identity', {})
    prev = event.get('prev', {})
    input_data = event.get('arguments', {}).get('input', {})
    prompt = input_data.get('prompt')

    # Condition 1: User is authenticated
    user_id = identity.get('sub')
    if not user_id:
        raise Exception('Missing identity')

    # Condition 2: Check if conversation is currently processing
    prev_result = prev.get('result', {})
    prev_status = prev_result.get('status')
    if not user_id or (prev_status and prev_status in 
                      [MessageSystemStatus.PENDING.value, MessageSystemStatus.PROCESSING.value]):
        raise Exception('Conversation is currently processing')

    # Condition 3: Check conversation ID
    conversation_id = prev_result.get('sk')
    if not conversation_id:
        raise Exception('That conversation does not exist')
    conversation_id = conversation_id.split('#')[1]

    try:
        # Start processing status update
        update_conversation_status(
            user_id=user_id,
            conversation_id=conversation_id,
            status=MessageSystemStatus.PENDING.value,
            table_name=TABLE_NAME
        )

        # Add the user's message to the conversation
        add_message(
            id=user_id,
            conversation_id=conversation_id,
            message=prompt,
            sender="User",
            table_name=TABLE_NAME
        )

        result = process_single_event({
            'userId': user_id,
            'conversationId': conversation_id,
            'query': prompt,
        })

        logger.info(f"Result: {result}")
        
        # Check if result is None or doesn't have required fields
        if not result or not isinstance(result, dict):
            logger.error(f"Invalid result received: {result}")
            result = {
                'message': "Sorry, there was an error processing your request. Please try again.",
                'citations': []
            }

        send_chunk(
            user_id=user_id,
            conversation_id=conversation_id,
            status=MessageSystemStatus.PROCESSING.value,
            chunk_type='text',
            chunk=result.get('message', ''),
            citations=result.get('citations', [])
        )

        # Add the agent's response to the conversation
        add_message(
            id=user_id,
            conversation_id=conversation_id,
            message=result.get('message', ''),
            sender="Assistant",
            citations=result.get('citations', []),
            table_name=TABLE_NAME
        )

        # Update status to complete
        update_conversation_status(
            user_id=user_id,
            conversation_id=conversation_id,
            status=MessageSystemStatus.COMPLETE.value,
            table_name=TABLE_NAME
        )

        return {
            "message": {
                "sender": "Assistant",
                "message": result.get('message', ''),
                "createdAt": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
        }
    
    except Exception as error:
        logger.error(f'Processing error: {str(error)}', exc_info=True)
        update_conversation_status(
            user_id=user_id,
            conversation_id=conversation_id,
            status=MessageSystemStatus.ERROR.value,
            table_name=TABLE_NAME
        )
        raise Exception('An error occurred while processing your request')
