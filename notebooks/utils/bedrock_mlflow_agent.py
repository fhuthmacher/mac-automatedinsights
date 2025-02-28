import copy
import os
import uuid
from typing import Dict, List, Optional
import time
import random
import boto3
from botocore.exceptions import ClientError


import mlflow
from botocore.config import Config
from mlflow.entities import SpanType
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatResponse, ChatMessage, ChatParams, ChatChoice


class BedrockMultiAgentModel(ChatModel):
    def __init__(self):
        """
        Initializes the BedrockModel instance with placeholder values.

        Note:
            The `load_context` method cannot create new instance variables; it can only modify existing ones.
            Therefore, all instance variables should be defined in the `__init__` method with placeholder values.
        """
        self.brt = None
        self._main_bedrock_agent = None
        self._bedrock_agent_id = None
        self._bedrock_agent_alias_id = None
        self._inference_configuration = None
        self._agent_instruction = None
        self._model = None
        self._aws_region = None
        self._messages = []
        self._conversation_json_dict = None
        self._metrics = None
    def __getstate__(self):
        """
        Prepares the instance state for pickling.

        This method is needed because the `boto3` client (`self.brt`) cannot be pickled.
        By excluding `self.brt` from the state, we ensure that the model can be serialized and deserialized properly.
        """
        # Create a dictionary of the instance's state, excluding the boto3 client
        state = self.__dict__.copy()
        state["_session"] = None
        state["brt"] = None
        
        return state

    def __setstate__(self, state):
        """
        Restores the instance state during unpickling.

        This method is needed to reinitialize the `boto3` client (`self.brt`) after the instance is unpickled,
        because the client was excluded during pickling.
        """
        self.__dict__.update(state)
        self.brt = None
        self._session = None

    @staticmethod
    def retry_with_backoff(func, max_retries=5, initial_backoff=1, max_backoff=32):
        """Retry a function with exponential backoff."""
        retries = 0
        backoff = initial_backoff
        
        while retries < max_retries:
            try:
                return func()
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code == "dependencyFailedException" or error_code == "ThrottlingException":
                    wait_time = backoff + random.uniform(0, 1)
                    print(f"Request failed, retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                    backoff = min(backoff * 2, max_backoff)
                else:
                    # If it's not a retryable error, re-raise it
                    raise
        
        # If we've exhausted our retries, raise the last exception
        raise Exception(f"Failed after {max_retries} retries")

    # Wrap your bedrock client invoke_agent call with the retry function

    def invoke_agent_with_retry(self, session_id, formatted_input, max_retries=5):
        def _invoke_agent():
            return self.brt.invoke_agent(
                agentId=self._bedrock_agent_id,
                agentAliasId=self._bedrock_agent_alias_id,
                sessionId=session_id,
                inputText=self._get_agent_prompt(formatted_input),  # Apply the same formatting
                enableTrace=True,
                endSession=False
            )
        
        return self.retry_with_backoff(_invoke_agent, max_retries=max_retries)

    def load_context(self, context):
        """
        Initializes the Bedrock client with AWS credentials.

        Args:
            context: The MLflow context containing model configuration.

        Note:
            Dependent secret variables must be in the execution environment prior to loading the model;
            else they will not be available during model initialization.
        """
        self._main_bedrock_agent = context.model_config.get("agents", {}).get(
            "main", {}
        )
        self._bedrock_agent_id = self._main_bedrock_agent.get("bedrock_agent_id")
        self._bedrock_agent_alias_id = self._main_bedrock_agent.get(
            "bedrock_agent_alias_id"
        )
        self._inference_configuration = self._main_bedrock_agent.get(
            "inference_configuration"
        )
        self._agent_instruction = self._main_bedrock_agent.get("instruction")
        self._model = self._main_bedrock_agent.get("model")
        self._aws_region = self._main_bedrock_agent.get("aws_region")

        # Initialize the Bedrock client
        if "aws_profile" in context.model_config:
            # print(f"Using AWS profile: {context.model_config['aws_profile']}")
            # print(f"Using AWS region: {self._aws_region}")
            self._session = boto3.Session(
                profile_name=context.model_config["aws_profile"],
                region_name=self._aws_region
            )
        else:
            # print(f"Using Environment Variables")
            self._session = boto3.Session(
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
                region_name=self._aws_region
            )
        
        self.brt = self._session.client(
            service_name="bedrock-agent-runtime",
            config=Config(region_name=self._aws_region)
        )

    @staticmethod
    def create_conversation_json_dict(messages) -> dict:
        """
        Create a JSON file containing conversation trajectories.
        
        Args:
            messages (list): List of messages
        
        Returns:
            conversation_dict: Dictionary containing conversation trajectories
        """
        agent_groups = {}

        # Keep track of seen messages to avoid duplicates
        seen_messages = set()

        for msg in messages:
            source = msg.get('source')
            destination = msg.get('destination')
            content = msg.get('content')
            
            # Skip if source or destination is missing
            if not source or not destination:
                continue

            # Create a unique identifier for the message
            message_id = f"{source}:{destination}:{content}"
            
            # Skip if we've seen this message before
            if message_id in seen_messages:
                continue
                
            seen_messages.add(message_id)
                
            # Create group key from source-destination pair
            group_key = f"{source}"
            
            # Initialize group if it doesn't exist
            if group_key not in agent_groups:
                agent_groups[group_key] = []
                
            # Add message to appropriate group
            agent_groups[group_key].append({
                'role': msg.get('role'),
                'source': msg.get('source'),
                'destination': msg.get('destination'),
                'content': msg.get('content'),
                'actions': msg.get('actions'),
                'observation': msg.get('observation')
            })
        
        conversation_dict =  {
            "trajectories": agent_groups
        }
        return conversation_dict

    @staticmethod
    def create_message(
        role: Optional[str] = None,
        source: Optional[str] = None,
        destination: Optional[str] = None,
        content: Optional[str] = None,
        actions: Optional[List[Dict]] = None,
        observation: Optional[str] = None
    ) -> Dict:
        """
        Create a message dictionary with the specified fields.
        
        Args:
            role (str, optional): Role of the message sender
            source (str, optional): Source agent ID
            destination (str, optional): Destination agent ID
            content (str, optional): Message content
            actions (List[Dict], optional): List of actions
            observation (str, optional): Observation data
        
        Returns:
            Dict: Message dictionary
        """
        message = {
            "role": role,
            "source": source,
            "destination": destination,
            "content": content,
            "actions": actions,
            "observation": observation
        }
        return message

    @staticmethod
    def _extract_trace_groups(events):
        """
        Extracts trace groups from a list of events based on their trace IDs.

        Args:
            events (list): A list of event dictionaries.

        Returns:
            dict: A dictionary where keys are trace IDs and values are lists of trace items.
        """
        from collections import defaultdict

        trace_groups = defaultdict(list)

        def find_trace_ids(obj, original_trace, depth=0, parent_key=None):
            if depth > 5:
                return  # Stop recursion after 5 levels if no traceId has been found
            if isinstance(obj, dict):
                trace_id = obj.get("traceId")
                if trace_id:
                    # Include the parent key as the 'type'
                    item = {
                        "type": parent_key,
                        "data": obj,
                        "event_order": original_trace.get("trace", {}).get(
                            "event_order"
                        ),
                        "callerChain": original_trace.get("trace", {}).get("callerChain"),
                    }
                    trace_groups[trace_id].append(item)
                else:
                    for key, value in obj.items():
                        find_trace_ids(
                            value, original_trace, depth=depth + 1, parent_key=key
                        )
            elif isinstance(obj, list):
                for item in obj:
                    find_trace_ids(item, item, depth=depth + 1, parent_key=parent_key)

        find_trace_ids(events, {})
        return dict(trace_groups)


    def _get_final_response_with_trace(self,trace_id_groups: dict[str, list[dict]]):
        """
        Processes trace groups to extract the final response and create relevant MLflow spans.

        Args:
            trace_id_groups (dict): A dictionary of trace groups keyed by trace IDs.

        Returns:
            str: The final response text extracted from the trace groups.
            metrics: A dictionary containing the metrics for the trace groups.
        """

        # Initialize metric counters at the start
        metrics = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "num_agent_calls": 0,
            "num_tool_calls": 0,
            "num_kb_lookups": 0
        }

        trace_id_groups_copy = copy.deepcopy(trace_id_groups)

        # Check if we have any trace groups
        if not trace_id_groups_copy or not trace_id_groups_copy.values():
            return None, metrics

        pending_agent_outputs = {
                    "name": None,
                    "output": None,
                    "event_order": None,
                    "trace_id": None
        }
        
        for _trace_id, _trace_group in trace_id_groups_copy.items():
            trace_group = sorted(_trace_group, key=lambda tg: tg["event_order"])
            
            ## parse trace_group to mlflow traces
            def process_trace_group(self,trace_group):
                nonlocal pending_agent_outputs 
                trace_info = {
                    "rationale": None,
                    "inference_configuration": None,
                    "usage_metadata": None
                    }
                action_group_trace_info = {
                    "action_group_name": None,
                    "api_path": None,
                    "execution_type": None
                    }
                knowledge_base_trace_info = {
                    "knowledge_base_id": None,
                    "text": None
                    }
                

                # with mlflow.start_span(name="Supervisor-Agent", span_type=SpanType.AGENT) as supervisor_span:

                for _trace in trace_group:
                    # print(f"_trace: {_trace}")
                    type = _trace.get('type')
                    # print(f"type: {type}")
                    callerChain = _trace.get('callerChain',{})
                    agentAliasArn = callerChain[-1]["agentAliasArn"]
                    # print(f"callerChain: {callerChain}")

                    agent_type ="Sub-Agent" if len(callerChain) > 1 else "Supervisor-Agent"


                    if type == "modelInvocationInput":
                        trace_info["inference_configuration"]  = _trace.get('data',{}).get('inferenceConfiguration')
                        # print(f'inference_configuration: {trace_info["inference_configuration"] }')

                    if type == "modelInvocationOutput":
                        # print(f"modelInvocationOutput: {_trace}")
                        trace_info["usage_metadata"] = _trace.get('data', {}).get('metadata', {}).get('usage')  # This is correct
                        # print(f'usage metadata: {trace_info["usage_metadata"]}')  # Should now print: {'inputTokens': 1560, 'outputTokens': 267}
                        # update usage
                        metrics["prompt_tokens"] = metrics["prompt_tokens"] + trace_info["usage_metadata"].get('inputTokens')
                        metrics["completion_tokens"] = metrics["completion_tokens"] + trace_info["usage_metadata"].get('outputTokens')
                        metrics["total_tokens"] = metrics["total_tokens"] + trace_info["usage_metadata"].get('inputTokens') + trace_info["usage_metadata"].get('outputTokens')

                    if type == "rationale":
                        # rational for either sub-agent or action-group or kb call
                        trace_info["rationale"] = _trace.get('data',{}).get('text')
                        # print(f'rationale: {trace_info["rationale"]}')
                        # print(f"trace event_order: {_trace.get('event_order')}")

                        # print(f"checking pending_agent_outputs: {pending_agent_outputs}")
                        # check if we first want to output the pending_agent_outputs
                        if pending_agent_outputs["event_order"] is not None:
                            # print(f"pending_agent_outputs event_order: {pending_agent_outputs['event_order']}")
                            
                            # check if event_order of current trace is greater than pending_agent_outputs["event_order"]
                            if _trace.get('event_order') > pending_agent_outputs["event_order"]:
                                # print(f"outputting pending_agent_outputs")
                                with mlflow.start_span(
                                    name=f"{pending_agent_outputs['name']}_Output",
                                    attributes={"trace_attributes": trace_id_groups[_trace_id]},
                                    span_type=SpanType.AGENT
                                ) as agent_output_span:
                                    agent_output_span.set_outputs({"agentOutput": pending_agent_outputs["output"]})
                                
                                # add message
                                self._messages.append(
                                    self.create_message(
                                        role="Observation",
                                        source=agentAliasArn.split('/')[1],
                                        destination=agentAliasArn.split('/')[1],
                                        content=pending_agent_outputs["output"],
                                        actions=None,
                                        observation=pending_agent_outputs["output"]
                                    )
                                )
                                # reset pending_agent_outputs
                                pending_agent_outputs = {
                                    "name": None,
                                    "output": None,
                                    "event_order": None,
                                    "trace_id": None
                                }

                        
                        with mlflow.start_span(
                                    name=f"{agent_type}_Decision",
                                    attributes={"trace_attributes": trace_id_groups[_trace_id]},
                                    span_type=SpanType.CHAT_MODEL
                                ) as decision_span:
                                    decision_span.set_inputs(trace_info['inference_configuration'])
                                    decision_span.set_outputs({"rationale": trace_info["rationale"], "usage_metadata": trace_info["usage_metadata"]})
                        
                        # add message
                        self._messages.append(
                            self.create_message(
                                role="Observation",
                                source=agentAliasArn.split('/')[1],
                                destination=agentAliasArn.split('/')[1],
                                content=trace_info["rationale"],
                                actions=None,
                                observation=trace_info["rationale"]
                            )
                        )
                        # reset trace_info
                        trace_info = {
                            "rationale": None,
                            "inference_configuration": None,
                            "usage_metadata": None
                        }

                    if type == "invocationInput":
                        sub_agent_invocation_input = _trace.get('data',{}).get('agentCollaboratorInvocationInput')
                        action_group_invocation_input = _trace.get("data", {}).get("actionGroupInvocationInput")
                        knowledge_base_lookup_input = _trace.get("data", {}).get("knowledgeBaseLookupInput")
                        caller_chain = _trace.get('callerChain')
                        agentAliasArn = caller_chain[-1]["agentAliasArn"]

                        if action_group_invocation_input is not None:
                            metrics["num_tool_calls"] += 1
                            action_group_name = action_group_invocation_input.get(
                                "actionGroupName"
                            )
                            api_path = action_group_invocation_input.get(
                                "apiPath"
                            )
                            execution_type = action_group_invocation_input.get(
                                "executionType"
                            )
                            action_group_trace_info["action_group_name"] = action_group_name
                            action_group_trace_info["api_path"] = api_path
                            action_group_trace_info["execution_type"] = execution_type

                            
                        
                        if knowledge_base_lookup_input is not None:
                            metrics["num_kb_lookups"] += 1
                            knowledge_base_id = knowledge_base_lookup_input.get("knowledgeBaseId")
                            text = knowledge_base_lookup_input.get("text")
                            knowledge_base_trace_info["knowledge_base_id"] = knowledge_base_id
                            knowledge_base_trace_info["text"] = text
                            
                            
                        if sub_agent_invocation_input is not None:
                            collaborator_name = sub_agent_invocation_input.get("agentCollaboratorName")
                            
                            sub_agent_input = sub_agent_invocation_input.get("input",{}).get("text",{})
                            
                            metrics["num_agent_calls"] += 1

                            with mlflow.start_span(
                                    name=f"Supervisor-Input_to_{collaborator_name}",
                                    attributes={"trace_attributes": trace_id_groups[_trace_id]}
                                ) as agent_input_span:
                                    agent_input_span.set_inputs(sub_agent_input)
                            
                            # add message
                            # print(f"sub-agent input caller_chain: {caller_chain}")
                            self._messages.append(
                                self.create_message(
                                    role=None,
                                    source=self._bedrock_agent_id,
                                    destination=agentAliasArn.split('/')[1],
                                    content=sub_agent_input,
                                    actions=None,
                                    observation=None
                                )
                            )
                                    
                            

                    if type == "observation":
                        sub_agent_output = _trace.get('data',{}).get('agentCollaboratorInvocationOutput')
                        action_group_output = _trace.get("data", {}).get("actionGroupInvocationOutput")
                        knowledge_base_lookup_output = _trace.get("data", {}).get("knowledgeBaseLookupOutput")
                        finalResponse_output = _trace.get("data", {}).get("finalResponse")
                        caller_chain = _trace.get('callerChain')
                        agentAliasArn = caller_chain[-1]["agentAliasArn"]

                        if sub_agent_output is not None:
                            # print(f"sub_agent_output: {sub_agent_output}")
                            # print (f"sub_agent_output caller_chain: {caller_chain}")
                            collaborator_name = sub_agent_output.get('agentCollaboratorName',{})
                            agentOutput = sub_agent_output.get('output',{}).get('text',{})
                            pending_agent_outputs["name"] = collaborator_name
                            pending_agent_outputs["output"] = agentOutput
                            pending_agent_outputs["agentAliasArn"] = agentAliasArn
                            event_order = _trace.get('event_order')
                            pending_agent_outputs["event_order"] = event_order

                            

                        if action_group_output is not None:
                            execution_output = action_group_output.get('text',{})

                            with mlflow.start_span(
                                    name=f"ActionGroup_{action_group_trace_info['action_group_name']}",
                                    attributes={"trace_attributes": trace_id_groups[_trace_id]},
                                    span_type=SpanType.TOOL
                                ) as action_output_span:
                                    action_output_span.set_inputs(action_group_trace_info)
                                    action_output_span.set_outputs({"execution_output": execution_output})
                            
                            # add two messages, first action input, second action output
                            self._messages.append(
                                self.create_message(
                                    role="Action",
                                    source=agentAliasArn.split('/')[1],
                                    destination=agentAliasArn.split('/')[1],
                                    content=None,
                                    actions=action_group_trace_info,
                                    observation=None
                                )
                            )
                            self._messages.append(
                                self.create_message(
                                    role="Observation",
                                    source=agentAliasArn.split('/')[1],
                                    destination=agentAliasArn.split('/')[1],
                                    content=execution_output,
                                    actions=None,
                                    observation=execution_output
                                )
                            )
                            # reset action_group_trace_info
                            action_group_trace_info = {
                                "action_group_name": None,
                                "api_path": None,
                                "execution_type": None
                            }

                        if knowledge_base_lookup_output is not None:
                            knowledge_base_id = knowledge_base_lookup_output.get("knowledgeBaseId")
                            retrieved_references = knowledge_base_lookup_output.get("retrievedReferences")
                            
                            with mlflow.start_span(
                                    name=f"KnowledgeBase_{knowledge_base_trace_info['knowledge_base_id']}",
                                    attributes={"trace_attributes": trace_id_groups[_trace_id]},
                                    span_type=SpanType.RETRIEVER
                                ) as kb_output_span:
                                    kb_output_span.set_inputs(knowledge_base_trace_info)
                                    kb_output_span.set_outputs({"knowledge_base_id": knowledge_base_id,"retrieved_references": retrieved_references})
                            
                            # add two messages, first kb call input, second kb output
                            self._messages.append(
                                self.create_message(
                                    role="Action",
                                    source=agentAliasArn.split('/')[1],
                                    destination=agentAliasArn.split('/')[1],
                                    content=None,
                                    actions=knowledge_base_trace_info,
                                    observation=None
                                )
                            )
                            self._messages.append(
                                self.create_message(
                                    role="Observation",
                                    source=agentAliasArn.split('/')[1],
                                    destination=agentAliasArn.split('/')[1],
                                    content={"knowledge_base_id": knowledge_base_id,"retrieved_references": retrieved_references},
                                    actions=None,
                                    observation={"knowledge_base_id": knowledge_base_id,"retrieved_references": retrieved_references}
                                )
                            )
                            
                            # reset knowledge_base_trace_info
                            knowledge_base_trace_info = {
                                "knowledge_base_id": None,
                                "text": None
                            }

                        if finalResponse_output is not None:
                            final_response = finalResponse_output.get("text")
                            with mlflow.start_span(
                                    name=f"Response_{agentAliasArn.split('/')[1]}",
                                    attributes={"trace_attributes": trace_id_groups[_trace_id]},
                                    span_type=SpanType.AGENT
                                ) as agentresponse_span:
                                    
                                    agentresponse_span.set_outputs({"final_response": final_response})
                            
                            # print(f"final_response caller_chain: {caller_chain}")
                            # add message (assuming there is a max of 2 agents in the caller_chain)
                            self._messages.append(
                                self.create_message(
                                    role=None,
                                    source=caller_chain[-1]["agentAliasArn"].split('/')[1], # last one in caller_chain
                                    destination= caller_chain[0]["agentAliasArn"].split('/')[1], # first one in caller_chain
                                    content=final_response,
                                    actions=None,
                                    observation=None
                                )
                            )
                
            process_trace_group(self,trace_group)

        # check that list(trace_id_groups_copy.values()) and then list(trace_id_groups_copy.values())[-1][-1] is not empty or null
        if list(trace_id_groups_copy.values()) is not None:
            if list(trace_id_groups_copy.values())[-1][-1] is not None:
                final_response = (
                    list(trace_id_groups_copy.values())[-1][-1]
                    .get("data", {})
                    .get("finalResponse", {})
                    .get("text")
                )
            else:
                final_response = "No data returned from Bedrock Agent"
        else:
            final_response = "No data returned from Bedrock Agent"
        return final_response, metrics

    @mlflow.trace(name="Bedrock Input Prompt")
    def _get_agent_prompt(self, raw_input_question):
        """
        Constructs the agent prompt by combining the input question and the agent instruction.

        Args:
            raw_input_question (str): The user's input question.

        Returns:
            str: The formatted agent prompt.
        """

        # add message
        self._messages.append(
            self.create_message(
                role="User",
                source="User",
                destination=self._bedrock_agent_id,
                content=raw_input_question,
                actions=None,
                observation=None
            )
        )

        
        return f"""
        Answer the following question and pay strong attention to the prompt:
        <question>
        {raw_input_question}
        </question>
        <instruction>
        {self._agent_instruction}
        </instruction>
        """

  

    @mlflow.trace(name="bedrock-multi-agent-collaboration", span_type=SpanType.CHAT_MODEL)
    def predict(
        self, context, messages: List[ChatMessage], params: Optional[ChatParams]
    ) -> ChatResponse:
        """
        Makes a prediction using the Bedrock agent and processes the response.

        Args:
            context: The MLflow context.
            messages (List[ChatMessage]): A list of chat messages.
            params (Optional[ChatParams]): Optional parameters for the chat.

        Returns:
            ChatResponse: The response from the Bedrock agent.
        """

        metrics = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "num_agent_calls": 0,
            "num_tool_calls": 0,
            "num_kb_lookups": 0
        }
        
    
        formatted_input = messages[-1].content
        session_id = uuid.uuid4().hex

        # response = self.brt.invoke_agent(
        #     agentId=self._bedrock_agent_id,
        #     agentAliasId=self._bedrock_agent_alias_id,
        #     inputText=self._get_agent_prompt(formatted_input),
        #     enableTrace=True,
        #     sessionId=session_id,
        #     endSession=False,
        # )
        response = self.invoke_agent_with_retry(
            session_id, 
            formatted_input
        )

        # Since this provider's output doesn't match the OpenAI specification,
        # we need to go through the returned trace data and map it appropriately
        # to create the MLflow span object.
        events = []
        for index, event in enumerate(response.get("completion", [])):
            if "trace" in event:
                event["trace"]["event_order"] = index
            events.append(event)
            # print(f"event: {event}")

        trace_id_groups = self._extract_trace_groups(events)

        final_response, metrics = self._get_final_response_with_trace(trace_id_groups)
        # print(f"final_response: {final_response}")
        if final_response is None:
            final_response = "No data returned from Bedrock Agent"
        with mlflow.start_span(
            name="retrieved-response", span_type=SpanType.AGENT
        ) as span:
            span.set_inputs(messages)
            span.set_attributes({})

            output = ChatResponse(
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="user", content=final_response)
                    )
                ],
                usage= {
                    "total_tokens": metrics["total_tokens"],
                    "prompt_tokens": metrics["prompt_tokens"],
                    "completion_tokens": metrics["completion_tokens"]
                },
                model=self._model,
            )

            span.set_outputs(output)

        # add message
        # delete the last message in self._messages
        self._messages.pop()
        # now append the final_response
        self._messages.append(
            self.create_message(
                role=None,
                source=self._bedrock_agent_id,
                destination="User",
                content=final_response,
                actions=None,
                observation=None
            )
        )

                
        self._conversation_json_dict = self.create_conversation_json_dict(self._messages)
        self._metrics = metrics
  
        self._messages = []
            
        return output