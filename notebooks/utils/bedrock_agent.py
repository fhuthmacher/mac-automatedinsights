# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
import time
from botocore.exceptions import ClientError

import random
import string
import uuid
import io
import zipfile
import yaml
import re
import logging
import json

ROLE_POLICY_NAME = "agent_permissions"


logger = logging.getLogger(__name__)


class MaxRetriesExceededError(Exception):
    pass


def wait(seconds, tick=12):
    """
    Waits for a specified number of seconds, while also displaying an animated
    spinner.

    :param seconds: The number of seconds to wait.
    :param tick: The number of frames per second used to animate the spinner.
    """
    progress = "|/-\\"
    waited = 0
    while waited < seconds:
        for frame in range(tick):
            sys.stdout.write(f"\r{progress[frame % len(progress)]}")
            sys.stdout.flush()
            time.sleep(1 / tick)
        waited += 1
    sys.stdout.write("\r")
    sys.stdout.flush()

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Bedrock to manage
Bedrock Agents.
"""

import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class BedrockAgentWrapper:
    """Encapsulates Amazon Bedrock Agent actions."""

    def __init__(self, client):
        """
        :param client: A Boto3 Amazon Bedrock Agents client, which is a low-level client that
                       represents Amazon Bedrock Agents and describes the API operations
                       for creating and managing Bedrock Agent resources.
        """
        self.client = client
    
    def wait_agent_status_update(self, agent_id):
        response = self.client.get_agent(agentId=agent_id)
        agent_status = response["agent"]["agentStatus"]
        _waited_at_least_once = False
        while agent_status.endswith("ING"):
            print(f"Waiting for agent status to change. Current status {agent_status}")
            time.sleep(5)
            _waited_at_least_once = True
            try:
                response = self.client.get_agent(agentId=agent_id)
                agent_status = response["agent"]["agentStatus"]
            except self._bedrock_agent_client.exceptions.ResourceNotFoundException:
                agent_status = "DELETED"
        if _waited_at_least_once:
            print(f"Agent id {agent_id} current status: {agent_status}")

    def associate_sub_agents(self, supervisor_agent_id, sub_agents_list):
        for sub_agent in sub_agents_list:
            
            association_response = self.client.associate_agent_collaborator(
                    agentId=supervisor_agent_id,
                    agentVersion="DRAFT",
                    agentDescriptor={"aliasArn": sub_agent["sub_agent_alias_arn"]},
                    collaboratorName=sub_agent["sub_agent_association_name"],
                    collaborationInstruction=sub_agent["sub_agent_instruction"],
                    relayConversationHistory=sub_agent["relay_conversation_history"],
                )
            # print(f'association_response: {association_response}')
            # logger.info(f"Associated sub-agent {sub_agent['sub_agent_association_name']} with supervisor agent {supervisor_agent_id}")


    def create_agent(self, agent_name, foundation_model, role_arn, instruction, agentCollaboration='DISABLED', promptOverrideConfiguration=None, idleSessionTTLInSeconds=3600):
        """
        Creates an agent that orchestrates interactions between foundation models,
        data sources, software applications, user conversations, and APIs to carry
        out tasks to help customers.

        :param agent_name: A name for the agent.
        :param foundation_model: The foundation model to be used for orchestration by the agent.
        :param role_arn: The ARN of the IAM role with permissions needed by the agent.
        :param instruction: Instructions that tell the agent what it should do and how it should
                            interact with users.
        :param agentCollaboration: Configuration for agent collaboration.
        :param promptOverrideConfiguration: Configuration for prompt overrides.
        :param idleSessionTTLInSeconds: Time to live for idle sessions in seconds.
        :return: The response from Amazon Bedrock Agents if successful, otherwise raises an exception.
        """
        try:
            # Add basic validation
            if not instruction:
                raise ValueError("Instruction cannot be null or empty")
            
            if promptOverrideConfiguration is None:
                response = self.client.create_agent(
                    agentName=agent_name,
                    foundationModel=foundation_model,
                    agentResourceRoleArn=role_arn,
                    instruction=instruction,
                    agentCollaboration=agentCollaboration,
                    idleSessionTTLInSeconds=idleSessionTTLInSeconds
                )
            else:
                print(f'promptOverrideConfiguration: {promptOverrideConfiguration}')
                response = self.client.create_agent(
                agentName=agent_name,
                foundationModel=foundation_model,
                agentResourceRoleArn=role_arn,
                instruction=instruction,
                agentCollaboration=agentCollaboration,
                promptOverrideConfiguration=promptOverrideConfiguration,
                idleSessionTTLInSeconds=idleSessionTTLInSeconds
            )
        except ClientError as e:
            logger.error(f"Error: Couldn't create agent. Here's why: {e}")
            raise
        else:
            return response["agent"]

    def update_agent(self, agent_id,agent_name, foundation_model, instruction, agent_role_arn, agentCollaboration, promptOverrideConfiguration):
        """
        Creates an agent that orchestrates interactions between foundation models,
        data sources, software applications, user conversations, and APIs to carry
        out tasks to help customers.

        :param agent_id: The unique identifier of the agent to update.
        :param agent_version: The version of the agent to update.
        :param agentCollaboration: Configuration for agent collaboration.
        :param promptOverrideConfiguration: Configuration for prompt overrides.
        :return: The response from Amazon Bedrock Agents if successful, otherwise raises an exception.
        """
        try:


            response = self.client.update_agent(
                agentId=agent_id,
                agentName=agent_name,
                foundationModel=foundation_model,
                agentResourceRoleArn=agent_role_arn,
                instruction=instruction,
                agentCollaboration=agentCollaboration,
                promptOverrideConfiguration=promptOverrideConfiguration
            )
        except ClientError as e:
            logger.error(f"Error: Couldn't update agent. Here's why: {e}")
            raise
        else:
            return response["agent"]


    
    # snippet-start:[python.example_code.bedrock-agent.CreateAgentAlias]
    def create_agent_alias(self, name, agent_id):
        """
        Creates an alias of an agent that can be used to deploy the agent.

        :param name: The name of the alias.
        :param agent_id: The unique identifier of the agent.
        :return: Details about the alias that was created.
        """
        try:
            response = self.client.create_agent_alias(
                agentAliasName=name, agentId=agent_id
            )
            agent_alias = response["agentAlias"]
        except ClientError as e:
            logger.error(f"Couldn't create agent alias. {e}")
            raise
        else:
            return agent_alias

    # snippet-end:[python.example_code.bedrock-agent.CreateAgentAlias]

    # snippet-start:[python.example_code.bedrock-agent.DeleteAgent]
    def delete_agent(self, agent_id):
        """
        Deletes an Amazon Bedrock agent.

        :param agent_id: The unique identifier of the agent to delete.
        :return: The response from Amazon Bedrock Agents if successful, otherwise raises an exception.
        """

        try:
            response = self.client.delete_agent(
                agentId=agent_id, skipResourceInUseCheck=False
            )
        except ClientError as e:
            logger.error(f"Couldn't delete agent. {e}")
            raise
        else:
            return response

    # snippet-end:[python.example_code.bedrock-agent.DeleteAgent]

    # snippet-start:[python.example_code.bedrock-agent.DeleteAgentAlias]
    def delete_agent_alias(self, agent_id, agent_alias_id):
        """
        Deletes an alias of an Amazon Bedrock agent.

        :param agent_id: The unique identifier of the agent that the alias belongs to.
        :param agent_alias_id: The unique identifier of the alias to delete.
        :return: The response from Amazon Bedrock Agents if successful, otherwise raises an exception.
        """

        try:
            response = self.client.delete_agent_alias(
                agentId=agent_id, agentAliasId=agent_alias_id
            )
        except ClientError as e:
            logger.error(f"Couldn't delete agent alias. {e}")
            raise
        else:
            return response

    # snippet-end:[python.example_code.bedrock-agent.DeleteAgentAlias]

    # snippet-start:[python.example_code.bedrock-agent.GetAgent]
    def get_agent(self, agent_id, log_error=True):
        """
        Gets information about an agent.

        :param agent_id: The unique identifier of the agent.
        :param log_error: Whether to log any errors that occur when getting the agent.
                          If True, errors will be logged to the logger. If False, errors
                          will still be raised, but not logged.
        :return: The information about the requested agent.
        """

        try:
            response = self.client.get_agent(agentId=agent_id)
            agent = response["agent"]
        except ClientError as e:
            if log_error:
                logger.error(f"Couldn't get agent {agent_id}. {e}")
            raise
        else:
            return agent

    # snippet-end:[python.example_code.bedrock-agent.GetAgent]

    # snippet-start:[python.example_code.bedrock-agent.ListAgents]
    def list_agents(self):
        """
        List the available Amazon Bedrock Agents.

        :return: The list of available bedrock agents.
        """

        try:
            all_agents = []

            paginator = self.client.get_paginator("list_agents")
            for page in paginator.paginate(PaginationConfig={"PageSize": 10}):
                all_agents.extend(page["agentSummaries"])

        except ClientError as e:
            logger.error(f"Couldn't list agents. {e}")
            raise
        else:
            return all_agents

    # snippet-end:[python.example_code.bedrock-agent.ListAgents]

    # snippet-start:[python.example_code.bedrock-agent.ListAgentActionGroups]
    def list_agent_action_groups(self, agent_id, agent_version):
        """
        List the action groups for a version of an Amazon Bedrock Agent.

        :param agent_id: The unique identifier of the agent.
        :param agent_version: The version of the agent.
        :return: The list of action group summaries for the version of the agent.
        """

        try:
            action_groups = []

            paginator = self.client.get_paginator("list_agent_action_groups")
            for page in paginator.paginate(
                    agentId=agent_id,
                    agentVersion=agent_version,
                    PaginationConfig={"PageSize": 10},
            ):
                action_groups.extend(page["actionGroupSummaries"])

        except ClientError as e:
            logger.error(f"Couldn't list action groups. {e}")
            raise
        else:
            return action_groups

    # snippet-end:[python.example_code.bedrock-agent.ListAgentActionGroups]

    # snippet-start:[python.example_code.bedrock-agent.ListAgentKnowledgeBases]
    def list_agent_knowledge_bases(self, agent_id, agent_version):
        """
        List the knowledge bases associated with a version of an Amazon Bedrock Agent.

        :param agent_id: The unique identifier of the agent.
        :param agent_version: The version of the agent.
        :return: The list of knowledge base summaries for the version of the agent.
        """

        try:
            knowledge_bases = []

            paginator = self.client.get_paginator("list_agent_knowledge_bases")
            for page in paginator.paginate(
                    agentId=agent_id,
                    agentVersion=agent_version,
                    PaginationConfig={"PageSize": 10},
            ):
                knowledge_bases.extend(page["agentKnowledgeBaseSummaries"])

        except ClientError as e:
            logger.error(f"Couldn't list knowledge bases. {e}")
            raise
        else:
            return knowledge_bases

    # snippet-end:[python.example_code.bedrock-agent.ListAgentKnowledgeBases]

    # snippet-start:[python.example_code.bedrock-agent.PrepareAgent]
    def prepare_agent(self, agent_id):
        """
        Creates a DRAFT version of the agent that can be used for internal testing.

        :param agent_id: The unique identifier of the agent to prepare.
        :return: The response from Amazon Bedrock Agents if successful, otherwise raises an exception.
        """
        try:
            prepared_agent_details = self.client.prepare_agent(agentId=agent_id)
        except ClientError as e:
            logger.error(f"Couldn't prepare agent. {e}")
            raise
        else:
            return prepared_agent_details



class BedrockAgentScenarioWrapper:
    """Runs a scenario that shows how to get started using Amazon Bedrock Agents."""

    def __init__(
            self, bedrock_agent_client, 
            runtime_client, 
            lambda_client, 
            iam_resource, 
            postfix, 
            agent_name, 
            model_id, 
            prompt, 
            action_group_schema_path,
            lambda_image_uri,
            lambda_environment_variables,
            instruction,
            agentCollaboration,
            promptOverrideConfiguration,
            sub_agents_list,
            enable_trace=True
    ):
        self.iam_resource = iam_resource
        self.lambda_client = lambda_client
        self.bedrock_agent_runtime_client = runtime_client
        self.postfix = postfix
        self.agent_name = agent_name
        self.model_id = model_id
        self.bedrock_agent_client = bedrock_agent_client
        self.bedrock_wrapper = BedrockAgentWrapper(bedrock_agent_client)

        self.agent = None
        self.agent_alias = None
        self.agent_role = None
        self.prepared_agent_details = None
        self.lambda_role = None
        self.lambda_function = None
        self.action_group_schema_path = action_group_schema_path
        self.lambda_image_uri = lambda_image_uri
        self.lambda_environment_variables = lambda_environment_variables
        self.prompt = prompt
        self.instruction = instruction
        self.agentCollaboration = agentCollaboration
        self.promptOverrideConfiguration = promptOverrideConfiguration
        self.sub_agents_list = sub_agents_list
        self.enable_trace = enable_trace

    def run_scenario(self):
        print("=" * 88)
        print("Amazon Bedrock Agents scenario.")
        print("=" * 88)

        # Query input from user
        print("Let's start with creating an agent:")
        print("-" * 40)
        name, foundation_model = self._request_name_and_model_from_user()
        print("-" * 40)

        # Create an execution role for the agent
        self.agent_role = self._create_agent_role(foundation_model)

        # Create the agent
        self.agent = self._create_agent(name, foundation_model, self.agent_role.arn, self.instruction, self.agentCollaboration, self.promptOverrideConfiguration, 3600)

        # Prepare a DRAFT version of the agent
        self.prepared_agent_details = self._prepare_agent()

        if self.action_group_schema_path:
            # Do not create the agent's Lambda function, just use the one passed in
            self.lambda_function = self._create_lambda_function()

            # Configure permissions for the agent to invoke the Lambda function
            self._allow_agent_to_invoke_function()
            self._let_function_accept_invocations_from_agent()

            # Create an action group to connect the agent with the Lambda function
            self._create_agent_action_group()

        # If the agent has been modified or any components have been added, prepare the agent again
        components = [self._get_agent()]
        components += self._get_agent_action_groups()
        components += self._get_agent_knowledge_bases()

        latest_update = max(component["updatedAt"] for component in components)
        if latest_update > self.prepared_agent_details["preparedAt"]:
            self.prepared_agent_details = self._prepare_agent()

        # Create an agent alias
        self.agent_alias = self._create_agent_alias()

        # Test the agent
        print("Test agent")
        response = ""
        trace_events = []
        response, trace_events = self.chat_with_agent()
        return response, trace_events


    def _request_name_and_model_from_user(self):
        
        name = self.agent_name
        model_id = self.model_id

        return name, model_id

    def _create_agent_role(self, model_id):
        role_name = f"AmazonBedrockExecutionRoleForAgents_{self.postfix}"
        model_arn = f"*"

        print("Creating an an execution role for the agent...")

        try:
            role = self.iam_resource.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": [
                                    "bedrock.amazonaws.com"
                                    
                                ]},
                                "Action": "sts:AssumeRole",
                            }
                        ],
                    }
                ),
            )

            role.Policy(ROLE_POLICY_NAME).put(
                PolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Action": ["bedrock:InvokeModel", 
                                           "bedrock:InvokeModelWithResponseStream",
                                           "bedrock:GetInferenceProfile",
                                           "bedrock:ListInferenceProfiles",
                                           "bedrock:DeleteInferenceProfile",
                                           "bedrock:TagResource",
                                           "bedrock:UntagResource",
                                           "bedrock:ListTagsForResource",
                                           "bedrock:*"],
                                "Resource": model_arn,
                            },
                            {
                                "Effect": "Allow",
                                "Action": "iam:PassRole",
                                "Resource": "*",
                                "Condition": {
                                    "StringEquals": {
                                        "iam:PassedToService": ["lambda.amazonaws.com",
                                                                "bedrock.amazonaws.com"]
                                    }
                                }
                            },
                        ],
                    }
                )
            )
        except ClientError as e:
            logger.error(f"Couldn't create role {role_name}. Here's why: {e}")
            raise

        return role
    
    def _associate_sub_agents(self, supervisor_agent_id, sub_agents_list):
            print(f'start associating sub_agents_list: {sub_agents_list}')
            
            self.bedrock_wrapper.associate_sub_agents(
                supervisor_agent_id=supervisor_agent_id,
                sub_agents_list=sub_agents_list,
            )
            
            self._wait_for_agent_status(supervisor_agent_id, "NOT_PREPARED")
            print(f'end associating sub_agents_list')
    def _create_agent(self, name, model_id, agent_role_arn, instruction, agentCollaboration, promptOverrideConfiguration, idleSessionTTLInSeconds):
        print("Creating the agent...")

        agent = self.bedrock_wrapper.create_agent(
            agent_name=name,
            foundation_model=model_id,
            instruction=instruction,
            role_arn=agent_role_arn,
            agentCollaboration=agentCollaboration,
            promptOverrideConfiguration=promptOverrideConfiguration,
            idleSessionTTLInSeconds=idleSessionTTLInSeconds
        )
        self.bedrock_wrapper.wait_agent_status_update(agent["agentId"])

        # update_agent_response = self.bedrock_wrapper.update_agent(
        #     agent_id=agent["agentId"],
        #     agent_name=name,
        #     foundation_model=model_id,
        #     instruction=instruction,
        #     agent_role_arn=agent_role_arn,
        #     agentCollaboration=agentCollaboration,
        #     promptOverrideConfiguration=promptOverrideConfiguration,
        # )
        # print(f"update_agent_response: {update_agent_response}")

        print(f'created agent: {agent}')

        associate_sub_agents_response = self._associate_sub_agents(agent["agentId"], self.sub_agents_list)
        print(f"associate_sub_agents_response: {associate_sub_agents_response}")

        return agent

    def _prepare_agent(self):
        print("Preparing the agent...")

        agent_id = self.agent["agentId"]
        prepared_agent_details = self.bedrock_wrapper.prepare_agent(agent_id)
        self._wait_for_agent_status(agent_id, "PREPARED")

        return prepared_agent_details

    def _create_lambda_function(self):
        print("Creating the Lambda function...")

        function_name = f"AmazonBedrockAgentFunction_{self.postfix}"

        self.lambda_role = self._create_lambda_role()

        try:
            # deployment_package = self._create_deployment_package(function_name)

            lambda_function = self.lambda_client.create_function(
                FunctionName=function_name,
                Description="Lambda function for Bedrock Agent",
                Role=self.lambda_role.arn,                
                PackageType='Image',
                Code={
                    'ImageUri': self.lambda_image_uri
                },
                EphemeralStorage={
                    'Size': 10240  # Size in MB
                },
                MemorySize=10240,  # Maximum memory
                Timeout=900,  # 15 minutes
                Architectures=['arm64'], #['x86_64']
                Environment={
                    'Variables': self.lambda_environment_variables
                }
            )

            waiter = self.lambda_client.get_waiter("function_active_v2")
            waiter.wait(FunctionName=function_name)

        except ClientError as e:
            logger.error(
                f"Couldn't create Lambda function {function_name}. Here's why: {e}"
            )
            raise

        return lambda_function

    def _create_lambda_role(self):
        print("Creating an execution role for the Lambda function...")

        role_name = f"AmazonBedrockExecutionRoleForLambda_{self.postfix}"

        try:
            role = self.iam_resource.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": "lambda.amazonaws.com"},
                                "Action": "sts:AssumeRole",
                            },
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": [
                                        "bedrock.amazonaws.com"
                                        
                                    ]
                                },
                                "Action": "sts:AssumeRole"
                            },
                            
                            
                        ],
                    }
                ),
            )
            role.attach_policy(
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
            )
            role.attach_policy(
                PolicyArn="arn:aws:iam::aws:policy/AmazonBedrockFullAccess"
            )
            role.attach_policy(
                PolicyArn="arn:aws:iam::aws:policy/AmazonAthenaFullAccess"
            )

            # add inline policy to role
            role.Policy('passediam').put(
                PolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Action": "iam:PassRole",
                                "Resource": "*",
                                "Condition": {
                                    "StringEquals": {
                                        "iam:PassedToService": ["lambda.amazonaws.com",
                                                                "bedrock.amazonaws.com"]
                                    }
                                }
                            },
                            {
                                "Effect": "Allow",
                                "Action": "lambda:*",
                                "Resource": "*",
                                
                            },
                            {
                                "Effect": "Allow",
                                "Action": "s3:*",
                                "Resource": "*",
                                
                            },
                        ],
                    }
                )
            )
            print(f"Created role {role_name}")
        except ClientError as e:
            logger.error(f"Couldn't create role {role_name}. Here's why: {e}")
            raise

        print("Waiting for the execution role to be fully propagated...")
        wait(30)

        return role

    def _allow_agent_to_invoke_function(self):
        policy = self.iam_resource.RolePolicy(
            self.agent_role.role_name, ROLE_POLICY_NAME
        )
        doc = policy.policy_document
        doc["Statement"].append(
            {
                "Effect": "Allow",
                "Action": "lambda:InvokeFunction",
                "Resource": self.lambda_function["FunctionArn"],
            }
        )
        self.agent_role.Policy(ROLE_POLICY_NAME).put(PolicyDocument=json.dumps(doc))

    def _let_function_accept_invocations_from_agent(self):
        try:
            self.lambda_client.add_permission(
                FunctionName=self.lambda_function["FunctionName"],
                SourceArn=self.agent["agentArn"],
                StatementId="BedrockAccess",
                Action="lambda:InvokeFunction",
                Principal="bedrock.amazonaws.com",
            )
        except ClientError as e:
            logger.error(
                f"Couldn't grant Bedrock permission to invoke the Lambda function. Here's why: {e}"
            )
            raise

    def _create_agent_action_group(self):
        print("Creating an action group for the agent...")

        try:
            if self.action_group_schema_path:
                with open(self.action_group_schema_path) as file:
                    # load yaml file
                    yaml_content = yaml.safe_load(file)
                    print(f'title: {yaml_content["info"]["title"]}')
                    print(f'description: {yaml_content["info"]["description"]}')
                api_schema = {
                    "payload": json.dumps(yaml_content)
                }
                
                response = self.bedrock_agent_client.create_agent_action_group(
                    actionGroupName=yaml_content["info"]["title"],
                    description=yaml_content["info"]["description"],
                    agentId=self.agent["agentId"],
                    agentVersion=self.prepared_agent_details["agentVersion"],
                    actionGroupExecutor={"lambda": self.lambda_function["FunctionArn"]},
                    apiSchema=api_schema
                    
                )
                agent_action_group = response["agentActionGroup"]
            else:
                agent_action_group = None
                logger.info("No action group schema provided, skipping creation of agent action group")
            # logger.info(agent_action_group)
        except ClientError as e:
            logger.error(f"Couldn't create agent action group. Here's why: {e}")
            raise

    def _get_agent(self):
        return self.bedrock_wrapper.get_agent(self.agent["agentId"])

    def _get_agent_action_groups(self):
        return self.bedrock_wrapper.list_agent_action_groups(
            self.agent["agentId"], self.prepared_agent_details["agentVersion"]
        )

    def _get_agent_knowledge_bases(self):
        return self.bedrock_wrapper.list_agent_knowledge_bases(
            self.agent["agentId"], self.prepared_agent_details["agentVersion"]
        )

    def _create_agent_alias(self):
        print("Creating an agent alias...")

        agent_alias_name = "test_agent_alias"
        agent_alias = self.bedrock_wrapper.create_agent_alias(
            agent_alias_name, self.agent["agentId"]
        )

        # Wait for agent alias to be ready
        # Wait for agent alias to be ready
        print("Waiting for agent alias to be ready...")
        max_attempts = 12  # Maximum number of attempts (1 minute total)
        attempts = 0
        
        while attempts < max_attempts:
            try:
                response = self.bedrock_agent_client.get_agent_alias(
                    agentId=self.agent["agentId"],
                    agentAliasId=agent_alias["agentAliasId"]
                )
                status = response["agentAlias"]["agentAliasStatus"]
                print(f"Current alias status: {status}")
                
                # Check for successful states
                if status in ["PREPARED", "ACTIVE"]:
                    break
                # Check for failure states
                elif status in ["FAILED"]:
                    raise Exception(f"Agent alias creation failed with status: {status}")
                    
                attempts += 1
                wait(5)  # Wait 5 seconds before checking again
                
            except ClientError as e:
                logger.error(f"Error checking agent alias status: {e}")
                raise

        if attempts >= max_attempts:
            raise TimeoutError("Timed out waiting for agent alias to become ready")

        self._wait_for_agent_status(self.agent["agentId"], "PREPARED")

        return agent_alias

    def _wait_for_agent_status(self, agent_id, status):
        while self.bedrock_wrapper.get_agent(agent_id)["agentStatus"] != status:
            wait(2)

    def chat_with_agent(self):
        print("-" * 88)
        
        # Create a unique session ID for the conversation
        session_id = uuid.uuid4().hex

        response = ""
        trace_events = []

        response, trace_events = self._invoke_agent(self.agent_alias, self.prompt, session_id, self.enable_trace)
        return response, trace_events

    def _invoke_agent(self, agent_alias, prompt, session_id, enable_trace):
        try:

            response = self.bedrock_agent_runtime_client.invoke_agent(
                agentId=self.agent["agentId"],
                agentAliasId=agent_alias["agentAliasId"],
                sessionId=session_id,
                inputText=prompt,
                enableTrace=enable_trace
            )
            completion = ""
            trace_events = []
            index = 0
            for event in response.get("completion"):
                if "chunk" in event:
                    chunk = event["chunk"]
                    completion += chunk["bytes"].decode()
    
                if "trace" in event:
                    event["trace"]["event_order"] = index
                    trace_events.append(event)
                    index += 1
            logger.info(completion)
            # logger.info(f"Trace events: {trace_events}")

            return completion, trace_events
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            print(f"Error invoking agent: {e}")
            return "Error invoking agent"

    def _delete_resources(self):
        if self.agent:
            agent_id = self.agent["agentId"]

            if self.agent_alias:
                agent_alias_id = self.agent_alias["agentAliasId"]
                print("Deleting agent alias...")
                self.bedrock_wrapper.delete_agent_alias(agent_id, agent_alias_id)

            print("Deleting agent...")
            agent_status = self.bedrock_wrapper.delete_agent(agent_id)["agentStatus"]
            while agent_status == "DELETING":
                wait(5)
                try:
                    agent_status = self.bedrock_wrapper.get_agent(
                        agent_id, log_error=False
                    )["agentStatus"]
                except ClientError as err:
                    if err.response["Error"]["Code"] == "ResourceNotFoundException":
                        agent_status = "DELETED"

        if self.lambda_function:
            name = self.lambda_function["FunctionName"]
            print(f"Deleting function '{name}'...")
            self.lambda_client.delete_function(FunctionName=name)

        if self.agent_role:
            print(f"Deleting role '{self.agent_role.role_name}'...")
            self.agent_role.Policy(ROLE_POLICY_NAME).delete()
            self.agent_role.delete()

        if self.lambda_role:
            print(f"Deleting role '{self.lambda_role.role_name}'...")
            for policy in self.lambda_role.attached_policies.all():
                policy.detach_role(RoleName=self.lambda_role.role_name)
            self.lambda_role.delete()

    def _list_resources(self):
        print("-" * 40)
        print(f"Here is the list of created resources'.")
        
        if self.agent:
            print(f"Bedrock Agent:   {self.agent['agentName']}")
        if self.lambda_function:
            print(f"Lambda function: {self.lambda_function['FunctionName']}")
        if self.agent_role:
            print(f"IAM role:        {self.agent_role.role_name}")
        if self.lambda_role:
            print(f"IAM role:        {self.lambda_role.role_name}")
        
        # save these ARNs to a file
        with open(f'agent_{self.agent["agentName"]}_arns.txt', 'w') as f:
            f.write(f"Agent ARN: {self.agent['agentArn']}\n")
            f.write(f"Agent Alias ARN: {self.agent_alias['agentAliasArn']}\n")
            f.write(f"Lambda Function ARN: {self.lambda_function['FunctionArn']}\n")
            f.write(f"IAM Role ARN: {self.agent_role.role_name}\n")
            f.write(f"Lambda Role ARN: {self.lambda_role.role_name}\n")
        print(f'Created agent_{self.agent["agentName"]}_arns.txt which contains these details as well.')
        print("Make sure you delete them once you're done to avoid unnecessary costs.")
        
    @staticmethod
    def is_valid_agent_name(answer):
        valid_regex = r"^[a-zA-Z0-9_-]{1,100}$"
        return (
            answer
            if answer and len(answer) <= 100 and re.match(valid_regex, answer)
            else None,
            "I need a name for the agent, please. Valid characters are a-z, A-Z, 0-9, _ (underscore) and - (hyphen).",
        )

    @staticmethod
    def _create_deployment_package(function_name):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zipped:
            zipped.write(
                "./scenario_resources/lambda_function.py", f"{function_name}.py"
            )
        buffer.seek(0)
        return buffer.read()