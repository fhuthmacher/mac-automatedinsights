import json
from tqdm import tqdm
from utils.bedrock import BedrockLLMWrapper
from pathlib import Path
import boto3
import os
import json
import re
import uuid

APP_INF_PROFILE_ARN = ''

bedrock_llm = None

USER_GSR_PROMPT = """Determine whether the conversation between the user and agent satisfies a list of assertions. 
Pay attention to dates, time, location, and other relevant information about the user.
The judgement should be based on the given user scenario and the conversation history.
The user scenario provides the background information of the conversation.
The conversation history shows the interaction between user and agent.
 
Scenario:
{scenario}
 
Conversation History:
{history}
 
Assertions:
{assertions}
 
Answer TRUE or FALSE for each assertion. Provide answers in JSON array format with keys "assertion", "answer", and "evidence".
Please address every assertion. 
"""

SYSTEM_GSR_PROMPT = """Determine whether the conversation between the user and agent satisfies a list of assertions. 
Pay attention to dates, time, location, and other relevant information about the user.
The judgement should be based on the given user scenario, the conversation history, and the tool invocations.
The user scenario provides the background information of the conversation.
The conversation history shows the interaction between user and agent.
The tool invocations shows tool actions and observations from the agents during the conversation.
 
Scenario:
{scenario}
 
Conversation History:
{history}

Tool Invocations:
{invocations}
 
Assertions:
{assertions}
 
Answer TRUE or FALSE for each assertion. Provide answers in JSON array format with keys "assertion", "answer", and "evidence".
Please address every assertion.
"""

ISSUES_PROMPT = SYSTEM_GSR_PROMPT + "\nThis conversation has been judged as {judgement} which is either caused by the user, primary agent, sub-agents, or tool failures. If the conversation has failed, please take this into account when determining reliability of agents and tools."



def parse_conversation(conversation, primary_agent_id, human_id):
    # Check if conversation is in trace events format
    if 'trace_events' in conversation:
        conversation = conversation['trace_events']
        
    if isinstance(conversation, list) and all('trace' in event for event in conversation):
        primary_traj_string = ""
        subagent_traj_strings = []
        
        for event in conversation:
            trace_data = event['trace']['trace']
            
            # Handle orchestration traces
            if 'orchestrationTrace' in trace_data:
                orch_trace = trace_data['orchestrationTrace']
                
                # Handle model invocation input
                if 'modelInvocationInput' in orch_trace:
                    input_data = orch_trace['modelInvocationInput']
                    if isinstance(input_data, dict) and 'text' in input_data:
                        try:
                            input_json = json.loads(input_data['text'])
                            if 'messages' in input_json:
                                for msg in input_json['messages']:
                                    if isinstance(msg, dict) and 'content' in msg:
                                        primary_traj_string += f"[User -> Agent]: {msg['content']}\n"
                        except json.JSONDecodeError:
                            primary_traj_string += f"[System -> Agent]: {input_data['text']}\n"

                # Handle model invocation output
                if 'modelInvocationOutput' in orch_trace:
                    output_data = orch_trace['modelInvocationOutput']
                    if isinstance(output_data, dict) and 'rawResponse' in output_data:
                        try:
                            response_json = json.loads(output_data['rawResponse']['content'])
                            if 'content' in response_json:
                                for content in response_json['content']:
                                    if content.get('type') == 'text':
                                        primary_traj_string += f"[Agent -> User]: {content['text']}\n"
                                    elif content.get('type') == 'tool_use':
                                        subagent_traj_strings.append(
                                            f"[Tool Use]: {json.dumps(content['input'])}"
                                        )
                        except json.JSONDecodeError:
                            continue

                # Handle observations
                if 'observation' in orch_trace:
                    obs = orch_trace['observation']
                    if 'actionGroupInvocationOutput' in obs:
                        output = obs['actionGroupInvocationOutput']
                        if isinstance(output, dict) and 'text' in output:
                            try:
                                output_json = json.loads(output['text'])
                                subagent_traj_strings.append(
                                    f"[Action Output]: {json.dumps(output_json)}"
                                )
                            except json.JSONDecodeError:
                                subagent_traj_strings.append(
                                    f"[Action Output]: {output['text']}"
                                )
                    elif 'finalResponse' in obs:
                        final_resp = obs['finalResponse']
                        if isinstance(final_resp, dict) and 'text' in final_resp:
                            primary_traj_string += f"[Agent -> User]: {final_resp['text']}\n"

        return primary_traj_string, "\n".join(subagent_traj_strings)
    
    # Original code for handling the old conversation format
    else:
        # parse primary agent's trajectory
        primary_traj = conversation["trajectories"][primary_agent_id]
        primary_traj_string = "\n".join(
            [
                "[{} -> {}]: {}".format(
                    turn["source"], turn["destination"], turn["content"]
                ) for turn in primary_traj
            ]
        )
        # parse subagents trajectory
        subagent_traj_strings = []
        for subagent_id, subagent_traj in conversation["trajectories"].items():
            if subagent_id == primary_agent_id or subagent_id == human_id:
                continue
            def parse_subagent_turn(row):
                """Parse a single turn in the subagent trajectory."""
                subagent_string = ""
                if row["role"] == "Action":
                    # Handle case where actions is a dict directly (not a list)
                    if isinstance(row['actions'], dict):
                        action_str = f"action_group_name={row['actions'].get('action_group_name', '')}, "
                        action_str += f"api_path={row['actions'].get('api_path', '')}, "
                        action_str += f"execution_type={row['actions'].get('execution_type', '')}"
                        subagent_string = f"[{row['source']} {row['role']}]: {action_str}"
                    # Handle case where actions is a list of dicts
                    elif isinstance(row['actions'], list):
                        action_strings = []
                        for action in row['actions']:
                            if isinstance(action, dict):
                                action_str = f"action_group_name={action.get('action_group_name', '')}, "
                                action_str += f"api_path={action.get('api_path', '')}, "
                                action_str += f"execution_type={action.get('execution_type', '')}"
                                action_strings.append(action_str)
                            else:
                                action_strings.append(str(action))
                        subagent_string = f"[{row['source']} {row['role']}]: " + "; ".join(action_strings)
                elif row["role"] == "Observation":
                    subagent_string = f"[{row['source']} {row['role']}]: {row['observation']}"
                return subagent_string.rstrip("\n")

            subagent_traj_strings += [parse_subagent_turn(row) for row in subagent_traj]
        subagents_traj_string = "\n".join(subagent_traj_strings)

    return primary_traj_string, subagents_traj_string

def parse_assertions(assertions, gsr_type):
    """
    Assertions may have `User:` or `Agent:` in front of them to
    distinguish between user and agent assertions.

    Removes the prefix and returns type of assertions.
    """
    clean_assertions = []
    user_prefix = "user:"
    system_prefix = "agent:"
    for assertion in assertions:
        assertion = assertion.lstrip()
        if gsr_type == "user" and assertion.lower().startswith(user_prefix):
            clean_assertion = assertion[5:].strip()
            clean_assertions.append(clean_assertion)
        elif gsr_type == "system" and assertion.lower().startswith(system_prefix):
            clean_assertion = assertion[6:].strip()
            clean_assertions.append(clean_assertion)
        elif gsr_type == "user" and not assertion.lower().startswith(system_prefix):
            # assume that no prefix means user assertion
            clean_assertions.append(assertion)
    return clean_assertions

def parse_llm_judge_response(raw_response):
    if type(raw_response) == str:
        # Remove the first line if the first line begins with "Here"
        if raw_response.startswith("Here"):
            raw_response = "\n".join(raw_response.split("\n")[1:])

        substrings_to_remove = ["\n", "```json", "```" ]
        pattern = "|".join(substrings_to_remove)
        parsed_response = re.sub(pattern, "", raw_response)
        dict_list_response = json.loads(parsed_response)
    else:
        dict_list_response = raw_response
    # handle wrapper keys
    for k in ["assertions", "results", "duplicate_or_irrelevant_messages"]:
        if k in dict_list_response:
            dict_list_response = dict_list_response[k]
            break
    if type(dict_list_response) == dict:
        # wrap with list (happens for single-item responses) 
        dict_list_response = [dict_list_response]
    # concatenate any items that are lists
    for row in dict_list_response:
        for key, value in row.items():
            if type(value) == list:
                row[key] = " ".join(value) 
            if type(value) != str:
                row[key] = str(value)
    return dict_list_response



def compute_gsr(response):
    # count the number of true and false
    true_count = [row["answer"].lower() for row in response].count("true")
    false_count = [row["answer"].lower() for row in response].count("false")
    # binary score whether the conversation satisfies all assertions
    gsr = float(false_count == 0)  
    # partial score equal to the percentage of true assertions
    partial_gsr = true_count / (true_count + false_count)
    return gsr, partial_gsr

def evaluate_gsr(conversation, scenario, primary_agent_id, human_id, gsr_type, llm_judge_id):
    assert gsr_type in ["user", "system"], "gsr_type must be 'user' or 'system'"
    primary_string, subagent_string = parse_conversation(conversation, primary_agent_id, human_id)
    clean_assertions = parse_assertions(scenario["assertions"], gsr_type)
    if not clean_assertions:
        # no assertions for this type of GSR
        # provide default assertion to not break code
        clean_assertions = ["User goals are achieved with help from the agent."]

    if gsr_type == "user":
        prompt = USER_GSR_PROMPT.format(
            scenario=scenario, 
            history=primary_string, 
            assertions="\n".join(clean_assertions)
        )

    else:
        prompt = SYSTEM_GSR_PROMPT.format(
            scenario=scenario, 
            history=primary_string, 
            assertions="\n".join(clean_assertions),
            invocations=subagent_string
        )

    raw_response = bedrock_llm.generate(prompt=prompt)[0]
    

    response = parse_llm_judge_response(raw_response)

    gsr, partial_gsr = compute_gsr(response)
    for row in response:
        row["assertion_type"] = gsr_type
    return gsr, partial_gsr, response


def evaluate_conversation(conversation, scenario, primary_agent_id, human_id, llm_judge_id, trajectory_index):
    # evaluate user-side GSR
    user_gsr, user_partial_gsr, user_llm_report = evaluate_gsr(
        conversation, scenario, primary_agent_id, human_id, gsr_type="user", llm_judge_id=llm_judge_id  
    )
    # evaluate system-side GSR
    system_gsr, system_partial_gsr, system_llm_report = evaluate_gsr(
        conversation, scenario, primary_agent_id, human_id, gsr_type="system",
        llm_judge_id=llm_judge_id
    )
    # compute overall GSR
    overall_report = user_llm_report + system_llm_report
    overall_gsr, partial_gsr = compute_gsr(overall_report)
    result = {
        "trajectory_index": trajectory_index,
        "user_gsr": user_gsr,
        "system_gsr": system_gsr,
        "overall_gsr": overall_gsr,
        "partial_gsr": partial_gsr,
        "report": overall_report,
    }
    return result

def format_results(evals, num_scenarios):
    if not evals:
        results = {
            "user_gsr": 0.0,
            "system_gsr": 0.0,
            "overall_gsr": 0.0,
            "partial_gsr": 0.0,
            "scenario_count": num_scenarios,
            "conversation_count": 0,
            "conversation_evals": [],
            "error": "No conversations were successfully evaluated"
        }
    else:
        results = {
            "user_gsr": sum([e["user_gsr"] for e in evals]) / len(evals),
            "system_gsr": sum([e["system_gsr"] for e in evals]) / len(evals),
            "overall_gsr": sum([e["overall_gsr"] for e in evals]) / len(evals),
            "partial_gsr": sum([e["partial_gsr"] for e in evals]) / len(evals),
            "scenario_count": num_scenarios,
            "conversation_count": len(evals),
            "conversation_evals": evals
        }
    
    return results

def run_benchmark(
    dataset_dir: str | Path = "../data/eval_dataset",
    scenario_filename: str = "scenarios.json",
    conversations_dir: str | Path = "../data/eval_dataset/conversations",
    llm_judge_id: str = APP_INF_PROFILE_ARN,
    region: str = "us-east-1",
    session: boto3.Session = None
) -> None:
    """
    Run benchmark evaluation on conversation scenarios.
    
    Args:
        dataset_dir (str | Path): Directory containing scenario and agent files
        scenario_filename (str): Name of the scenario JSON file
        conversations_dir (str | Path): Directory containing conversation files
        llm_judge_id (str): ID of the LLM judge to use for evaluation
        region (str): AWS region to use for evaluation
        session (boto3.Session): AWS session to use for evaluation
    """
    # Convert string paths to Path objects
    dataset_dir = Path(dataset_dir)
    conversations_dir = Path(conversations_dir)
    
    if session is None:
        session = boto3.Session(region_name=region)
    
    global bedrock_llm
    bedrock_llm = BedrockLLMWrapper(
        model_id=llm_judge_id, 
        max_token_count=2000,
        temperature=0,
        region=region,
        system_prompt="return response in json format and only the json object",
        session=session
    )
    
    # load scenarios and agents 
    with open(dataset_dir / scenario_filename) as f:
        scenarios = json.load(f)["scenarios"]
    
    with open(dataset_dir / "agents.json") as f:
        agents = json.load(f)
    primary_agent_id = agents["primary_agent_id"]
    human_id = agents["human_id"]

    # evaluate conversations
    evals = []
    for i in range(len(scenarios)):
        if not (conversations_dir / f"conversation_{i}.json").exists():
            continue
        with open(conversations_dir / f"conversation_{i}.json") as f:
            conversation = json.load(f)
        result = evaluate_conversation(
            conversation, 
            scenarios[i], 
            primary_agent_id, 
            human_id, 
            llm_judge_id,
            trajectory_index=i
        )
        evals.append(result)

    # format results
    results = format_results(evals, len(scenarios))
    return results


def evaluate_single_agent_conversation(conversation, scenario, primary_agent_id, human_id, llm_judge_id, trajectory_index):
    # evaluate user-side GSR
    user_gsr, user_partial_gsr, user_llm_report = evaluate_gsr(
        conversation, scenario, primary_agent_id, human_id, gsr_type="user", llm_judge_id=llm_judge_id  
    )
    # print(f"user_llm_report: {user_llm_report}")
    # evaluate system-side GSR
    system_gsr, system_partial_gsr, system_llm_report = evaluate_gsr(
        conversation, scenario, primary_agent_id, human_id, gsr_type="system",
        llm_judge_id=llm_judge_id
    )
    # print(f"system_llm_report: {system_llm_report}")
    # compute overall GSR
    overall_report = user_llm_report + system_llm_report
    overall_gsr, partial_gsr = compute_gsr(overall_report)
    result = {
        "trajectory_index": trajectory_index,
        "user_gsr": user_gsr,
        "system_gsr": system_gsr,
        "overall_gsr": overall_gsr,
        "partial_gsr": partial_gsr,
        "report": overall_report,
    }
    # print(f"result: {result}")
    return result


def run_agent_evaluation(
    scenario_filepath: str = "scenarios.json",
    agent_filepath: str = "agent.json",
    llm_judge_id: str = APP_INF_PROFILE_ARN,
    region: str = "us-east-1",
    session: boto3.Session = None
) -> None:
    """
    Run agent evaluation on scenarios.
    
    Args:
        scenario_filepath (str): filepath to the scenario JSON file
        agent_filepath (str): filepath to the agent JSON file
        llm_judge_id (str): ID of the LLM judge to use for evaluation
        region (str): AWS region to use for evaluation
        session (boto3.Session): AWS session to use for evaluation
    """
    
    if session is None:     
        session = boto3.Session(region_name=region)
    
    global bedrock_llm
    bedrock_llm = BedrockLLMWrapper(
        model_id=llm_judge_id, 
        max_token_count=2000,
        temperature=0,
        region=region,
        system_prompt="return response in json format and only the json object",
        session=session
    )
    
    bedrock_agent_runtime_client = session.client("bedrock-agent-runtime")

    # load scenarios and agents 
    with open(scenario_filepath) as f:
        scenarios = json.load(f)["scenarios"]
    
    with open(agent_filepath) as f:
        agent = json.load(f)
    agent_id = agent["agent_id"]
    agent_alias_id = agent["agent_alias_id"]
    human_id = agent["human_id"]
    
    session_id = uuid.uuid4().hex

    # evaluate conversations
    evals = []
    for i in range(len(scenarios)):
        
        # invoke bedrock agent to get response
        response = bedrock_agent_runtime_client.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                inputText=scenarios[i]["input_problem"],
                enableTrace=True
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

        agent_response = trace_events
        
        result = evaluate_single_agent_conversation(
            agent_response, 
            scenarios[i], 
            agent_id, 
            human_id,
            llm_judge_id,
            trajectory_index=i
        )
        # print(f"result: {result}")
        evals.append(result)

    # format results
    results = format_results(evals, len(scenarios))
    return results