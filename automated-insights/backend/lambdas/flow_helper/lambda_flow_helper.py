import json

# extract_string: 
# Use this if you have a prompt node that outputs a string which you want to send to an iterator node. 
# This is not possible by default because a prompt node can only output type 'string', 
# whilst the iterator node can only input type 'array'.

def extract_string(input_json, key):
    try:
        data = input_json[key]
        print("Extracted value:", data)
        return data
    except KeyError as e:
        raise ValueError(f"Missing expected key in JSON structure: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing JSON: {str(e)}")

# json_string_to_array: 
# Use this if you have a prompt node that outputs JSON objects which you want to send to an iterator node. 
# This is not possible by default because a prompt node can only output type 'string', 
# whilst the iterator node can only input type 'array'.

def extract_array(input_json):
    try:
        data = input_json
        print("Extracted array:", json.dumps(data, indent=4))
        return data
    except KeyError as e:
        raise ValueError(f"Missing expected key in JSON structure: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing JSON: {str(e)}")
    

def lambda_handler(event, context):
    try:
        # get request type from event
        request_type = event["node"]["inputs"][0]["value"]
        print(f"Request type: {request_type}")

        input_value = event["node"]["inputs"][1]["value"]
        print(f"Input value: {input_value}")
        
        key = event["node"]["inputs"][2]["value"]
        print(f"Key: {key}")

        if request_type == "extract_string":
            
            result = extract_string(input_value, key)

        elif request_type == "extract_array":
            result = extract_array(input_value)

        else:
            raise ValueError(f"Invalid request type: {request_type}")
        
    except ValueError as e:
        print(f"Error: {str(e)}")
    return result