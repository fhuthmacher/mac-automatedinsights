from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3, json
import time
from botocore.config import Config
import base64
from PIL import Image
from io import BytesIO

class BedrockLLMWrapper():
    def __init__(self,
        model_id: str = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        embedding_model_id: str = 'amazon.titan-embed-image-v1',
        system_prompt: str = 'You are a helpful AI Assistant.',
        region: str = 'us-east-1',
        top_k: int = 5,
        top_p: int = 0.7,
        temperature: float = 0.0,
        max_token_count: int = 4000,
        max_attempts: int = 10,
        debug: bool = False,
        session: boto3.Session = None

    ):

        
        self.embedding_model_id = embedding_model_id
        self.system_prompt = system_prompt
        self.region = region
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_token_count = max_token_count
        self.max_attempts = max_attempts
        self.debug = debug
        config = Config(
            retries = {
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )
        if session is None:
            self.bedrock_runtime = boto3.client(service_name="bedrock-runtime", config=config, region_name=self.region)
        else:
            self.bedrock_runtime = session.client(service_name="bedrock-runtime", config=config, region_name=self.region)

        
        self.model_id = model_id

    def get_valid_format(self, file_format):
        format_mapping = {
            'jpg': 'jpeg',
            'gif': 'gif',
            'png': 'png',
            'webp': 'webp'
        }
        return format_mapping.get(file_format.lower(), 'jpeg')  # Default to 'jpeg' if format is not recognized
    
    def process_image(self, image_path, max_size=(512, 512)):
        with open(image_path, "rb") as image_file:
            # Read the image file
            image = image_file.read()
            image = Image.open(BytesIO(image)).convert("RGB")
            
            # Resize image while maintaining aspect ratio
            image.thumbnail(max_size, Image.LANCZOS)
            
            # Create a new image with the target size and paste the resized image
            new_image = Image.new("RGB", max_size, (255, 255, 255))
            new_image.paste(image, ((max_size[0] - image.size[0]) // 2,
                                    (max_size[1] - image.size[1]) // 2))
            
            # Save to BytesIO object
            buffered = BytesIO()
            new_image.save(buffered, format="JPEG")
            
            # Encode to base64
            input_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf8')
        
        return input_image_base64

    def get_embedding(self, input_text=None, image_path=None):
        '''
        This function is used to generate the embeddings for a specific chunk of text
        '''
        accept = 'application/json'
        contentType = 'application/json'
        request_body = {}

        if input_text:
            request_body["inputText"] = input_text
        if image_path:
            # Process and encode the image
            img_base64 = self.process_image(image_path)
            request_body["inputImage"] = img_base64

        # request_body["dimensions"] = 1024
        # request_body["normalize"] = True

        if 'amazon' in self.embedding_model_id:
            embeddingInput = json.dumps(request_body)
            response = self.bedrock_runtime.invoke_model(body=embeddingInput, 
                                                        modelId=self.embedding_model_id, 
                                                        accept=accept, 
                                                        contentType=contentType)
            embeddingVector = json.loads(response['body'].read().decode('utf8'))
            return embeddingVector['embedding']
                
        if 'cohere' in self.embedding_model_id:
            request_body["input_type"] = "search_document" # |search_query|classification|clustering
            request_body["truncate"] = "NONE" # NONE|START|END
            embeddingInput = json.dumps(request_body)
    
            response = self.bedrock_runtime.invoke_model(body=embeddingInput, 
                                                            modelId=self.embedding_model_id, 
                                                            accept=accept, 
                                                            contentType=contentType)
    
            response_body = json.loads(response.get('body').read())
            # print(response_body)
            embeddingVector = response_body['embedding']
            
            return embeddingVector
    
    def generate(self,prompt,image_file=None, image_file2=None):
        if self.debug: 
            print('entered BedrockLLMWrapper generate')
        message = {}
        attempt = 1
        if image_file is not None:
            if self.debug: 
                print('processing image1: ', image_file)
            # extract file format from the image file
            file_format = image_file.split('.')[-1]
            valid_format = self.get_valid_format(file_format)

            # Open and read the image file
            with open(image_file, 'rb') as img_file:
                image_bytes = img_file.read()
                if self.debug: 
                    print('image_bytes: ', image_bytes)
                    print('valid_format: ', valid_format)

            message = {
                "role": "user",
                "content": [
                    { "text": "Image 1:" },
                    {
                        "image": {
                            "format": valid_format,
                            "source": {
                                "bytes": image_bytes 
                            }
                        }
                    },
                    { "text": prompt }
                ],
                    }
            
        if image_file is not None and image_file2 is not None:
            if self.debug: 
                print('processing image2: ', image_file2)
            # extract file format from the image file
            file_format2 = image_file2.split('.')[-1]
            valid_format2 = self.get_valid_format(file_format2)

            with open(image_file2, 'rb') as img_file:
                image_bytes2 = img_file.read()
                if self.debug: 
                    print('image_bytes2: ', image_bytes2)
                    print('valid_format2: ', valid_format2)
            
            message = {
            "role": "user",
            "content": [
                { "text": "Image 1:" },
                {
                    "image": {
                        "format": valid_format,
                        "source": {
                            "bytes": image_bytes 
                        }
                    }
                },
                { "text": "Image 2:" },
                {
                    "image": {
                        "format": valid_format2,
                        "source": {
                            "bytes": image_bytes2 
                        }
                    }
                },
                { "text": prompt }
            ],
                } 
            
        if image_file is None and image_file2 is None:
            message = {
                "role": "user",
                "content": [{"text": prompt}]
            }
        messages = []
        messages.append(message)
        
        # model specific inference parameters to use.
        if "anthropic" in self.model_id.lower():
            system_prompts = [{"text": self.system_prompt}]
            # Base inference parameters to use.
            inference_config = {
                                "temperature": self.temperature, 
                                "maxTokens": self.max_token_count,
                                "stopSequences": ["\n\nHuman:"],
                                "topP": self.top_p,
                            }
            additional_model_fields = {"top_k": self.top_k}
        else:
            system_prompts = []
            # Base inference parameters to use.
            inference_config = {
                                "temperature": self.temperature, 
                                "maxTokens": self.max_token_count,
                            }
            additional_model_fields = {}

        if self.debug: 
            print("Sending:\nSystem:\n",system_prompts,"\nMessages:\n",str(messages))

        while True:
            try:
                # print(f"model_id: {self.model_id}")

                # Send the message.
                response = self.bedrock_runtime.converse(
                    modelId=self.model_id,
                    messages=messages,
                    system=system_prompts,
                    inferenceConfig=inference_config,
                    additionalModelRequestFields=additional_model_fields
                )

                # Log token usage.
                text = response['output'].get('message').get('content')[0].get('text')
                usage = response['usage']
                latency = response['metrics'].get('latencyMs')

                if self.debug: 
                    print(f'text: {text} ; and token usage: {usage} ; and query_time: {latency}')    
                
                break
               
            except Exception as e:
                print("Error with calling Bedrock: "+str(e))
                attempt+=1
                if attempt>self.max_attempts:
                    print("Max attempts reached!")
                    result_text = str(e)
                    break
                else:#retry in 10 seconds
                    print("retry")
                    time.sleep(60)

        # return result_text
        return [text,usage,latency]

     # Threaded function for queue processing.
    def thread_request(self, q, results):
        while True:
            try:
                index, prompt = q.get(block=False)
                data = self.generate(prompt)
                results[index] = data
            except Queue.Empty:
                break
            except Exception as e:
                print(f'Error with prompt: {str(e)}')
                results[index] = str(e)
            finally:
                q.task_done()

 
    def generate_threaded(self, prompts, images=None, max_workers=15):
        
        if images is None:
            images = [None] * len(prompts)
        elif len(prompts) != len(images):
            raise ValueError("The number of prompts must match the number of images (or images must be None)")

        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            future_to_index = {executor.submit(self.generate, prompt, image): i 
                               for i, (prompt, image) in enumerate(zip(prompts, images))}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
                    results[index] = str(exc)
        
        return results