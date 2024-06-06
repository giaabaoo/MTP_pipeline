import openai
from tqdm import tqdm
import pandas as pd
import pdb
import json
import time
import os

class ChatGPTEvaluator:
    def __init__(self, config):
        self.config = config
        openai.api_key = config.api_key
        
    def predict(self, conversation):
        if self.config.use_instructions:
            user_content = f"This is the conversation: \n {conversation} \n {self.config.instructions}"
        else:
            user_content = f"Help me find the turning point in this conversation: {conversation}"
        
        retries = 3    
        while retries > 0:    
            try: 
                completion = openai.ChatCompletion.create(
                model=self.config.finetuned_model_name,
                messages=[
                    {"role": "system", "content": self.config.system_content},
                    {"role": "user", "content": user_content}
                ]
                )
                prediction = completion.choices[0].message
                return prediction
            except Exception as e:    
                if e: 
                    print(e)   
                    print('Timeout error, retrying...')    
                    retries -= 1    
                    time.sleep(2)    
                else:    
                    raise e    
                
        print('API is not responding, moving on...')   
        bad_api = "bad_api"  
        return bad_api
        
        
    def run(self):
        correct = 0
        results = {}
        total = 0
        with open(self.config.test_data_path, 'r', encoding='utf-8') as jsonl_file:
            data = [json.loads(line) for line in jsonl_file]

            for i, item in tqdm(enumerate(data, start=1), total=total, desc="Processing"):
                conversation, label = item['summary'], item['label']
                prediction = self.predict(conversation)
                if "bad_api" in prediction:
                    continue
                
                response = self.validate_context(prediction, label)
                
                if "bad_api" in response:
                    continue
                
                conclusion, explanation = response['conclusion'], response['explanation']
                if "true" in conclusion.lower() or "yes" in conclusion.lower():
                    correct += 1

                test_idx = f"sample_{i}"
                results[test_idx] = {'prediction': prediction, 'label': label, 'conclusion': conclusion, 'explanation': explanation}
                total += 1
                
                print("Prediction: ", prediction)
                print("\n")
                print("Label: ", label)
                print("\n")
                print("Conclusion: ", conclusion)
                print("\n")
                print("Explanation: ", explanation)
                print("\n")

        acc = correct / total
        results['accuracy'] = f"Accuracy: {acc}"
        print("Accuracy: ", acc)

        return results

    def validate_context(self, prediction, label):
        custom_functions = [
            {
                'name': 'validate_context',
                'description': 'Validate the semantic accuracy of the prediction in relation to the label. ',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'conclusion': {
                            'type': 'string',
                            'description': 'True (same turning point event) or False (different turning point event)'
                        },
                        'explanation': {
                            'type': 'string',
                            'description': 'A rationale for your conclusion'
                        }
                        
                    }
                }
            }
        ]
        
        sample = f"This is the prediction: {prediction} AND this is the label: {label}. Do they somewhat have the same turning point event?"
        
        retries = 3    
        while retries > 0:    
            try: 
                response = openai.ChatCompletion.create(
                    model = 'gpt-4',
                    messages = [{'role': 'user', 'content': sample}],
                    functions = custom_functions,
                    function_call = 'auto')
                
                json_response = json.loads(response['choices'][0]['message']['function_call']['arguments'])

                return json_response
            except Exception as e:    
                if e: 
                    print(e)   
                    print('Timeout error, retrying...')    
                    retries -= 1    
                    time.sleep(2)    
                else:    
                    raise e    
        
        print('API is not responding, moving on...')   
        bad_api = "bad_api"  
        return bad_api