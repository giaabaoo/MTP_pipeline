import openai
from tqdm import tqdm
import pandas as pd
import pdb
import json
import os
import time
from openai import OpenAI
from pathlib import Path

class ChatGPTZeroShotEvaluator:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key = self.config.api_key)
        Path(self.config.evaluated_conversations_path).mkdir(exist_ok=True, parents=True)

    def refine_predictions(self, conversation,  prediction):
        user_content = f"Read this conversation: {conversation} \n and these detection results {prediction} \n {self.config.instruction_4}"
        
        retries = 3    
        while retries > 0:    
            try: 
                completion = self.client.chat.completions.create(
                    model=self.config.model_engine, 
                    messages =[
                    {"role": "system", "content": self.config.system_content},
                    {"role": "user", "content": user_content}],
                    temperature = 1)

                answer = completion.choices[0].message.content  
                
                return answer
            except Exception as e:    
                if e: 
                    print(e)   
                    pdb.set_trace()

                    print('Timeout error, retrying...')    
                    retries -= 1    
                    time.sleep(2)    
                else:    
                    raise e    
                
        print('API is not responding, moving on...')   
        bad_api = "bad_api"  
        return bad_api
    
        
         
    def predict(self, conversation):
        if self.config.model_engine == "gpt-3.5-turbo":
            user_content = f"{conversation} \n {self.config.instruction_2} \n {self.config.instruction_3}"
            
        elif self.config.tracking == False:
            user_content = f"{self.config.instruction_1}: {conversation} \n {self.config.instruction_3}"
        elif self.config.fewshot == True:
            user_content = f"{self.config.instruction_1}: {conversation} \n {self.config.instruction_2} \n {self.config.instruction_3}. {self.config.fewshot_examples}"
        else:
            user_content = f"{self.config.instruction_1}: {conversation} \n {self.config.instruction_2} \n {self.config.instruction_3}"

        retries = 3    
        while retries > 0:    
            try: 
                completion = self.client.chat.completions.create(
                    model=self.config.model_engine, 
                    messages =[
                    {"role": "system", "content": self.config.system_content},
                    {"role": "user", "content": user_content}], temperature=1)

                answer = completion.choices[0].message.content  
                
                return answer
            except Exception as e:    
                if e: 
                    print(e)   
                    pdb.set_trace()

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
        
        total = 0

        with open(self.config.test_data_path, 'r', encoding='utf-8') as jsonl_file:
            data = [json.loads(line) for line in jsonl_file]

            for i, item in enumerate(tqdm(data, desc="Processing"), start=1):
                results = {}
                conversation, label = item['summary'], item['label']
                conversation_file = f"conversation_{i}.json"
                conversation_file_path = os.path.join(self.config.evaluated_conversations_path, conversation_file)
                if os.path.exists(conversation_file_path):
                    with open(os.path.join(conversation_file_path), "r") as f:
                        data = json.load(f)
                    prediction, detection_results, label = data['prediction'], data['detection_results'], data['label']
                else:
                    prediction = self.predict(conversation)
                    if "bad_api" in prediction:
                        continue
                    
                    # refined_prediction = self.refine_predictions(conversation, prediction)
                    # refined_prediction = ""
                    # if "bad_api" in refined_prediction:
                    #     continue
                    
                    
                    detection_results  = self.locate_utterance(conversation, prediction)
                    
                    # deprecated
                    # response = self.validate_context(conversation, refined_prediction, label)
                    # response = ""
                    
                    if "bad_api" in detection_results:
                        continue
                    # try:
                    #     # conclusion, explanation = response['conclusion'], response['explanation']
                    #     # conclusion, explanation = "", ""
                    # except:
                    #     pdb.set_trace()
                        
                    results = {'prediction': prediction, 'detection_results': detection_results, 'label': label}
                    
                    print("Prediction: ", prediction)
                    print("\n")
                    print("Label: ", label)
                    print("\n")
             
                    
                    with open(conversation_file_path, "w") as f:
                        json.dump(results, f, indent=4)
                    
                    Path(self.config.prediction_conversations_path).mkdir(exist_ok=True, parents=True)
                    prediction_conversation_file_path = os.path.join(self.config.prediction_conversations_path, conversation_file)
                    with open(prediction_conversation_file_path, "w") as f:
                        json.dump(prediction, f, indent=4)

                    
                # if "true" in conclusion.lower() or "yes" in conclusion.lower():
                #     correct += 1
                    
                total += 1

        acc = correct / total
        
        final_results = {}
        final_results['accuracy'] = f"Accuracy: {acc}"
        print("Accuracy: ", acc)

        return final_results

    def validate_context(self, conversation, prediction, label):
        custom_functions = [
        {
            "name": "validate_context",
            "description": "Validate the semantic accuracy of the prediction in relation to the label.",
            "parameters": {
                "type": "object",
                "properties": {
                "conclusion": {
                    "type": "string",
                    "description": "True (same turning point event) or False (different turning point event)"
                },
                "explanation": {
                    "type": "string",
                    "description": "A rationale for your conclusion"
                }
                },
                "required": ["conclusion"]
            }
        }

        ]
        
        sample = f"{self.config.instruction_1}: \n {conversation} \n \
                label = {label} \n \
                prediction = {prediction} \n \
                Is any turning point event in the prediction correlated to the label (pointing to the same event in the conversation)?"
                    
                # Does the label and prediction reach the same conclusion? Specifically, if both texts state that there is no turning point, return 'yes' because the texts agree. In the case of a predicted turning point event, attempt to determine whether it is indicated by the label. If any event is indicated by the label, also return 'yes.' However, if the prediction mentions the possibility of no significant turning point but still suggests some potential, ignore those instances and conclude 'no turning point' for that prediction. Then, compare to see if the ground truth agrees or not."

        
        retries = 3    
        while retries > 0:    
            try:     
                response = self.client.chat.completions.create(
                    model=self.config.model_engine, 
                    messages = [{"role": "system", "content": "You are a turning point detection evaluator"}, 
                                {'role': 'user', 'content': sample}],
                    functions = custom_functions,
                    function_call = 'auto',
                    temperature=0)

                json_response = json.loads(response.choices[0].message.function_call.arguments)
                
                return json_response
            except Exception as e:    
                if e: 
                    print(e)   
                    # pdb.set_trace()
                    print('Timeout error, retrying...')    
                    retries -= 1    
                    time.sleep(2)    
                else:    
                    raise e    
        
        print('API is not responding, moving on...')   
        bad_api = "bad_api"  
        return bad_api
    
    def locate_utterance(self, conversation, prediction):
        custom_functions = [
        {
            "name": "locate_utterance",
            "description": "Locate the start utterance that cause the turning points",
            "parameters": {
                "type": "object",
                "properties": {
                "utterances": {
                    "type": "string",
                    "description": "A list of utterances"
                }
                },
                "required": ["utterances"]
            }
        }

        ]
        
        sample = f"{self.config.instruction_1}: \n {conversation} \n \
                prediction = {prediction} \n \
                For each found turning point in the prediction, find the starting utterance index only. Return a list of n utterance start indices corresponding to a turning point in the prediction. Follow strictly this format in your response: e.g. utterances = [utterance_5, utterance_25]. Return None if there is no turning point found. Limit the response to 50 words. "
                # Does the label and prediction reach the same conclusion? Specifically, if both texts state that there is no turning point, return 'yes' because the texts agree. In the case of a predicted turning point event, attempt to determine whether it is indicated by the label. If any event is indicated by the label, also return 'yes.' However, if the prediction mentions the possibility of no significant turning point but still suggests some potential, ignore those instances and conclude 'no turning point' for that prediction. Then, compare to see if the ground truth agrees or not."

        retries = 3    
        while retries > 0:    
            try:     
                response = self.client.chat.completions.create(
                    model="gpt-4", 
                    messages = [{"role": "system", "content": "You are a turning point localizer"}, 
                                {'role': 'user', 'content': sample}],
                    functions = custom_functions,
                    function_call = 'auto',
                    temperature=0)

                json_response = json.loads(response.choices[0].message.function_call.arguments)
                
                return json_response
            except Exception as e:    
                if e: 
                    print(e)   
                    # pdb.set_trace()
                    print('Timeout error, retrying...')    
                    retries -= 1    
                    time.sleep(2)    
                else:    
                    raise e    
        
        print('API is not responding, moving on...')   
        bad_api = "bad_api"  
        return bad_api