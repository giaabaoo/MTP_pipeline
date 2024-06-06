from src.utils import Config
import os
from pathlib import Path
import json
import pdb
from tqdm import tqdm
import base64
import requests
import time
from openai import OpenAI
import openai


class GPT4VSceneDescriber:
    def __init__(self, config: Config):
        self.config = config
        openai.api_key = config.api_key
        self.prompt = config.prompt
        # Path("./tmp/").mkdir(exist_ok=True, parents=True)
  
    def main_run(self, conversation_data, utterance_level_videos, utterance_level_transcripts):
        output_video_description_path = conversation_data['output_video_description_path']
        video_descriptions = {}
        if os.path.exists(output_video_description_path):
            with open(output_video_description_path, 'r') as fp:
                video_descriptions = json.load(fp)
        else:                    
            for i, utterance in enumerate(tqdm(utterance_level_videos)):
                self.config.image_file = utterance
                
                message = self.get_message(self.config.image_file)

                utterance_name = f"utterance_{i+1}"
                video_descriptions[utterance_name] = message

            with open(output_video_description_path, 'w') as fp:
                json.dump(video_descriptions, fp, indent=4)
        
        video_summary = {}
        output_video_summary_path = conversation_data['output_video_summary_path']
        if os.path.exists(output_video_summary_path):
            with open(output_video_summary_path, 'r') as fp:
                video_summary = json.load(fp)
        else:
            utterance_level_video_descriptions_items = list(video_descriptions.items())  # Convert to list

            for i, (utterance_video_name, utterance_video_content) in tqdm(enumerate(utterance_level_video_descriptions_items), total=len(utterance_level_video_descriptions_items)):
                with open(utterance_level_transcripts, "r") as f:
                    transcripts = json.load(f)
                    
                transcript = transcripts["segments"][i]["text"]

                video_summary[utterance_video_name] = {'visual_description' : utterance_video_content, \
                                            'transcript' : transcript}
            
            with open(output_video_summary_path, 'w') as fp:
                json.dump(video_summary, fp, indent=4)
        
        return video_descriptions, video_summary
    
    def get_message(self, image_path):
        base64_image = self.encode_image(image_path)


        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer <insert_your_api_key_here>"
        }

        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"{self.prompt}"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        retries = 3    
        while retries > 0:    
            try: 
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                return response.json()['choices'][0]['message']['content']

            except Exception as e:    
                if e: 
                    print(e)   
                    print('Timeout error, retrying...')   
 
                    retries -= 1    
                    time.sleep(180)    
                else:    
                    raise e    
                
        print('API is not responding, moving on...')   
        bad_api = "There is no visual information for this utterance"  
        return bad_api
        
    
    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
                
    
if __name__ == "__main__":
    config = {}  # Provide your configuration here
    scene_describer = GPT4VSceneDescriber(config)
    
    print(scene_describer.get_message("image.png"))