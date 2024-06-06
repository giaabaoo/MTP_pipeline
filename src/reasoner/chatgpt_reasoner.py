import openai  
from tqdm import tqdm
import time
import pdb
import re
import json
import os
import tiktoken
from openai import OpenAI

class ChatGPTReasoner:
    def __init__(self, config):
        self.config = config
        openai.api_key = config.api_key
        
        self.enc = tiktoken.get_encoding("cl100k_base")
        
    def main_run_reasoner(self, video_summary):
        try:
            prediction = self.detect_turning_point(video_summary)
            
            prediction, explanation = self.postprocess(prediction)
        except Exception as e:
            prediction, explanation = e, e
            
        return prediction, explanation
    
    
    def run_reasoner(self, video_summary):
        prediction = self.detect_turning_point(video_summary)
        
        prediction, explanation = self.postprocess(prediction)
        return prediction, explanation
        
    def run_summarizer(self, video_description):
        video_summary = {}
        for utterance_video_name, utterance_video_content in tqdm(video_description.items()):
            video_description_summary = self.summarize_utterance_description(utterance_video_content['description'])
            refined_video_description_summary = self.postprocess_summary(video_description_summary)
            video_summary[utterance_video_name] = {'description' : refined_video_description_summary, \
                                        'transcript' : utterance_video_content['transcript']}
        
        return video_summary
        
    def main_run_summarizer_transcripts_only(self, conversation_data, utterance_level_videos, utterance_level_transcripts):
        video_summary = {}
        output_video_summary_path = conversation_data['output_video_summary_path']
        if os.path.exists(output_video_summary_path):
            with open(output_video_summary_path, 'r') as fp:
                video_summary = json.load(fp)
        else:
            for i, (utterance_video) in tqdm(enumerate(utterance_level_videos), total=len(utterance_level_videos)):
                utterance_video_name = utterance_video.split("/")[-1].replace(".jpg", "")
                with open(utterance_level_transcripts, "r") as f:
                    transcripts = json.load(f)
                transcript = transcripts["segments"][i]["text"]

                video_summary[utterance_video_name] = {'transcript' : transcript}
            
            with open(output_video_summary_path, 'w') as fp:
                json.dump(video_summary, fp, indent=4)
        
        return video_summary
    
    def main_run_summarizer(self, conversation_data, utterance_level_video_descriptions, utterance_level_transcripts):
        video_summary = {}
        output_video_summary_path = conversation_data['output_video_summary_path']
        if os.path.exists(output_video_summary_path):
            with open(output_video_summary_path, 'r') as fp:
                video_summary = json.load(fp)
        else:
            utterance_level_video_descriptions_items = list(utterance_level_video_descriptions.items())  # Convert to list

            for i, (utterance_video_name, utterance_video_content) in tqdm(enumerate(utterance_level_video_descriptions_items), total=len(utterance_level_video_descriptions_items)):
                with open(utterance_level_transcripts, "r") as f:
                    transcripts = json.load(f)
                transcript = transcripts["segments"][i]["text"]

                video_description_summary = self.main_summarize_utterance_description(utterance_video_content)
                # refined_video_description_summary = self.postprocess_summary(video_description_summary)
                video_summary[utterance_video_name] = {'visual_description' : video_description_summary, \
                                            'transcript' : transcript}
            
            with open(output_video_summary_path, 'w') as fp:
                json.dump(video_summary, fp, indent=4)
        
        return video_summary
    
    def detect_turning_point(self, summary):
        prompt = f"Given this definition: {self.config.turning_point_definition} \n \
            Execute the command {self.config.command}. \n \
            And this is the conversation: {summary}. Give me the output in this format: \
            [ANS] The event that cause the turning point is [the name of the event], in utterance [the name of the utterance] [/ANS]. \n \
            [EXP] [Fill in the explanations] [/EXP]. \n \
            Return [ANS] None [/ANS]. [EXP] None [/EXP] if the evidence is weak. "
            
        retries = 3    
        while retries > 0:    
            try: 
                dic = {'role': 'user', 'content': prompt}
                completion = openai.ChatCompletion.create(model=self.config.model_engine, messages = [dic])
                prediction = completion.choices[0].message['content']
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
        bad_api = "x"  
        return bad_api
        
    
    def summarize_utterance_description(self, utterance_description):
        '''
        Summarize the second-level descriptions considering the actions, gestures, postures, emotions
        '''
        
        prompt = f"From this descriptions {utterance_description}. \n \
                Extract the exact words regarding actions, facial expressions, gestures, postures, and potential emotions \
                for each scene. \
                For example: \
                Scene 1:  \n \
                Person sitting in a dark room, looking at the camera. \n \
                Neutral facial expression, indicating no strong emotions. \n \
                Engaged in a conversation or interaction. \n \
                Comfortable and relaxed posture. \n \
                Awareness of the camera's presence. \n\n"
                
        dic = {'role': 'user', 'content': prompt}
        completion = openai.ChatCompletion.create(model=self.config.model_engine, messages = [dic])
        video_description_summary = completion.choices[0].message['content']
        time.sleep(60)
        
        return video_description_summary
    
    def main_summarize_utterance_description(self, utterance_description):
        '''
        Summarize the second-level descriptions considering the actions, gestures, postures, emotions
        '''
        
        prompt = f"From this descriptions {utterance_description}. \n \
                Try to guess the possible emotions and actions with evidence. Answer in this format (max 10 words): \
                Example: \
                Input: \
                - In the image, a person is standing in a hallway with a door in front of them. They are holding a cell phone in their hand, possibly looking at it or using it. The person's facial expression is a mix of surprise and curiosity, as they seem to be intrigued by something they see or hear on the phone. The person's posture and body language suggest that they are engaged in a conversation or interaction with the device. The potential emotions include surprise, curiosity, and interest, as the person is intrigued by the content on the phone.\
                Output: \
                - Emotions: surprise (facial expression), curiosity (their gaze).  \
                - Actions: Using a cell phone. \n"
            
        
        retries = 3    
        while retries > 0:    
            try: 
                dic = {'role': 'user', 'content': prompt}
                client = OpenAI(api_key = self.config.api_key)
                completion = client.chat.completions.create(model=self.config.model_engine, messages = [dic])
                video_description_summary = completion.choices[0].message.content  
                return video_description_summary

            except Exception as e:    
                if e: 
                    print(e)   
                    print('Timeout error, retrying...') 
                    retries -= 1    
                    time.sleep(2)    
                else:    
                    raise e    
        print('API is not responding, moving on...')   
        bad_api = "x"  
        return bad_api
    
    # def summarize_conversation(self, video_summary):
    #     summary = ""
    #     for i, utterance_description, transcript in enumerate(video_summary):
    #         tmp_text = f"<BEGIN_UTTERANCE> \n Utterance {i}: \n"
    #         summary += tmp_text + f"Description {utterance_description} \n" + f"Transcript {transcript} \n" + "<END_UTTERANCE> \n\n"
        
    #     return summary
    
    def postprocess_summary(self, summary):
        refined_summary = {}
        scenes = re.split(r"\n(?=Scene \d+:)", summary)
        refined_summary = {scene.strip().split('\n')[0]: [line.lstrip('- ').strip() for line in scene.strip().split('\n')[1:]] for scene in scenes}
        
        return refined_summary
        
    def postprocess(self, prediction):
        # Regular expression patterns for answer and explanation extraction
        answer_pattern = r'\[ANS\](.*?)\[/ANS\]'
        explanation_pattern = r'\[EXP\](.*?)$'

        answer_match = re.search(answer_pattern, prediction, re.DOTALL)
        explanation_match = re.search(explanation_pattern, prediction, re.DOTALL)

        answer = answer_match.group(1).strip() if answer_match else prediction
        explanation = explanation_match.group(1).strip() if explanation_match else prediction
            
        return answer, explanation
    
    def count_tokens(self, summary):
        prompt = f"Given this definition: {self.config['turning_point_definition']} \n \
            Execute the command {self.config['command']}. \n \
            And this is the conversation: {summary}. Give me the output in this format: \
            [ANS] The event that cause the turning point is [the name of the event], in utterance [the name of the utterance] [/ANS]. \n \
            [EXP] [Fill in the explanations] [/EXP]. \n \
            Return [ANS] None [/ANS]. [EXP] None [/EXP] if the evidence is weak. "
        
        print("Number of tokens: ", len(self.enc.decode(self.enc.encode(prompt))))
        
        dic = {'role': 'user', 'content': prompt}
        completion = openai.ChatCompletion.create(model=self.config['model_engine'], messages = [dic])
        prediction = completion.choices[0].message['content']
        
        
        print(prediction)

import yaml

if __name__ == "__main__":
    config_path = "/home/dhgbao/Research_Monash/code/others/my_code/methods_experiments/multimodal/conversational_turning_point_detection/multimodal_TPD/configs/reasoner/reasoner.yml"
    summary_path = "/home/dhgbao/Research_Monash/code/others/my_code/methods_experiments/multimodal/conversational_turning_point_detection/multimodal_TPD/results/inference/Segmentorwhisperx-SceneDescriber_LLAVA-Reasoner_chatgpt-Data_CCTP/video_summaries/conversation_1.json"
    # Open and read the YAML file
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)['reasoner']
    
    openai.api_key = config['api_key']
    
    reasoner = ChatGPTReasoner(config=config)
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    reasoner.count_tokens(summary)