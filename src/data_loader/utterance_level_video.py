import os
import os.path as osp
import json
from pathlib import Path
import pdb

class UtteranceLevelVideoLoader:
    def __init__(self, config):
        self.config = config
        self.output_path = osp.join(config.output_dir, config.data.output_path)
        self.output_video_description_path = osp.join(config.output_dir, config.data.output_video_description_path)
        self.output_video_summary_path = osp.join(config.output_dir, config.data.output_video_summary_path)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.output_video_description_path).mkdir(parents=True, exist_ok=True)
        Path(self.output_video_summary_path).mkdir(parents=True, exist_ok=True)
        
    def get_items(self):
        items_dict = dict()
        
        for conversation in sorted(os.listdir(self.config.data.input_path)):
            utterance_level_videos_list = []
            for utterance_level_video in sorted(os.listdir(os.path.join(self.config.data.input_path, conversation))):
                video_name = utterance_level_video.split(".")[0]
                
                video_file_path = os.path.join(self.config.data.input_path, conversation, video_name+".mp4")
                transcript = self.get_transcript(video_name)
                
                utterance_level_videos_list.append({'video_path' : video_file_path, \
                                                    'transcript' : transcript})
                
            output_file_path = os.path.join(self.output_path, conversation + ".json")
            output_video_description_file_path = os.path.join(self.output_video_description_path, conversation + ".json")
            output_video_summary_file_path = os.path.join(self.output_video_summary_path, conversation + ".json")
            
            if conversation not in items_dict:
                # If not, create the 'conversation' key with an empty dictionary
                items_dict[conversation] = {}
                
            items_dict[conversation]['utterance_level_videos'] = utterance_level_videos_list
            items_dict[conversation]['output_video_description_file'] = output_video_description_file_path
            items_dict[conversation]['output_video_summary_file'] = output_video_summary_file_path
            items_dict[conversation]['output_file'] = output_file_path

        return items_dict
                
    def get_transcript(self, video_name):
        with open(self.config.data.transcript_path, "r") as f:
            data = json.load(f)
        
        video_data = data[video_name]
        
        sentences, speakers = video_data['sentences'], video_data['speakers']
        
        transcript_list = []
        for speaker, sentence in zip(speakers, sentences):
            transcript_list.append(f"Speaker {speaker} : {sentence}")
        
        return transcript_list
        