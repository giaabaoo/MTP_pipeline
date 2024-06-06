import os
import os.path as osp
import json
from pathlib import Path
import pdb
import natsort
class VideoLoader:
    def __init__(self, config):
        self.config = config
        self.output_path = osp.join(config.output_dir, config.data.output_path)
        self.output_video_description_path = osp.join(config.output_dir, config.data.output_video_description_path)
        self.output_video_summary_path = osp.join(config.output_dir, config.data.output_video_summary_path)
        
        self.utterance_video_path = self.config.data.utterance_video_path
        self.utterance_transcript_path = self.config.data.utterance_transcript_path
        Path(self.utterance_video_path).mkdir(parents=True, exist_ok=True)
        Path(self.utterance_transcript_path).mkdir(parents=True, exist_ok=True)
        
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.output_video_description_path).mkdir(parents=True, exist_ok=True)
        Path(self.output_video_summary_path).mkdir(parents=True, exist_ok=True)

    def get_items(self):
        items_dict = dict()
        
        for conversation_file in natsort.natsorted(os.listdir(self.config.data.input_path)):
            conversation_name = conversation_file.replace(".mp4", "")
            if conversation_name not in items_dict:
                # If not, create the 'conversation' key with an empty dictionary
                items_dict[conversation_name] = {}
            items_dict[conversation_name]['input_path'] = os.path.join(self.config.data.input_path, conversation_file)
            items_dict[conversation_name]['utterance_video_path'] = os.path.join(self.utterance_video_path, conversation_name)
            items_dict[conversation_name]['utterance_transcript_path'] = os.path.join(self.utterance_transcript_path, conversation_name) 
            items_dict[conversation_name]['output_path'] = os.path.join(self.output_path, conversation_name + ".json")
            items_dict[conversation_name]['output_video_description_path'] = os.path.join(self.output_video_description_path, conversation_name + ".json")
            items_dict[conversation_name]['output_video_summary_path'] = os.path.join(self.output_video_summary_path, conversation_name + ".json")

        return items_dict

