from src.utils import setup_logger

from src.data_loader import get_data_loader
from src.reasoner import get_reasoner
from src.scene_describer import get_scene_describer, get_frames_sampler

from tqdm import tqdm
import os
from datetime import datetime
from moviepy.editor import VideoFileClip
import json
import pdb

class NonSegmentedSummarizerInference():
    def __init__(self, config):
        self.config = config
    
    def run(self):
        # Generate the output directory and save the full config file
        setup_logger(self.config)
        
        # Get all the necessary components
        data_loader = get_data_loader(self.config)
        reasoner = get_reasoner(self.config)
        frames_sampler = get_frames_sampler(self.config)
        scene_describer = get_scene_describer(self.config)
        
        for conversation_name, conversation_data in data_loader.get_items().items():
            '''
            GET NON-SEGMENTED DATA
            '''
            print(f"DETECTING TURNING POINTS FOR {conversation_name}")
            utterance_level_videos = conversation_data["utterance_level_videos"]
            output_video_description_path = conversation_data["output_video_description_file"]
            output_video_summary_path = conversation_data["output_video_summary_file"]
            output_file_path = conversation_data["output_file"]
            
            # Check if the output path exists
            if os.path.exists(output_file_path):
                print(f"Skipping: {output_file_path}, results are available")
                continue
            
            # Recording processing time
            start = datetime.now()
            
            '''
            SCENE DESCRIBER
            '''
            print("===================== SCENE DESCRIBING WITH LLAVA =====================")
            if os.path.exists(output_video_description_path):
                with open(output_video_description_path, 'r') as fp:
                    video_description = json.load(fp)
            else:
                video_description = {}
                for i, utterance_level_video in enumerate(tqdm(utterance_level_videos)):
                    video = VideoFileClip(utterance_level_video['video_path'])
                    
                    utterance_level_video_description = scene_describer.run(video, frames_sampler)
                    
                    utterance_name = f"utterance_{i}"
                        
                    video_description[utterance_name] = {'description' : utterance_level_video_description, \
                                            'transcript' : utterance_level_video['transcript']}
                
                if self.config.data.dataset_name == "DARPA":
                    with open(output_video_description_path, 'w', encoding='utf8') as fp:
                        json.dump(video_description, fp, indent=4, ensure_ascii=False)
                elif self.config.data.dataset_name == "TBBT":
                    with open(output_video_description_path, 'w') as fp:
                        json.dump(video_description, fp, indent=4)
            
            '''
            REASONER
            '''
            print("===================== REASONING WITH CHATGPT =====================")
            print("===================== SUMMARIZING =====================")
            if os.path.exists(output_video_summary_path):
                with open(output_video_description_path, 'r') as fp:
                    video_summary = json.load(fp)
            else:
                video_summary = reasoner.run_summarizer(video_description)
                if self.config.data.dataset_name == "DARPA":
                        with open(output_video_summary_path, 'w', encoding='utf8') as fp:
                            json.dump(video_summary, fp, indent=4, ensure_ascii=False)
                elif self.config.data.dataset_name == "TBBT":
                    with open(output_video_summary_path, 'w') as fp:
                        json.dump(video_summary, fp, indent=4)
                    
            '''
            SAVE RESULTS
            '''
            time_processing = datetime.now() - start
    
            result = {
                      'processing time': int(time_processing.total_seconds())
                      }        
            
            with open(output_file_path, 'w') as fp:
                json.dump(result, fp, indent=4)