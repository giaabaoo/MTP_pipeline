from src.utils import setup_logger

from src.data_loader import get_data_loader
from src.segmentor import get_segmentor
from src.reasoner import get_reasoner
from src.scene_describer import get_scene_describer, get_frames_sampler

from tqdm import tqdm
import os
from datetime import datetime
from moviepy.editor import VideoFileClip
import json
import pdb

class Inference():
    def __init__(self, config):
        self.config = config
    
    def run(self):
        # Generate the output directory and save the full config file
        setup_logger(self.config)
        
        # Get all the necessary components
        data_loader = get_data_loader(self.config)
        segmentor = get_segmentor(self.config)
        reasoner = get_reasoner(self.config)
        frames_sampler = get_frames_sampler(self.config)
        scene_describer = get_scene_describer(self.config)
        
        for conversation_name, conversation_data in data_loader.get_items().items():
            if os.path.exists(conversation_data['output_path']):
                print(f"Skipping: {conversation_data['output_path']}, results are available")
                continue
            
            print(f"Processing {conversation_name}")
            # Recording processing time
            start = datetime.now()
            
            '''
            SEGMENTOR
            '''
            utterance_level_videos, utterance_level_transcripts = segmentor.run(conversation_name, conversation_data)

            '''
            SCENE DESCRIBER
            '''
            print("===================== SCENE DESCRIBING WITH LLAVA =====================")
            utterance_level_video_descriptions = scene_describer.main_run(conversation_data, utterance_level_videos)
            
            '''
            REASONER
            '''
            print("===================== REASONING WITH CHATGPT =====================")   
            print("===================== SUMMARIZING =====================")
            video_summary = reasoner.main_run_summarizer(conversation_data, utterance_level_video_descriptions, utterance_level_transcripts)
            

            print("===================== REASONING =====================")
            prediction, explanation = reasoner.main_run_reasoner(video_summary)
            
            # cut_video_summary = {k: video_summary[k] for k in list(video_summary.keys())[5:5+25]}
            # prediction, explanation = "", ""

    
            '''
            SAVE RESULTS
            '''
            time_processing = datetime.now() - start
    
            result = {"prediction": prediction, 
                      "explanation": explanation, 
                      'processing time': int(time_processing.total_seconds())
                      }        
            
            with open(conversation_data['output_path'], 'w') as fp:
                json.dump(result, fp, indent=4)
