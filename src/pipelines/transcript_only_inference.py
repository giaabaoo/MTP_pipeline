from src.utils import setup_logger

from src.data_loader import get_data_loader
from src.reasoner import get_reasoner

from tqdm import tqdm
import os
from datetime import datetime
import json
import pdb

class TranscriptOnlyInference():
    def __init__(self, config):
        self.config = config
    
    def run(self):
        # Generate the output directory and save the full config file
        setup_logger(self.config)
        
        # Get all the necessary components
        data_loader = get_data_loader(self.config)
        reasoner = get_reasoner(self.config)
                
        for conversation_name, conversation_data in tqdm(data_loader.get_items().items()):
            '''
            GET NON-SEGMENTED DATA
            '''
            print(f"DETECTING TURNING POINTS FOR {conversation_name}")
            utterance_level_transcripts = conversation_data["transcript"]
            output_file_path = conversation_data["output_file"]
            
            # Check if the output path exists
            if os.path.exists(output_file_path):
                print(f"Skipping: {output_file_path}, results are available")
                continue
            
            # Recording processing time
            start = datetime.now()

            '''
            REASONER
            '''
            print("===================== REASONING WITH CHATGPT =====================")   
            print("===================== REASONING =====================")
            prediction, explanation = reasoner.run_reasoner(utterance_level_transcripts)
            '''
            SAVE RESULTS
            '''
            time_processing = datetime.now() - start
    
            result = {"prediction": prediction, 
                      "explanation": explanation, 
                      'processing time': int(time_processing.total_seconds())
                      }        
            
            with open(output_file_path, 'w') as fp:
                json.dump(result, fp, indent=4)