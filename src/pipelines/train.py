from src.utils import setup_logger

from src.data_loader import get_data_loader
from src.preprocessor import get_preprocessor
from src.trainer import get_trainer

import os
import json
import pdb

class Train():
    def __init__(self, config):
        self.config = config
    
    def run(self):
        # Generate the output directory and save the full config file
        setup_logger(self.config)
        preprocessor = get_preprocessor(self.config)
        trainer = get_trainer(self.config)
        
        if os.path.exists(os.path.join(self.config.output_dir, "finetuning_responses.json")):
            with open(os.path.join(self.config.output_dir, "finetuning_responses.json"), "r") as f:
                finetuning_response = json.load(f)
                
            fine_tune_events = trainer.check_finetune_status(finetuning_response)
            with open(os.path.join(self.config.output_dir, "finetuning_status_responses.json"), "w") as f:
                    json.dump(fine_tune_events, f, indent=4)
                    
            trainer.debug(finetuning_response)
            
        else:
            print("===================== PREPROCESSING =====================")
            train_data = preprocessor.run()
            
            print("===================== TRAINING =====================")
            
            
            uploading_responses, finetuning_responses = {}, {}
            if os.path.exists(os.path.join(self.config.output_dir, "file_id.txt")):
                with open(os.path.join(self.config.output_dir, "file_id.txt"), "r") as f:
                    file_id = f.readlines()
            else:
                file_id, uploading_responses = trainer.submit_train_data(train_data)
                with open(os.path.join(self.config.output_dir, "file_id.txt"), "w") as f:
                    f.writelines(file_id)
                    
                with open(os.path.join(self.config.output_dir, "uploading_responses.json"), "w") as f:
                    json.dump(uploading_responses, f, indent=4)
            
            
            finetuning_responses = trainer.finetune(file_id)
            
            with open(os.path.join(self.config.output_dir, "finetuning_responses.json"), "w") as f:
                    json.dump(finetuning_responses, f, indent=4)
            
        
            
        
         
