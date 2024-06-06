from src.utils import setup_logger

from src.evaluator import get_evaluator
from datetime import datetime

import os
import json
import pdb

class Evaluate():
    def __init__(self, config):
        self.config = config
    
    def run(self):
        # Generate the output directory and save the full config file
        setup_logger(self.config)
        evaluator = get_evaluator(self.config)

        print("===================== EVALUATING =====================")
        if not os.path.exists(os.path.join(self.config.output_dir, "evaluation_result.json")):
            start = datetime.now()
            results = evaluator.run()
            time_processing = datetime.now() - start
            results['processing time'] = int(time_processing.total_seconds())     
            
            with open(os.path.join(self.config.output_dir, "evaluation_result.json"), "w") as f:
                json.dump(results, f, indent=4)
        else: 
            result_path = os.path.join(self.config.output_dir, "evaluation_result.json")
            print(f"The evaluation result is saved in {result_path}")
           
        
            
        
         
