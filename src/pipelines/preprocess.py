from src.utils import setup_logger

from src.preprocessor import get_preprocessor

import os
import json
import pdb

class Preprocess():
    def __init__(self, config):
        self.config = config
    
    def run(self):
        # Generate the output directory and save the full config file
        setup_logger(self.config)
        preprocessor = get_preprocessor(self.config)
        
        preprocessor.run()
            
           
        
            
        
         
