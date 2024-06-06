import os
import os.path as osp
import pdb
from pathlib import Path

class UtteranceLevelTranscriptLoader:
    def __init__(self, config):
        self.config = config
        self.output_path = osp.join(config.output_dir, config.data.output_path)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def custom_sort_key(filename):
        # Remove the ".txt" extension before splitting
        filename_without_extension = os.path.splitext(filename)[0]
        
        # Assuming the filenames are in the format "conver_X"
        parts = filename_without_extension.split("_")
        if len(parts) == 2 and parts[0] == "conversation":
            return int(parts[1])  # Convert the numeric part to an integer
        return filename
        
    def get_items(self):
        items_dict = dict()
        
        count_conversation = 0
        for conversation_file in sorted(os.listdir(self.config.data.input_path), key=self.custom_sort_key):
           
            with open(osp.join(self.config.data.input_path, conversation_file)) as f:
                conversation = f.readlines()
            conversation_name = conversation_file.replace(".txt", "")
            
            output_file_path = os.path.join(self.output_path, conversation_name + ".json")
            
            if conversation_name not in items_dict:
                items_dict[conversation_name] = {}
                
            items_dict[conversation_name]['transcript'] = conversation
            items_dict[conversation_name]['output_file'] = output_file_path
            count_conversation += 1
            
            if count_conversation >= self.config.data.number_of_conversations:
                break
            
        return items_dict
            
        