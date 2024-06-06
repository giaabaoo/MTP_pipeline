import openai
from tqdm import tqdm
import pandas as pd
import pdb
import json
import os

class ChatGPTTrainer:
    def __init__(self, config):
        self.config = config
        openai.api_key = config.api_key
        
    def submit_train_data(self, train_data):
        uploading_response = openai.File.create(file=open(train_data, "rb"), purpose='fine-tune')
        file_id = uploading_response.id
        return file_id, uploading_response
    
    def finetune(self, file_id):
        finetuning_response = openai.FineTuningJob.create(training_file=file_id[0], model=self.config.model_engine)

        return finetuning_response
    
    def check_finetune_status(self, finetuning_response):
        fine_tune_events = openai.FineTuningJob.list_events(id=finetuning_response['id'])
        
        return fine_tune_events
    
    def debug(self, finetuning_response):
        # res =  openai.FineTuningJob.list_events(id=finetuning_response['id'], limit=10)
        # openai.Model.delete("ft:gpt-3.5-turbo-0613:personal::83O5ov7L")
        # files = openai.File.list()

        # openai.FineTuningJob.cancel("ftjob-vkViJxM9vtl5kFgp3XZQ6qs6")

        # for file in files['data']:
        #     pdb.set_trace()
        #     openai.File.delete("file-AJB1BGuVJPDskJIGF27wsUWR")
            
        # openai.File.delete(file_id)
        pdb.set_trace()
        
