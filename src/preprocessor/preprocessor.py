import pandas as pd
import os
import json
import re
import pdb
from sklearn.model_selection import train_test_split

def time_to_seconds(time_str):
    # time_str = time_str.strftime('%H:%M')
    try:
        numeric_parts = re.findall(r'\d+', time_str)[:2]
    except TypeError:
        time_str = time_str.strftime('%H:%M')
        numeric_parts = re.findall(r'\d+', time_str)[:2]
    except Exception as e:
        print(e)
     
    # Convert the extracted numeric parts to integers
    minutes, seconds = map(int, numeric_parts)
    return minutes * 60 + seconds

class Preprocessor:
    def __init__(self, config):
        self.config = config
        
    def run(self):
        if not os.path.exists(self.config.annotations_path):
            self.match_timestamp_with_utterance()
            
        annotation_df = pd.read_excel(self.config.annotations_path)
        # Get unique conversation indices
        unique_indices = annotation_df['conversation_index'].unique()

        # Split unique indices into train and test
        train_indices, test_indices = train_test_split(unique_indices, test_size=0.2, random_state=42)

        # Filter rows for train and test sets
        train_df = annotation_df[annotation_df['conversation_index'].isin(train_indices)]
        test_df = annotation_df[~annotation_df['conversation_index'].isin(train_indices)]
        
        if self.config.no_training_set:
            if not os.path.exists(self.config.test_data_path):
                self.prepare_testing_dataset(unique_indices, annotation_df)   
        else:
            if not os.path.exists(self.config.train_data_path):
                self.prepare_training_dataset(train_indices, train_df)

            if not os.path.exists(self.config.test_data_path):
                self.prepare_testing_dataset(test_indices, test_df)
            
       
    
        return self.config.train_data_path
        
    def match_timestamp_with_utterance(self):
        annotation_df = pd.read_excel(self.config.raw_annotations_path)
        annotation_df = annotation_df.dropna(subset=['TP_location'], how='all')

        # Initialize a list to store the utterances
        utterances = []

        # Iterate through the rows of the Excel DataFrame
        for index, row in annotation_df.iterrows():
            if row['TP_location'] == "-":
                utterance = ""
            else:
                TP_location = time_to_seconds(row['TP_location'])
                conversation_name = "conversation_" + str(row['conversation_index'])
                
                with open(os.path.join(self.config.transcripts_path, conversation_name, conversation_name + ".json"), "r") as f:
                    transcript_data = json.load(f)

                # Find the corresponding transcript segment based on the nearest increasing time direction
                matched_segment = None
                for segment in transcript_data['segments']:
                    if segment['start'] >= TP_location:
                        matched_segment = segment
                        break

                # Extract the text from the matched segment or set to empty string if not found
                utterance = matched_segment['text'] if matched_segment else ""

            # Append the utterance to the list
            utterances.append(utterance)

        # Add the 'utterance' column to the Excel DataFrame
        annotation_df['utterance'] = utterances
        
        # Save the updated DataFrame to a new Excel file
        annotation_df.to_excel(self.config.annotations_path, index=False)
        
    def prepare_training_dataset(self, train_indices, train_df):        
        if self.config.train_positive_samples_only:
            annotation_df = annotation_df[annotation_df['TP_location'] != '-']
            # Reset the index if needed
            annotation_df.reset_index(drop=True, inplace=True)            
        
        train_conversations = []

        for conversation_index in train_indices:
            conversation_name = "conversation_" + str(conversation_index)
                
            with open(os.path.join(self.config.summaries_path, conversation_name + ".json"), "r") as f:
                summary_data = json.load(f)
                
            system_content = "You are a trained chatbot that can find turning points in conversations. A turning point in a conversation is an identifiable event that leads to an unexpected and significant transformation in the subjective personal states (including decisions, behaviors, perspectives, and feelings) of at least one speaker during the given conversation. "
            summary = summary_data
            
            assistant_content = self.get_answer_event_utt(train_df, conversation_index)  # Implement get_answer function
            messages = []
            
            sample = {'messages' : messages}
            system = {
                "role": "system",
                "content": system_content
            }
            sample['messages'].append(system)
            
            user_message = {
                "role": "user",
                "content": f"Help me find the events that cause the turning points in this conversation {summary}."  
            }
            sample['messages'].append(user_message) 
            
            assistant_message = {
                "role": "assistant",
                "content": assistant_content
            }
            sample['messages'].append(assistant_message)  
            
            train_conversations.append(sample)

        

        with open(self.config.train_data_path, "w") as train_jsonl_file:
            for conversation in train_conversations:
                train_jsonl_file.write(json.dumps(conversation) + "\n")

    def prepare_testing_dataset(self, test_indices, test_df):
        test_conversations = []
        
        test_indices = list(range(1, max(test_indices) + 1))
        
        for conversation_index in test_indices:
            conversation_name = "conversation_" + str(conversation_index)

            
            with open(os.path.join(self.config.summaries_path, conversation_name + ".json"), "r") as f:
                summary_data = json.load(f)
                
            if conversation_index == 204:
                break
                
            summary = summary_data
            
            assistant_content, timestamp = self.get_answer_event_utt(test_df, conversation_index) 
                        
            sample = {'conversation_index': str(conversation_index), 'summary' : summary, 'label' : assistant_content, 'label_timestamp': timestamp}

           
            test_conversations.append(sample)

        # try:
        with open(self.config.test_data_path, "w") as test_jsonl_file:
            for conversation in test_conversations:
                test_jsonl_file.write(json.dumps(conversation) + "\n")
        # except:
        #     import pdb
        #     pdb.set_trace()
                
    def get_answer(self, annotation_df, conversation_index):
        turning_points = annotation_df[annotation_df['conversation_index'] == conversation_index]
        
        answer = ""
        
        for i, (_, turning_point) in enumerate(turning_points.iterrows()):  
            if turning_point['TP_location'] == "-":
                answer += f"There is no turning point in this conversation. Reason: {turning_point['explanation']}"
            else:
                feelings_change = self.get_feelings_change(turning_point)
                others_change = self.get_others_change(turning_point)
                
                if feelings_change == "":
                    answer += f"Turning point {i}: At utterance \"{turning_point['utterance']}\", with {others_change}."
                elif others_change == "":
                    answer += f"Turning point {i}: At utterance \"{turning_point['utterance']}\", with {feelings_change}."
                elif feelings_change != "" and others_change != "":
                    answer += f"Turning point {i}: At utterance \"{turning_point['utterance']}\", with {others_change}, and {feelings_change}."
        
        return answer
    
    def get_answer_event_utt(self, annotation_df, conversation_index):
        turning_points = annotation_df[annotation_df['conversation_index'] == conversation_index]
        
        answer = ""
        timestamp = []
        
        for i, (_, turning_point) in enumerate(turning_points.iterrows()):  
            if turning_point['TP_location'] == "-":
                try:
                    answer += f"There is no turning point in this conversation. Reason: {turning_point['explanation']}"
                except:
                    answer += f"There is no turning point in this conversation. Reason: {turning_point['comments']}"
                timestamp = -1
            else:
                feelings_change = self.get_feelings_change(turning_point)
                others_change = self.get_others_change(turning_point)
                
                if feelings_change == "":
                    answer += f"Turning point {i}: When {turning_point['TP_cause']}, with a response of \"{turning_point['utterance']}\". It leads to {others_change}. "
                elif others_change == "":
                    answer += f"Turning point {i}: When {turning_point['TP_cause']}, with a response of \"{turning_point['utterance']}\". It leads to {feelings_change}. "
                elif feelings_change != "" and others_change != "":
                    answer += f"Turning point {i}: When {turning_point['TP_cause']}, with a response of \"{turning_point['utterance']}\". It leads to {others_change}, and {feelings_change}."
                timestamp.append(turning_point['TP_location'])
        return answer, timestamp
    
    def get_answer_event(self, annotation_df, conversation_index):
        turning_points = annotation_df[annotation_df['conversation_index'] == conversation_index]
        
        answer = ""
        
        for i, (_, turning_point) in enumerate(turning_points.iterrows()):  
            if turning_point['TP_location'] == "-":
                answer += f"There is no turning point in this conversation. Reason: {turning_point['explanation']}"
            else:
                feelings_change = self.get_feelings_change(turning_point)
                others_change = self.get_others_change(turning_point)
                
                if feelings_change == "":
                    answer += f"Turning point {i}: When {turning_point['TP_cause']}, leading to {others_change}. \n"
                elif others_change == "":
                    answer += f"Turning point {i}: When {turning_point['TP_cause']}, leading to {feelings_change}. \n"
                elif feelings_change != "" and others_change != "":
                    answer += f"Turning point {i}: When {turning_point['TP_cause']}, leading to {others_change}, and {feelings_change}. \n"
                timestamp = turning_point['TP_location']
                
        return answer, timestamp
    
    def extract_text_and_time(self, sentence):
        # Define a regular expression pattern to match the emotion in parentheses and the time format "1:40"
        pattern = r'([^()]+)\s*\((\d+:\d+)\)'

        # Use regex to find the pattern in the sentence
        match = re.search(pattern, sentence)

        if match:
            # Extract the matched emotion and time groups
            text = match.group(1)
            time = match.group(2)
            return text  # Return both text and time as a tuple
        else:
            # If no pattern is found, return None or handle as needed
            return None, None  # Return None for both text and time
    
    def get_feelings_change(self, turning_point):
        pre_point_feeling_text_list, post_point_feeling_text_list = [], []
        pre_point_feeling = turning_point['pre_point_feeling']
        post_point_feeling = turning_point['post_point_feeling']
        
        try:
            pre_point_feeling_list = pre_point_feeling.split(",")
            post_point_feeling_list = post_point_feeling.split(",")
            for text in pre_point_feeling_list:
                pre_emotion = self.extract_text_and_time(text)
                
                pre_point_feeling_text_list.append(pre_emotion)
            
            for text in post_point_feeling_list:
                post_emotion = self.extract_text_and_time(text)
                post_point_feeling_text_list.append(post_emotion)
        except:
            pre_point_feeling_text_list = "nan"
                
        if str(pre_point_feeling_text_list) != "nan":
            feelings_change = f"feelings change from ({pre_point_feeling_text_list}) to ({post_point_feeling_text_list}), which can be considered significant"
            
            return feelings_change
        else:
            return ""
        
    def get_others_change(self, turning_point):  # decisions, behaviors, perspectives
        pre_point_others_text_list, post_point_others_text_list = [], []
        
        pre_point_decision_behavior_perspective = turning_point['pre_point_decision_behavior_perspective']
        post_point_decision_behavior_perspective = turning_point['post_point_decision_behavior_perspective']
        
        try:
            pre_point_others_list = pre_point_decision_behavior_perspective.split(",")
            post_point_others_list = post_point_decision_behavior_perspective.split(",")
            for text in pre_point_others_list:
                pre_others = self.extract_text_and_time(text)
                pre_point_others_text_list.append(pre_others)
            
            for text in post_point_others_list:
                post_others = self.extract_text_and_time(text)
                post_point_others_text_list.append(post_others)
        except:
            pre_point_others_text_list = "nan"

        if str(pre_point_others_text_list) != "nan":
            others_change = f"behaviors/decisions/perspectives change from ({pre_point_others_text_list}) to ({post_point_others_text_list})"
            return others_change
        else:
            return ""
    

# if __name__ == "__main__":
    # transcripts_path =  "/home/dhgbao/Research_Monash/code/others/my_code/methods_experiments/multimodal/conversational_turning_point_detection/data/CCTP/utterances/transcripts"
    # raw_annotations_path = "/home/dhgbao/Research_Monash/code/others/my_code/methods_experiments/multimodal/conversational_turning_point_detection/data/CCTP/annotations/raw_annotations.xlsx"
    # annotations_path = "/home/dhgbao/Research_Monash/code/others/my_code/methods_experiments/multimodal/conversational_turning_point_detection/data/CCTP/annotations/annotations.xlsx"
    # preprocessor = Preprocessor(config={})
    # preprocessor.preprocess(transcripts_path, raw_annotations_path, annotations_path)