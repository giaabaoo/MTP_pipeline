import openai
import pandas as pd
import json
import argparse
import os
import pdb
from tqdm import tqdm
import re

def extract_string_results(string):
    regex_pattern = r'^(.*?) \| (.*?): (.*?) \[(.*?), (\d+)\]$'

    matches = re.findall(regex_pattern, string, flags=re.MULTILINE)
    
    if len(matches) == 0:
        segment_id = "None"
        speaker = "None"
        sentence_text = "None"
        emotion = "none"
        CP_label = 0

    for match in matches:
        segment_id = match[0]
        speaker = match[1]
        sentence_text = match[2]
        emotion = match[3]
        CP_label = int(match[4])
        
    return segment_id, speaker, sentence_text, emotion, CP_label
    
def process_dialogue(dialogue):
    d = []
    lines = dialogue.split('\n')

    for idx, line in enumerate(lines):
        segment_id, speaker, sentence_text, emotion, CP_label = extract_string_results(line)

        d_tmp = {}
        
        d_tmp['segment_id'] = segment_id
        d_tmp['speaker'] = speaker
        d_tmp['sentence_text'] = sentence_text
        d_tmp['emotion'] = emotion
        d_tmp['CP_label'] = CP_label
        d.append(d_tmp)
    
    return d

def process_explanation(explanation):
    e = {}
    # Extract the summary and the changepoints
    summary = re.search(r'SUMMARY: (.*)\n', explanation).group(1)
    changepoints = re.findall(r'CHANGEPOINT \d+ \((.*?)\): (.*?)\n', explanation)

    # Save the changepoints to a dictionary
    changepoints_dict = {}
    for i, cp in enumerate(changepoints, start=1):
        segment_id, comment = cp
        changepoints_dict[i] = {'segment_id': segment_id, 'comment': comment}
    
    e["summary"] = summary
    e["changepoints"] = changepoints_dict
    
    return e

language = ['English']
emotions = ["fear", "anger", "joy", "sadness", "disgust","surprise", "trust", "anticipation","neutral"]

def main(args):
    openai.api_key = args.api_key
    
    with open(args.captions, 'r') as fp:
        captions = json.load(fp)
    
    if os.path.isfile(args.syn_dials):
        with open(args.syn_dials, 'r') as fp:
            syn_dials = json.load(fp)
    else:
        syn_dials = {}
        
    ctr = 0

    for k, v in tqdm(captions.items()):
        k = k.split('_')[0]
        if k not in syn_dials.keys():
            for l in language:            
                ##dialogue
                definition_text = f"A changepoint in a conversation is when there's \
                a shift in the topic, speaker, tone, or energy, which can lead to a change \
                in social norms or emotions. This shift can temporarily or permanently \
                affect the conversation's outcome, relationship, goals, emotions, and flow, \
                either positively or negatively in a significant way."
                
                first_prompt = f"With this definition \"{definition_text}\" \n \
                Generate a dialogue with up to 20 turns of text messages between two people \
                (family, friends, acquaintances) about a topic (on a discussion forum or SMS app) \
                with at most 3 changepoints (label=1) (annotate only the first point of change). \
                For each line, annotate STRICTLY at the end the emotion of the line with the emotions \
                from this list {emotions} (use emotions from this list only) \
                and label 1 for CP and label 0 for non-CP. It should \
                have the document name {k} and the increasing id the sentence with the format xxxx \
                and the speaker id (a random number) in that conversation. \
                For example: \n \
                M010009BC_0005 | 137903: I've just broken up with my gf. [sadness, 1]. \n \
                M010009BC_0006 | 256348: Wow really?? I'm so sorry to hear that... [surprise, 0]. "
                
                dic = {'role': 'user', 'content': first_prompt}
                completion = openai.ChatCompletion.create(model=args.model_engine, messages = [dic])
                dialogue = completion.choices[0].message['content']
                                
                second_prompt = f"Follow this definition: \n {definition_text}. \
                Give me the summary of this conversation. And give explanation (evidence) of why \
                you think this each line in this dialogue contains a changepoint \
                (where there is the number 1 in the square brackets.): \n {dialogue.strip()} \
                The format should be like this: \n \
                SUMMARY: The conversation is about a man and a girl talking about their school projects. \n \
                CHANGEPOINT 1 (M010009BC_0005): There is a shift in the tone of the conversations. The girl is upset \
                because her mid-term grade is bad, it affects the overall mood significantly. \
                The inital tone is serious and professional. \n \
                CHANGEPOINT 2 (M010009BC_0010): There is a shift in the tone and topic of the conversations. \
                The man is cracking a very funny joke that clear the heaviness of the conversation. \
                This is a significant changepoint because most of the conversation is serious before this point."
                dic = {'role': 'user', 'content': second_prompt}
                completion = openai.ChatCompletion.create(model=args.model_engine, messages = [dic])
                explanation = completion.choices[0].message['content']
                                
                processed_dialogue = process_dialogue(dialogue)
                processed_explanation = process_explanation(explanation)
                
                syn_dials[k + '_' + str(ctr)] = {}
                syn_dials[k + '_' + str(ctr)]['language'] = l
                syn_dials[k + '_' + str(ctr)]['dialogue'] = dialogue
                syn_dials[k + '_' + str(ctr)]['explanation'] = explanation
                syn_dials[k + '_' + str(ctr)]['processed_dialogue'] = processed_dialogue
                syn_dials[k + '_' + str(ctr)]['processed_explanation'] = processed_explanation
        
                with open('synCP_dials.json', 'w') as fp: json.dump(syn_dials, fp)
                with open('last_processed.txt', 'a+') as fp: fp.write(k + '\n')
                ctr += 1
                if ctr == 5:
                    print('{} samples generated ...'.format(ctr))
                    exit(0)

