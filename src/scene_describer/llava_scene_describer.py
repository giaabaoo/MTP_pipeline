import subprocess
from src.utils import Config
import cv2
import os
from pathlib import Path
import torch

from src.scene_describer.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from src.scene_describer.llava.conversation import conv_templates, SeparatorStyle
from src.scene_describer.llava.model.custom_builder import load_pretrained_model
from src.scene_describer.llava.custom_utils import disable_torch_init
from src.scene_describer.llava.custom_mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import json
import pdb
from tqdm import tqdm

class LLAVASceneDescriber:
    def __init__(self, config: Config):
        self.config = config
        self.prompt = config.prompt
        Path("./tmp/").mkdir(exist_ok=True, parents=True)
        
    def load_image(self, image_file):
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    def llava_main(self, tokenizer, model, image_processor, roles, conv):
        image = self.load_image(self.config.image_file)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        fixed_prompt = self.config.prompt
        inp = fixed_prompt

        try:
            # Use the fixed prompt instead of getting input from the user
            print(f"{roles[0]}:", inp)
            
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            # conv.messages[-1][-1] = outputs
            
            if self.config.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
            return outputs

        except Exception as e:
            print("An error occurred:", e)
            return
            
    def llava_setup(self):
        # Model
        disable_torch_init()

        model_name = get_model_name_from_path(self.config.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(self.config.model_path, self.config.model_base, model_name, self.config.load_8bit, self.config.load_4bit)

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.config.conv_mode is not None and conv_mode != self.config.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, self.config.conv_mode, self.config.conv_mode))
        else:
            self.config.conv_mode = conv_mode

        conv = conv_templates[self.config.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        
        ###
        # first message
        inp = self.config.prompt
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        ###
        
        return tokenizer, model, image_processor, roles, conv
            
    def run(self, video, frames_sampler):
        frames_list = frames_sampler.run(video)
        
        video_descriptions = []
        tokenizer, model, image_processor, roles, conv = self.llava_setup()
        
        for frame in frames_list:
            # Save the temp image
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.config.image_file, frame_bgr)
            
            message = self.llava_main(tokenizer, model, image_processor, roles, conv)
            # Delete the temp image
            os.remove(self.config.image_file)
        
            video_descriptions.append(message)
        
        return video_descriptions
    
    def main_run(self, conversation_data, utterance_level_videos):
        # TODO: Update a proper frame samplers
        output_video_description_path = conversation_data['output_video_description_path']
        video_descriptions = {}
        if os.path.exists(output_video_description_path):
            with open(output_video_description_path, 'r') as fp:
                video_descriptions = json.load(fp)
        else:            
            tokenizer, model, image_processor, roles, conv = self.llava_setup()
        
            for i, utterance in enumerate(utterance_level_videos):
                self.config.image_file = utterance
                message = self.llava_main(tokenizer, model, image_processor, roles, conv)

                utterance_name = f"utterance_{i+1}"
                video_descriptions[utterance_name] = message

            with open(output_video_description_path, 'w') as fp:
                json.dump(video_descriptions, fp, indent=4)
        
        return video_descriptions
    
    def execute_cli(self, image_file): # deprecated
        python_script = f"python -m llava.serve.inference --model-path liuhaotian/LLaVA-Lightning-MPT-7B-preview --image-file {image_file} --fixed-prompt \"{self.prompt}\""
        result = subprocess.run(python_script, shell=True, check=True, capture_output=True, text=True)
        generated_message = result.stdout.strip()
        return generated_message

if __name__ == "__main__":
    config = {}  # Provide your configuration here
    scene_describer = LLAVASceneDescriber(config)
    fixed_prompt = "Fill in the blank. Describe each person from left to right. Example: \
                    Person 1, facial expressions:_, postures:_, potential emotions:_, actions:_ \n \
                    Person 2, facial expressions:_, postures:_, potential emotions:_, actions:_ "
    # generated_message = scene_describer.execute_cli("test.jpg", fixed_prompt)
    # print("Generated Message:", generated_message)
