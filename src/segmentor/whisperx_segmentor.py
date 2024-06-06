import os
import subprocess
from pathlib import Path
import json
import pdb
import logging
import natsort


class WhisperXSegmentor:
    def __init__(self, config):
        self.config = config

    def run(self, conversation_name, conversation_data):
        utterance_video_path, utterance_transcript_path = conversation_data["utterance_video_path"], conversation_data["utterance_transcript_path"]
        full_video_path = conversation_data["input_path"]
        
        # DIARIZE
        if not os.path.exists(utterance_transcript_path):
            # Create a folder for this transcript
            Path(utterance_transcript_path).mkdir(parents=True, exist_ok = True)

            # Run the whisperx command on the .mkv file and save output in the transcript folder
            cmd = ["whisperx", full_video_path, "--output_dir", utterance_transcript_path, "--output_format", "json"]
            subprocess.run(cmd, check=True)

        utterance_level_transcripts = os.path.join(utterance_transcript_path, conversation_name + ".json")
        
        ### SEGMENT
        utterance_level_videos = []

        if not os.path.exists(utterance_video_path):
            utterance_level_videos = self.diarize_video(full_video_path, utterance_video_path, utterance_level_transcripts)
        else:
            print("This conversation has already been segmented..")
            for utterance in natsort.natsorted(os.listdir(utterance_video_path)):
                utterance_level_videos.append(os.path.join(utterance_video_path, utterance))
                    
        return utterance_level_videos, utterance_level_transcripts

    def extract_frame(self, full_video_path, output_image_path, timestamp, output_filename):
        # Extract the frame at the specified timestamp as an image
        output_image_path = os.path.join(output_image_path, output_filename)
        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),   # Seek to the specified timestamp
            '-i', full_video_path,
            '-vframes', '1',         # Extract only one frame
            output_image_path
        ]
        subprocess.run(cmd, check=True)
        return output_image_path

    def diarize_video(self, full_video_path, output_image_path, utterance_level_transcripts):
        # Load the utterance-level timestamps from the JSON file
        with open(utterance_level_transcripts, 'r') as json_file:
            utterance_data = json.load(json_file)

        # Create the output directory if it doesn't exist
        os.makedirs(output_image_path, exist_ok=True)

        # Initialize a logger for error reporting
        logging.basicConfig(filename='diarize_video.log', level=logging.ERROR)

        # Initialize a list to store image paths
        image_paths = []

        # Iterate over each segment in the JSON data
        for i, segment in enumerate(utterance_data['segments']):
            start_time = segment['start']
            end_time = segment['end']

            try:
                # Calculate the middle time within the segment
                first_quarter_time = start_time + (end_time - start_time) / 4.0

                # Generate the output image filename (e.g., utterance_1.jpg, utterance_2.jpg, ...)
                output_filename = f'utterance_{i + 1}.jpg'

                # Extract and save the middle frame as a .jpg image
                image_path = self.extract_frame(full_video_path, output_image_path, first_quarter_time, output_filename)

                # Append the image path to the list
                image_paths.append(image_path)
            except Exception as e:
                # Handle any errors here if necessary
                continue

        # Return the list of image paths
        return image_paths
        

        
# if __name__ == "__main__":
#     config = {}
#     segmentor = WhisperXSegmentor(config=config)
#     segmentor.run("S01E01_000.mp4", "", "S01E01_000")