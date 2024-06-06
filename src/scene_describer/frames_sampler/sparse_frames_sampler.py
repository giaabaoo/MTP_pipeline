import pdb

class SparseFramesSampler:
    def __init__(self, config):
        self.config = config
        
    def run(self, video):
        step_size = video.duration / (self.config.num_frames - 1)
        time_list = [int(i * step_size) for i in range(self.config.num_frames)]
        
        frames_list = []
                
        for time in time_list:
            try:
                frame = video.get_frame(time)
            except:
                frame = video.get_frame(time-1)
                
            frames_list.append(frame)
        
        return frames_list
