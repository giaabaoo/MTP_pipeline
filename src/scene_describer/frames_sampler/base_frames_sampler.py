class BaseFramesSampler:
    def __init__(self, config):
        self.config = config
        
    def run(self, video):
        frames_list = []
            
        for time in range(0, int(video.duration), 1):
            frame = video.get_frame(time)
            frames_list.append(frame)
        
        return frames_list
