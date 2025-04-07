import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms

class VideoFrameDataset(Dataset):
    def __init__(self, video_dir, sequence_length=5, transform=None):
        """
        Dataset for loading sequences of video frames
        
        Args:
            video_dir (str): Directory containing video files
            sequence_length (int): Number of consecutive frames to include in each sample
            transform (callable): transform to be applied on frames ( in train_video_autoencoder.py)
        """
        self.video_dir = video_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # get videos
        self.video_files = [f for f in os.listdir(video_dir) 
                           if f.endswith(('.mp4', '.avi', '.mov'))]
        
        # map of sample indices to video_idx, start_frame
        self.samples = []
        for video_idx, video_file in enumerate(self.video_files):
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # For each possible starting frame that allows a full sequence
            for start_frame in range(0, frame_count - sequence_length + 1):
                self.samples.append((video_idx, start_frame))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_idx, start_frame = self.samples[idx]
        video_path = os.path.join(self.video_dir, self.video_files[video_idx])
        
        # loading the sequence of frames
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
                
            # convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # apply transforms
            if self.transform:
                frame = self.transform(frame)
            else:
                # conversion to tensor and normalization
                frame = torch.from_numpy(frame.transpose((2, 0, 1))).float() / 255.0
                
            frames.append(frame)
        
        cap.release()
        
        # handle empty frames list
        if len(frames) == 0:
            # Create a dummy tensor with the right dimensions (128x128)
            return torch.zeros((self.sequence_length, 3, 128, 128))
        
        # handle incomplete sequences by duplicating the last frame
        if len(frames) < self.sequence_length:
            last_frame = frames[-1]
            while len(frames) < self.sequence_length:
                frames.append(last_frame.clone())
        
        # stack frames
        sequence = torch.stack(frames)
        
        # check shape of all the frames
        if sequence.shape[2] != 128 or sequence.shape[3] != 128:
            sequence = torch.nn.functional.interpolate(
                sequence, 
                size=(128, 128), 
                mode='bilinear', 
                align_corners=False
            )
        
        return sequence