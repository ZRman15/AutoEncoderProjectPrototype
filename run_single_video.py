import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from video_dataset import VideoFrameDataset

def main():
    # Path to your specific video file
    video_file = '/Users/zohaib/Downloads/IMG_1271.mp4'
    
    # Create a temporary directory to hold this single video
    temp_dir = '/Users/zohaib/Desktop/University/Software Project/Prototype/temp_video_dir'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy or symlink the video to the temp directory
    import shutil
    video_filename = os.path.basename(video_file)
    dest_path = os.path.join(temp_dir, video_filename)
    
    if not os.path.exists(dest_path):
        shutil.copy2(video_file, dest_path)
        print(f"Copied video to {dest_path}")
    
    # Create the dataset using the temp directory
    sequence_length = 5
    dataset = VideoFrameDataset(temp_dir, sequence_length=sequence_length)
    
    # Print dataset information
    print(f"Dataset created successfully for video: {video_filename}")
    print(f"Total number of sequences: {len(dataset)}")
    
    # Display some frames
    if len(dataset) > 0:
        sequence = dataset[0]
        print(f"Sequence shape: {sequence.shape}")
        
        # Display the first sequence
        fig, axes = plt.subplots(1, sequence_length, figsize=(15, 3))
        for i in range(sequence_length):
            frame = sequence[i].permute(1, 2, 0).numpy()
            axes[i].imshow(frame)
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('/Users/zohaib/Desktop/University/Software Project/Prototype/single_video_preview.png')
        plt.show()
    else:
        print("No sequences found in the video.")

if __name__ == "__main__":
    main()