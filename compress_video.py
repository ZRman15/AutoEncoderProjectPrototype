import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from video_autoencoder import VideoAutoencoder

def compress_video(input_video_path, output_video_path, model_path, sequence_length=5, latent_dim=128):
    """
    Compress a video using the trained autoencoder and save the reconstructed video
    """
    # Check if input video exists
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at {input_video_path}")
        return
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained model
    model = VideoAutoencoder(sequence_length=sequence_length, in_channels=3, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Create transform for preprocessing frames
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Create transform for postprocessing frames
    postprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
    ])
    
    # Process the video in sequences
    frames_buffer = []
    processed_frames = 0
    total_output_frames = 0
    
    print(f"Processing video with {frame_count} frames...")
    
    # Read all frames first
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    
    print(f"Read {len(all_frames)} frames from video")
    
    # Process frames in sliding window fashion
    for i, frame in enumerate(all_frames):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess the frame
        frame_tensor = preprocess(frame_rgb)
        
        # Add to buffer
        frames_buffer.append(frame_tensor)
        
        # Process when buffer reaches sequence_length
        if len(frames_buffer) == sequence_length:
            # Create a batch with a single sequence
            sequence = torch.stack(frames_buffer).unsqueeze(0).to(device)
            
            # Process through the model
            with torch.no_grad():
                reconstructed, _ = model(sequence)
            
            # Process the middle frame (to reduce boundary artifacts)
            middle_idx = sequence_length // 2
            reconstructed_frame = reconstructed[0, middle_idx].cpu()
            
            # Convert tensor to PIL image and resize back to original dimensions
            reconstructed_pil = postprocess(reconstructed_frame)
            
            # Convert PIL image to numpy array and then to BGR for OpenCV
            reconstructed_np = np.array(reconstructed_pil)
            reconstructed_bgr = cv2.cvtColor(reconstructed_np, cv2.COLOR_RGB2BGR)
            
            # Write to output video
            out.write(reconstructed_bgr)
            total_output_frames += 1
            
            # Remove the first frame from buffer
            frames_buffer.pop(0)
            
            # Update progress
            processed_frames += 1
            if processed_frames % 10 == 0:
                print(f"Processed {processed_frames}/{len(all_frames)} frames ({processed_frames/len(all_frames)*100:.1f}%)")
    
    # Process any remaining frames in the buffer (for the last few frames)
    remaining_frames = len(frames_buffer)
    if remaining_frames > 0:
        print(f"Processing remaining {remaining_frames} frames...")
        
        # Pad the buffer to ensure it has exactly sequence_length frames
        while len(frames_buffer) < sequence_length:
            # Duplicate the last frame to pad the sequence
            frames_buffer.append(frames_buffer[-1].clone())
        
        # Now process each remaining frame one by one
        for i in range(remaining_frames):
            # Create a batch with the current buffer
            sequence = torch.stack(frames_buffer).unsqueeze(0).to(device)
            
            # Process through the model
            with torch.no_grad():
                reconstructed, _ = model(sequence)
            
            # Process the first frame (which is the next one we need)
            reconstructed_frame = reconstructed[0, 0].cpu()
            
            # Convert tensor to PIL image and resize back to original dimensions
            reconstructed_pil = postprocess(reconstructed_frame)
            
            # Convert PIL image to numpy array and then to BGR for OpenCV
            reconstructed_np = np.array(reconstructed_pil)
            reconstructed_bgr = cv2.cvtColor(reconstructed_np, cv2.COLOR_RGB2BGR)
            
            # Write to output video
            out.write(reconstructed_bgr)
            total_output_frames += 1
            
            # Remove the first frame and add a duplicate of the last frame
            frames_buffer.pop(0)
            frames_buffer.append(frames_buffer[-1].clone())
            
            # Update progress
            processed_frames += 1
            print(f"Processed remaining frame {i+1}/{remaining_frames}")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video compression complete. Output saved to {output_video_path}")
    print(f"Total input frames: {len(all_frames)}, Total output frames: {total_output_frames}")
    
    # Calculate file sizes for comparison
    original_size = os.path.getsize(input_video_path)
    compressed_size = os.path.getsize(output_video_path)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    print(f"Original video size: {original_size/1024/1024:.2f} MB")
    print(f"Compressed video size: {compressed_size/1024/1024:.2f} MB")
    print(f"File size compression ratio: {compression_ratio:.2f}x")

if __name__ == "__main__":
    input_video = "/Users/zohaib/Desktop/University/Software Project/Prototype/videos128/vid1.mp4"
    output_video = "/Users/zohaib/Desktop/University/Software Project/Prototype/compressed_video.mp4"
    model_path = "/Users/zohaib/Desktop/University/Software Project/Prototype/video_autoencoder_final.pth"
    
    compress_video(input_video, output_video, model_path)