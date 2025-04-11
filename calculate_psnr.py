import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from video_autoencoder import VideoAutoencoder
import matplotlib.pyplot as plt

def calculate_video_psnr(original_video_path, compressed_video_path, output_dir=None):
    """
    Calculate the PSNR between original and compressed video frames.
    
    Args:
        original_video_path (str): Path to the original video file
        compressed_video_path (str): Path to the compressed/reconstructed video file
        output_dir (str, optional): Directory to save comparison images
        
    Returns:
        float: Average PSNR across all frames
        list: PSNR values for each frame
    """
    # check if videos exist
    if not os.path.exists(original_video_path):
        raise FileNotFoundError(f"Original video not found: {original_video_path}")
    if not os.path.exists(compressed_video_path):
        raise FileNotFoundError(f"Compressed video not found: {compressed_video_path}")
    
    # open video files
    original_cap = cv2.VideoCapture(original_video_path)
    compressed_cap = cv2.VideoCapture(compressed_video_path)
    
    # check if videos opened successfully
    if not original_cap.isOpened() or not compressed_cap.isOpened():
        raise RuntimeError("Error opening video files")
    
    # get video properties
    original_frame_count = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    compressed_frame_count = int(compressed_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # use the minimum frame count to avoid index errors
    frame_count = min(original_frame_count, compressed_frame_count)
    
    # create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # calculate psnr for each frame
    psnr_values = []
    
    for frame_idx in range(frame_count):
        # read frames
        ret1, original_frame = original_cap.read()
        ret2, compressed_frame = compressed_cap.read()
        
        # check if frames were read successfully
        if not ret1 or not ret2:
            break
        
        # calculate psnr
        psnr = cv2.PSNR(original_frame, compressed_frame)
        psnr_values.append(psnr)
        
        # save comparison images if output directory is specified
        if output_dir and frame_idx % 60 == 0:  # save every 60th frame to avoid too many images
            comparison = np.hstack((original_frame, compressed_frame))
            cv2.putText(comparison, f"PSNR: {psnr:.2f} dB", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, f"comparison_frame_{frame_idx}.png"), comparison)
    
    # release video captures
    original_cap.release()
    compressed_cap.release()
    
    # calculate average psnr
    avg_psnr = np.mean(psnr_values) if psnr_values else 0
    
    # plot psnr over frames if output directory is specified
    if output_dir and psnr_values:
        plt.figure(figsize=(10, 5))
        plt.plot(psnr_values)
        plt.axhline(y=float(avg_psnr), color='r', linestyle='--', label=f'Average: {avg_psnr:.2f} dB')
        plt.title('PSNR per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "psnr_graph.png"))
        plt.close()
    
    return avg_psnr, psnr_values

def evaluate_compression_quality(original_video_path, model_path, output_dir=None, sequence_length=5, latent_dim=128):
    """
    Compress a video using the trained model and evaluate the PSNR.
    
    Args:
        original_video_path (str): Path to the original video
        model_path (str): Path to the trained model
        output_dir (str, optional): Directory to save results
        sequence_length (int): Number of frames in each sequence
        latent_dim (int): Dimension of the latent space
        
    Returns:
        float: Average PSNR
    """
    # create output directory
    if output_dir is None:
        output_dir = "/Users/zohaib/Desktop/University/Software Project/Prototype/psnr_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # paths for compressed video
    video_name = os.path.basename(original_video_path)
    compressed_video_path = os.path.join(output_dir, f"compressed_{video_name}")
    
    # import the compress_video function
    from compress_video import compress_video
    
    # compress the video
    print(f"Compressing video: {original_video_path}")
    compress_video(original_video_path, compressed_video_path, model_path, 
                  sequence_length=sequence_length, latent_dim=latent_dim)
    
    # calculate psnr
    print(f"Calculating PSNR between original and compressed videos...")
    avg_psnr, psnr_values = calculate_video_psnr(
        original_video_path, 
        compressed_video_path,
        os.path.join(output_dir, "frame_comparisons")
    )
    
    # print results
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Min PSNR: {min(psnr_values):.2f} dB")
    print(f"Max PSNR: {max(psnr_values):.2f} dB")
    
    # save results to a text file
    with open(os.path.join(output_dir, "psnr_results.txt"), "w") as f:
        f.write(f"Video: {original_video_path}\n")
        f.write(f"Compressed video: {compressed_video_path}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Min PSNR: {min(psnr_values):.2f} dB\n")
        f.write(f"Max PSNR: {max(psnr_values):.2f} dB\n")
    
    return avg_psnr

if __name__ == "__main__":
    # example usage
    original_video = "/Users/zohaib/Desktop/University/Software Project/Prototype/videos/Test2.mp4"
    model_path = "/Users/zohaib/Desktop/University/Software Project/Prototype/video_autoencoder_final.pth"
    
    # evaluate compression quality
    avg_psnr = evaluate_compression_quality(
        original_video,
        model_path,
        output_dir="/Users/zohaib/Desktop/University/Software Project/Prototype/psnr_evaluation"
    )