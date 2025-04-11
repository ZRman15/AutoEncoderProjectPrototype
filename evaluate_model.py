import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from video_dataset import VideoFrameDataset
from video_autoencoder import VideoAutoencoder

def evaluate_model():
    # Parameters
    video_dir = "/Users/zohaib/Desktop/University/Software Project/Prototype/test_videos"
    model_path = "/Users/zohaib/Desktop/University/Software Project/Prototype/video_autoencoder_final.pth"
    sequence_length = 5
    batch_size = 1  # Process one sequence at a time for visualization
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    dataset = VideoFrameDataset(video_dir, sequence_length=sequence_length, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load the trained model
    model = VideoAutoencoder(sequence_length=sequence_length, in_channels=3, latent_dim=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Create output directory for results
    results_dir = "/Users/zohaib/Desktop/University/Software Project/Prototype/evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Process test data
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if batch_idx >= 5:  # Only process a few samples
                break
                
            # Move data to device
            data = data.to(device)
            
            # Forward pass
            reconstructed, latent = model(data)
            
            # Create a figure to display original and reconstructed frames
            fig, axes = plt.subplots(2, sequence_length, figsize=(15, 6))
            
            # Get the first sequence in the batch
            original_seq = data[0].cpu()
            reconstructed_seq = reconstructed[0].cpu()
            
            # Display each frame
            for i in range(sequence_length):
                # Original frame
                orig_frame = original_seq[i].permute(1, 2, 0).numpy()
                axes[0, i].imshow(orig_frame)
                axes[0, i].set_title(f"Original {i+1}")
                axes[0, i].axis('off')
                
                # Reconstructed frame
                recon_frame = reconstructed_seq[i].permute(1, 2, 0).numpy()
                
                # Print min/max values to debug
                print(f"Frame {i} - Min: {recon_frame.min():.4f}, Max: {recon_frame.max():.4f}, Mean: {recon_frame.mean():.4f}")
                
                # Ensure values are in valid range
                recon_frame = np.clip(recon_frame, 0, 1)
                
                axes[1, i].imshow(recon_frame)
                axes[1, i].set_title(f"Reconstructed {i+1}")
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/sequence_{batch_idx}.png")
            plt.close()
            
            # Also save individual frames for closer inspection
            for i in range(sequence_length):
                orig_frame = original_seq[i].permute(1, 2, 0).numpy()
                recon_frame = reconstructed_seq[i].permute(1, 2, 0).numpy()
                recon_frame = np.clip(recon_frame, 0, 1)
                
                # Create a side-by-side comparison
                plt.figure(figsize=(10, 5))
                
                plt.subplot(1, 2, 1)
                plt.imshow(orig_frame)
                plt.title("Original")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(recon_frame)
                plt.title("Reconstructed")
                plt.axis('off')
                
                plt.savefig(f"{results_dir}/frame_{batch_idx}_{i}.png")
                plt.close()
            
            print(f"Processed batch {batch_idx+1}")

if __name__ == "__main__":
    evaluate_model()