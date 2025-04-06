import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from video_dataset import VideoFrameDataset
from video_autoencoder import VideoAutoencoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_latent_space():
    # Parameters
    video_dir = "/Users/zohaib/Desktop/University/Software Project/Prototype/videos"
    model_path = "/Users/zohaib/Desktop/University/Software Project/Prototype/video_autoencoder_final.pth"
    sequence_length = 5
    batch_size = 10
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
    
    # Create output directory for visualizations
    vis_dir = "/Users/zohaib/Desktop/University/Software Project/Prototype/latent_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Collect latent representations
    latent_vectors = []
    
    with torch.no_grad():
        for data in dataloader:
            # Move data to device
            data = data.to(device)
            
            # Forward pass to get latent representation
            _, latent = model(data)
            
            # Flatten the latent representation
            # Shape: [batch_size, latent_dim, h, w] -> [batch_size, latent_dim * h * w]
            batch_size = latent.size(0)
            latent_flat = latent.view(batch_size, -1).cpu().numpy()
            
            latent_vectors.append(latent_flat)
    
    # Concatenate all latent vectors
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    print(f"Collected {latent_vectors.shape[0]} latent vectors of dimension {latent_vectors.shape[1]}")
    
    # Visualize the distribution of latent values
    plt.figure(figsize=(10, 6))
    plt.hist(latent_vectors.flatten(), bins=50)
    plt.title('Distribution of Latent Space Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(f"{vis_dir}/latent_distribution.png")
    plt.close()
    
    # Reduce dimensionality for visualization
    # First with PCA
    if latent_vectors.shape[0] > 1:  # Need at least 2 samples for PCA
        pca = PCA(n_components=2)
        latent_2d_pca = pca.fit_transform(latent_vectors)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], alpha=0.5)
        plt.title('PCA Visualization of Latent Space')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar()
        plt.savefig(f"{vis_dir}/latent_pca.png")
        plt.close()
        
        # Then with t-SNE for better visualization of clusters
        if latent_vectors.shape[0] > 10:  # t-SNE works better with more samples
            tsne = TSNE(n_components=2, random_state=42)
            latent_2d_tsne = tsne.fit_transform(latent_vectors)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], alpha=0.5)
            plt.title('t-SNE Visualization of Latent Space')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.savefig(f"{vis_dir}/latent_tsne.png")
            plt.close()
    
    print(f"Latent space visualizations saved to {vis_dir}")

if __name__ == "__main__":
    visualize_latent_space()