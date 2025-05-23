import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from video_dataset import VideoFrameDataset
from video_autoencoder import VideoAutoencoder
import multiprocessing

# Move all code into a function
def train_model():
    # Parameters
    video_dir = "/Users/zohaib/Desktop/University/Software Project/Prototype/videos"  # Update this path
    sequence_length = 5
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 50
    latent_dim = 128
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Create directory for videos if it doesn't exist
    os.makedirs(video_dir, exist_ok=True)

    # Check if there are video files in the directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"Error: No video files found in {video_dir}")
        print("Please add video files to this directory before running the script.")
        print("Supported formats: .mp4, .avi, .mov")
        exit(1)
    else:
        print(f"Found {len(video_files)} video files: {video_files}")

    # Create transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((146, 146)),  # Resize frames to match model's expected input size
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    try:
        dataset = VideoFrameDataset(video_dir, sequence_length=sequence_length, transform=transform)
        print(f"Dataset created with {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("Error: Dataset is empty. Check that your videos contain enough frames.")
            exit(1)
            
        # Set num_workers=0 to avoid multiprocessing issues
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        exit(1)

    # Create model, loss function, and optimizer
    model = VideoAutoencoder(sequence_length=sequence_length, in_channels=3, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            # Move data to device
            data = data.to(device)
            
            # Forward pass
            reconstructed, latent = model(data)
            
            # Calculate loss
            loss = criterion(reconstructed, data)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.6f}")
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, f"/Users/zohaib/Desktop/University/Software Project/Prototype/model_checkpoint_epoch_{epoch+1}.pth")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig("/Users/zohaib/Desktop/University/Software Project/Prototype/training_loss.png")
    plt.show()

    # Save the final model
    torch.save(model.state_dict(), "/Users/zohaib/Desktop/University/Software Project/Prototype/video_autoencoder_final.pth")

# Simplified for macOS on Silicon Mac
if __name__ == '__main__':
    train_model()