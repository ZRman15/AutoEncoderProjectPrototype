import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from video_dataset import VideoFrameDataset
from video_autoencoder import VideoAutoencoder


def train_model():
    # parameters for model training 
    video_dir = "/Users/zohaib/Desktop/University/Software Project/Prototype/videos"  #video directory
    sequence_length = 5
    batch_size = 10
    learning_rate = 0.001
    num_epochs = 8
    latent_dim = 128
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # check if there are video files in the directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"Error: No video files found in {video_dir}")
        print("Please add video files to this directory before running the script.")
        print("Supported formats: .mp4, .avi, .mov")
        exit(1)
    else:
        print(f"Found {len(video_files)} video files: {video_files}")

    # create transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),  # resize frames to match model's expected input size
        transforms.ToTensor(),
    ])

    # create dataset and dataloader
    try:
        dataset = VideoFrameDataset(video_dir, sequence_length=sequence_length, transform=transform)
        print(f"Dataset created with {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("Error: Dataset is empty.")
            exit(1)

        # num_workers=0 to avoid issues with DataLoader / not really useful for macOS    
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        exit(1)

    # create model, loss function, and optimizer
    model = VideoAutoencoder(sequence_length=sequence_length, in_channels=3, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    losses = []
    
    # starting time
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            # moving data to device 
            data = data.to(device)
            
            # forward
            reconstructed, latent = model(data)
            
            # calculating loss
            loss = criterion(reconstructed, data)
            
            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update model stats
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        
        # Calculate and display time for this epoch
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f} seconds")
        
        # save model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, f"/Users/zohaib/Desktop/University/Software Project/Prototype/model_checkpoint_epoch_{epoch+1}.pth")

    # Calculate total training time
    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")

    # plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(f'Training Loss (Total time: {total_training_time:.2f}s, Samples: {len(dataset)})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig("/Users/zohaib/Desktop/University/Software Project/Prototype/training_loss.png")
    plt.show()

    # save the final trained model
    torch.save(model.state_dict(), "/Users/zohaib/Desktop/University/Software Project/Prototype/video_autoencoder_final.pth")
    
    return total_training_time

if __name__ == '__main__':
    train_model()