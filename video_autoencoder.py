import torch
import torch.nn as nn

class VideoAutoencoder(nn.Module):
    def __init__(self, sequence_length=5, in_channels=3, latent_dim=128, input_size=(146, 146)):
        """
        Autoencoder for video compression
        
        Args:
            sequence_length (int): Number of frames in each sequence
            in_channels (int): Number of input channels (3 for RGB)
            latent_dim (int): Dimension of the latent space
            input_size (tuple): Height and width of input frames (146, 146)
        """
        super(VideoAutoencoder, self).__init__()
        
        # Store input parameters
        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.input_size = input_size
        
        # Encoder
        self.encoder = nn.Sequential(
            # First conv layer
            nn.Conv2d(sequence_length*in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),  # Add batch normalization
            nn.LeakyReLU(0.2, inplace=True),  # Use LeakyReLU instead of ReLU
            
            # Second conv layer
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third conv layer
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fourth conv layer
            nn.Conv2d(256, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # First transposed conv
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Second transposed conv
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Third transposed conv
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Fourth transposed conv
            nn.ConvTranspose2d(64, sequence_length*in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Use Tanh instead of Sigmoid for better gradient flow
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, channels, height, width = x.size()
        
        # Reshape to combine sequence and channels dimensions for 2D convolution
        x = x.view(batch_size, seq_len * channels, height, width)
        
        # Scale input to [-1, 1] range for Tanh activation
        x = (x * 2) - 1
        
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        # Scale output back to [0, 1] range
        reconstructed = (reconstructed + 1) / 2
        
        # Get the actual output shape
        _, out_channels, out_height, out_width = reconstructed.shape
        
        # Check if dimensions match before reshaping
        if out_channels == seq_len * channels and out_height == height and out_width == width:
            # Reshape back to sequence format
            reconstructed = reconstructed.view(batch_size, seq_len, channels, height, width)
        else:
            # If dimensions don't match, resize the tensor to match expected dimensions
            print(f"Warning: Output dimensions mismatch. Expected {seq_len*channels}x{height}x{width}, got {out_channels}x{out_height}x{out_width}")
            # Resize to match expected dimensions
            reconstructed = torch.nn.functional.interpolate(
                reconstructed, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
            # Reshape to sequence format
            reconstructed = reconstructed.view(batch_size, seq_len, channels, height, width)
        
        return reconstructed, latent