import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def create_hourglass_diagram():
    # Create output directory
    output_dir = "/Users/zohaib/Desktop/University/Software Project/Prototype/model_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 16))
    
    # Define colors
    colors = {
        'input': '#ffcdd2',
        'reshape': '#c8e6c9',
        'encoder': '#bbdefb',
        'latent': '#fff9c4',
        'decoder': '#d1c4e9',
        'output': '#ffccbc'
    }
    
    # Define layer dimensions
    base_height = 0.6
    y_spacing = 1.0
    center_x = 6.0
    
    # Define layer information with channel counts
    layers = [
        {'name': 'Input: (batch_size, 5, 3, 146, 146)', 'type': 'input', 'channels': 15, 'size': '(5, 3, 146, 146)'},
        {'name': 'Reshape: (batch_size, 15, 146, 146)', 'type': 'reshape', 'channels': 15, 'size': '(15, 146, 146)'},
        
        {'name': 'Conv2d(15→64, k=3, s=2, p=1)', 'type': 'encoder', 'channels': 64, 'size': '(64, 73, 73)'},
        {'name': 'BatchNorm2d(64) → LeakyReLU', 'type': 'encoder', 'channels': 64, 'size': '(64, 73, 73)'},
        
        {'name': 'Conv2d(64→128, k=3, s=2, p=1)', 'type': 'encoder', 'channels': 128, 'size': '(128, 37, 37)'},
        {'name': 'BatchNorm2d(128) → LeakyReLU', 'type': 'encoder', 'channels': 128, 'size': '(128, 37, 37)'},
        
        {'name': 'Conv2d(128→256, k=3, s=2, p=1)', 'type': 'encoder', 'channels': 256, 'size': '(256, 19, 19)'},
        {'name': 'BatchNorm2d(256) → LeakyReLU', 'type': 'encoder', 'channels': 256, 'size': '(256, 19, 19)'},
        
        {'name': 'Conv2d(256→128, k=3, s=2, p=1)', 'type': 'encoder', 'channels': 128, 'size': '(128, 10, 10)'},
        {'name': 'BatchNorm2d(128) → LeakyReLU', 'type': 'encoder', 'channels': 128, 'size': '(128, 10, 10)'},
        
        {'name': 'Latent Space', 'type': 'latent', 'channels': 128, 'size': '(128, 10, 10)'},
        
        {'name': 'ConvTranspose2d(128→256, k=3, s=2, p=1, op=1)', 'type': 'decoder', 'channels': 256, 'size': '(256, 19, 19)'},
        {'name': 'BatchNorm2d(256) → ReLU', 'type': 'decoder', 'channels': 256, 'size': '(256, 19, 19)'},
        
        {'name': 'ConvTranspose2d(256→128, k=3, s=2, p=1, op=1)', 'type': 'decoder', 'channels': 128, 'size': '(128, 37, 37)'},
        {'name': 'BatchNorm2d(128) → ReLU', 'type': 'decoder', 'channels': 128, 'size': '(128, 37, 37)'},
        
        {'name': 'ConvTranspose2d(128→64, k=3, s=2, p=1, op=1)', 'type': 'decoder', 'channels': 64, 'size': '(64, 73, 73)'},
        {'name': 'BatchNorm2d(64) → ReLU', 'type': 'decoder', 'channels': 64, 'size': '(64, 73, 73)'},
        
        {'name': 'ConvTranspose2d(64→15, k=3, s=2, p=1, op=1)', 'type': 'decoder', 'channels': 15, 'size': '(15, 146, 146)'},
        {'name': 'Tanh', 'type': 'decoder', 'channels': 15, 'size': '(15, 146, 146)'},
        
        {'name': 'Reshape: (batch_size, 5, 3, 146, 146)', 'type': 'reshape', 'channels': 15, 'size': '(5, 3, 146, 146)'},
        {'name': 'Output: (batch_size, 5, 3, 146, 146)', 'type': 'output', 'channels': 15, 'size': '(5, 3, 146, 146)'}
    ]
    
    # Find max channels for scaling
    max_channels = max(layer['channels'] for layer in layers)
    min_width = 2.0  # Minimum width for the smallest layer
    max_width = 10.0  # Maximum width for the largest layer
    
    # Draw layers
    for i, layer in enumerate(layers):
        y = 20 - i * y_spacing
        
        # Scale width based on number of channels (proportional to the hourglass shape)
        width = min_width + (layer['channels'] / max_channels) * (max_width - min_width)
        
        # Calculate x position to center the rectangle
        x = center_x - width / 2
        
        # Draw the rectangle
        rect = patches.Rectangle((x, y), width, base_height, 
                                linewidth=1, edgecolor='black', 
                                facecolor=colors[layer['type']], alpha=0.7)
        ax.add_patch(rect)
        
        # Add layer name
        ax.text(center_x, y + base_height/2, layer['name'], 
                horizontalalignment='center', verticalalignment='center',
                fontsize=8)
        
        # Add size information on the right
        ax.text(center_x + max_width/2 + 0.5, y + base_height/2, layer['size'], 
                horizontalalignment='left', verticalalignment='center', fontsize=7)
        
        # Add channel count on the left
        ax.text(center_x - max_width/2 - 0.5, y + base_height/2, f"{layer['channels']} ch", 
                horizontalalignment='right', verticalalignment='center', fontsize=7)
        
        # Add arrow
        if i < len(layers) - 1:
            ax.arrow(center_x, y, 0, -y_spacing + base_height, head_width=0.2, 
                    head_length=0.1, fc='black', ec='black')
    
    # Set plot properties
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 21)
    ax.set_title('Video Autoencoder Architecture (Hourglass Visualization)', fontsize=16)
    ax.axis('off')
    
    # Add legend for layer types
    legend_elements = [patches.Patch(facecolor=color, edgecolor='black', alpha=0.7, label=layer_type)
                      for layer_type, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add annotations
    plt.figtext(0.5, 0.01, 'Width of each layer is proportional to the number of channels', 
                ha='center', fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hourglass_architecture_diagram.png'), dpi=300)
    print(f"Hourglass diagram saved to {output_dir}/hourglass_architecture_diagram.png")

if __name__ == "__main__":
    create_hourglass_diagram()