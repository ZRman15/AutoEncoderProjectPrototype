import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def create_matplotlib_diagram():
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
    layer_width = 8
    layer_height = 0.6
    y_spacing = 0.8
    
    # Define layer positions and properties
    layers = [
        {'name': 'Input: (batch_size, 5, 3, 146, 146)', 'type': 'input', 'size': '(5, 3, 146, 146)'},
        {'name': 'Reshape: (batch_size, 15, 146, 146)', 'type': 'reshape', 'size': '(15, 146, 146)'},
        
        {'name': 'Conv2d(15→64, k=3, s=2, p=1)', 'type': 'encoder', 'size': '(64, 73, 73)'},
        {'name': 'BatchNorm2d(64) → LeakyReLU', 'type': 'encoder', 'size': '(64, 73, 73)'},
        
        {'name': 'Conv2d(64→128, k=3, s=2, p=1)', 'type': 'encoder', 'size': '(128, 37, 37)'},
        {'name': 'BatchNorm2d(128) → LeakyReLU', 'type': 'encoder', 'size': '(128, 37, 37)'},
        
        {'name': 'Conv2d(128→256, k=3, s=2, p=1)', 'type': 'encoder', 'size': '(256, 19, 19)'},
        {'name': 'BatchNorm2d(256) → LeakyReLU', 'type': 'encoder', 'size': '(256, 19, 19)'},
        
        {'name': 'Conv2d(256→128, k=3, s=2, p=1)', 'type': 'encoder', 'size': '(128, 10, 10)'},
        {'name': 'BatchNorm2d(128) → LeakyReLU', 'type': 'encoder', 'size': '(128, 10, 10)'},
        
        {'name': 'Latent Space', 'type': 'latent', 'size': '(128, 10, 10)'},
        
        {'name': 'ConvTranspose2d(128→256, k=3, s=2, p=1, op=1)', 'type': 'decoder', 'size': '(256, 19, 19)'},
        {'name': 'BatchNorm2d(256) → ReLU', 'type': 'decoder', 'size': '(256, 19, 19)'},
        
        {'name': 'ConvTranspose2d(256→128, k=3, s=2, p=1, op=1)', 'type': 'decoder', 'size': '(128, 37, 37)'},
        {'name': 'BatchNorm2d(128) → ReLU', 'type': 'decoder', 'size': '(128, 37, 37)'},
        
        {'name': 'ConvTranspose2d(128→64, k=3, s=2, p=1, op=1)', 'type': 'decoder', 'size': '(64, 73, 73)'},
        {'name': 'BatchNorm2d(64) → ReLU', 'type': 'decoder', 'size': '(64, 73, 73)'},
        
        {'name': 'ConvTranspose2d(64→15, k=3, s=2, p=1, op=1)', 'type': 'decoder', 'size': '(15, 146, 146)'},
        {'name': 'Tanh', 'type': 'decoder', 'size': '(15, 146, 146)'},
        
        {'name': 'Reshape: (batch_size, 5, 3, 146, 146)', 'type': 'reshape', 'size': '(5, 3, 146, 146)'},
        {'name': 'Output: (batch_size, 5, 3, 146, 146)', 'type': 'output', 'size': '(5, 3, 146, 146)'}
    ]
    
    # Draw layers
    for i, layer in enumerate(layers):
        y = 20 - i * y_spacing
        rect = patches.Rectangle((2, y), layer_width, layer_height, 
                                linewidth=1, edgecolor='black', 
                                facecolor=colors[layer['type']], alpha=0.7)
        ax.add_patch(rect)
        ax.text(2 + layer_width/2, y + layer_height/2, layer['name'], 
                horizontalalignment='center', verticalalignment='center')
        
        # Add size information
        ax.text(11, y + layer_height/2, layer['size'], 
                horizontalalignment='left', verticalalignment='center', fontsize=9)
        
        # Add arrow
        if i < len(layers) - 1:
            ax.arrow(2 + layer_width/2, y, 0, -y_spacing + layer_height, head_width=0.2, 
                    head_length=0.1, fc='black', ec='black')
    
    # Set plot properties
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 21)
    ax.set_title('Video Autoencoder Architecture', fontsize=16)
    ax.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_architecture_diagram.png'), dpi=300)
    print(f"Matplotlib diagram saved to {output_dir}/model_architecture_diagram.png")

if __name__ == "__main__":
    create_matplotlib_diagram()