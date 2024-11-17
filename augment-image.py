import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from model import AugmentedMNIST

class AugmentedMNISTDemo(AugmentedMNIST):
    """Extended version of AugmentedMNIST with specific transformations for visualization"""
    def __init__(self, root='./data', train=True):
        super().__init__(root, train)
        
        # Define individual transformations for visualization
        self.transformations = {
            'Rotation (10°)': transforms.Compose([
                transforms.RandomRotation(degrees=10, fill=0),
                transforms.ToTensor()
            ]),
            'Scale (90%)': transforms.Compose([
                transforms.RandomAffine(degrees=0, scale=(0.9, 0.9), fill=0),
                transforms.ToTensor()
            ]),
            'Translation\n(10% right)': transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.1, 0), fill=0),
                transforms.ToTensor()
            ])
        }

    def get_specific_augmentation(self, img, transform_name):
        """Apply a specific transformation"""
        return self.transformations[transform_name](img)

def create_readme_visualization(dataset, num_samples=3):
    plt.style.use('seaborn-v0_8-white')
    
    # Create figure with white background
    fig = plt.figure(figsize=(12, 4))
    fig.patch.set_facecolor('white')
    
    # Add title
    plt.suptitle('MNIST Digit Augmentation Examples', fontsize=14, y=1.05)
    
    # Define transformations to show
    transform_names = ['Rotation (10°)', 'Scale (90%)', 'Translation\n(10% right)']
    
    # Create grid of subplots
    for i in range(num_samples):
        img, label = dataset.original_dataset[i]
        
        # Original image
        plt.subplot(num_samples, len(transform_names) + 1, i * (len(transform_names) + 1) + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Original', pad=10)
        
        # Augmented versions with specific transformations
        for j, transform_name in enumerate(transform_names):
            augmented_img = dataset.get_specific_augmentation(img, transform_name)
            plt.subplot(num_samples, len(transform_names) + 1, i * (len(transform_names) + 1) + j + 2)
            plt.imshow(augmented_img.squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title(transform_name, pad=10)
    
    # Add row labels
    for i in range(num_samples):
        fig.text(0.01, 0.75 - (i * 0.29), f'Digit {dataset.original_dataset[i][1]}', 
                ha='left', va='center', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high DPI and transparent background
    plt.savefig('augmentation_samples.png', 
                dpi=150, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create dataset with augmentations
    augmented_dataset = AugmentedMNISTDemo(train=True)
    
    # Create and save visualization
    create_readme_visualization(augmented_dataset)
    print("Created README visualization: augmentation_samples.png")

if __name__ == "__main__":
    main()