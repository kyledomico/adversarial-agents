import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import time
import os

def get_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description='PyTorch Fine-Tuning a Pre-trained Model')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0', 'vit_b_16', 'mobilenet_v3_large'],
                        help='Model to fine-tune (default: resnet50)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'svhn'],
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Directory to download data (default: ./dataset)')
    parser.add_argument('--save_dir', type=str, default='./victim_models',
                        help='Directory to save the trained model (default: ./victim_models)')
    return parser.parse_args()

def get_data(dataset_name, data_dir, batch_size):
    """
    Prepares the data loaders for the specified dataset.

    Args:
        dataset_name (str): The name of the dataset ('cifar10', 'svhn', 'cifar100').
        data_dir (str): The directory where the data is or will be stored.
        batch_size (int): The number of samples per batch.

    Returns:
        tuple: A tuple containing the training loader, validation loader, and number of classes.
    """
    # Pre-trained models expect 224x224 images and specific normalization values.
    # We must apply these transformations to our dataset.
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print(f"Loading dataset: {dataset_name.upper()}")

    # Datasets: CIFAR-10 and SVHN
    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=data_transforms['train'])
        val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=data_transforms['val'])
        num_classes = 10
    elif dataset_name == 'svhn':
        # SVHN uses 'split' instead of 'train'
        train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=data_transforms['train'])
        val_dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=data_transforms['val'])
        num_classes = 10
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader, num_classes


def get_model(model_name, num_classes, pretrained=True):
    """
    Loads a pre-trained model and replaces its final layer.

    Args:
        model_name (str): The name of the model architecture.
        num_classes (int): The number of output classes for the new task.
        pretrained (bool): Whether to load pre-trained weights.

    Returns:
        torch.nn.Module: The modified model.
    """
    print(f"Loading model: {model_name.upper()}")
    model = None
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        # The final fully connected layer in ResNet is named 'fc'
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        # The classifier in EfficientNet is a sequential block. We replace the final Linear layer.
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        # The classification head in ViT is named 'heads'
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        # The classifier in MobileNet is the last layer of a sequential block
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model

def train_model(model, model_name, dataset_name, train_loader, val_loader, criterion, optimizer, device, epochs, save_dir):
    """
    The main training and validation loop.
    """
    since = time.time()
    best_acc = 0.0

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}+{model_name}.pth")


    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only if in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the model if it has the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved to {save_path} with accuracy: {best_acc:.4f}")


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:4f}')


def main():
    """
    Main execution function.
    """
    args = get_args()
    print(f"Using arguments: {args}")

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data
    train_loader, val_loader, num_classes = get_data(args.dataset, args.data_dir, args.batch_size)

    # Get model
    model = get_model(args.model, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Use AdamW for ViT as it's often recommended, Adam for others is also a good choice.
    if args.model == 'vit_b_16':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Start training
    train_model(model, args.model, args.dataset, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.save_dir)

if __name__ == '__main__':
    main()