import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import time
import os
import torchattacks

# A custom nn.Module that combines preprocessing and the model
class ModelWithPreprocessing(nn.Module):
    def __init__(self, model, preprocess):
        super().__init__()
        self.model = model
        self.preprocess = preprocess

    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)

def get_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description='PyTorch Square Attack Benchmark on Raw Images')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0', 'vit_b_16', 'mobilenet_v3_large'],
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'svhn'],
                        help='Dataset to use for validation')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for testing (default: 32)')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Directory to download data (default: ./data)')
    parser.add_argument('--epsilon', type=float, default=0.35,
                        help='Epsilon value for the Square attack (default: 0.3)')
    parser.add_argument('--n_queries', type=int, default=1000, 
                        help='Number of queries for the Square attack (default: 1500)')
    return parser.parse_args()

def get_data(dataset_name, data_dir, batch_size):
    raw_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print(f"Loading dataset: {dataset_name.upper()}")
    if dataset_name == 'cifar10':
        val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=raw_transform)
        num_classes = 10
    elif dataset_name == 'svhn':
        val_dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=raw_transform)
        num_classes = 10
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Validation samples: {len(val_dataset)}")
    return val_loader, num_classes

def get_model(model_name, num_classes, model_path, device):
    """
    Loads a model architecture and then loads the fine-tuned weights into it.
    """
    print(f"Loading model architecture: {model_name.upper()}")
    model = None
    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=None)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=None)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")
        
    print(f"Loading weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the state_dict into the model architecture
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model

def main():
    """
    Main execution function for the attack benchmark.
    """
    args = get_args()
    print(f"Using arguments: {args}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_loader, num_classes = get_data(args.dataset, args.data_dir, args.batch_size)

    # 1. Load the entire model directly from the path.
    model_path = './victim_models/' + args.dataset + '+resnet50.pth'
    base_model = get_model(args.model, num_classes, model_path, device)
    
    # 2. Define the preprocessing steps
    preprocess = nn.Sequential(
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ).to(device)

    # 3. Create the full model and ensure it's on the correct device
    full_model = ModelWithPreprocessing(base_model, preprocess)
    full_model.to(device) # Explicitly move the wrapped model to the device
    full_model.eval()

    # 4. Initialize the attack on the full model.
    #    The attack object itself does not need to be moved to the device.
    atk = torchattacks.Square(full_model, verbose=False, eps=args.epsilon, n_queries=args.n_queries, norm='L2')
    print(f"Initialized Attack: {atk}")

    # Benchmark loop
    clean_correct, adv_correct, successes, total, batch_idx = 0, 0, 0, 0, 0
    num_batches = len(val_loader)
    start_time = time.time()
    
    print("Starting attack on raw images...")
    for images, labels in val_loader:
        # Move data to the same device as the model
        images, labels = images.to(device), labels.to(device)

        outputs_clean = full_model(images)
        clean_correct += (torch.max(outputs_clean, 1)[1] == labels).sum().item()

        adv_images = atk(images, labels)

        outputs_adv = full_model(adv_images)
        adv_correct += (torch.max(outputs_adv, 1)[1] == labels).sum().item()
        successes += (torch.max(outputs_adv, 1)[1] != labels).sum().item()
        
        total += labels.size(0)
        batch_idx += 1
        
        print(f"--- Finished batch {batch_idx} of {num_batches} ---")


    end_time = time.time()
    
    clean_accuracy = 100 * clean_correct / total
    robust_accuracy = 100 * adv_correct / total
    attack_success_rate = 100 * successes / total

    print("\n--- Benchmark Results (Attacking Raw Images) ---")
    print(f"Model Path:      {model_path}")
    print(f"Dataset:         {args.dataset}")
    print(f"Epsilon:         {args.epsilon}")
    print(f"Number of Queries: {args.n_queries}")
    print("-" * 25)
    print(f"Clean Accuracy:  {clean_accuracy:.2f}%")
    print(f"Robust Accuracy: {robust_accuracy:.2f}% (After Square Attack)")
    print(f"Attack Success Rate: {attack_success_rate:.2f}%")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print("-------------------------------------------------")


if __name__ == '__main__':
    main()