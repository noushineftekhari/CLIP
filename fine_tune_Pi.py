import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold
import clip

# Custom Dataset for your TIFF images and folder-based labels
class PlanktonDataset(Dataset):
    def __init__(self, root_dir, preprocess, tokenizer):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.data = []

        # Traverse directories and load image paths and corresponding labels (folder names)
        for label in os.listdir(root_dir):
            label_folder = os.path.join(root_dir, label)
            if os.path.isdir(label_folder):
                for img_file in os.listdir(label_folder):
                    if img_file.endswith('.tiff') or img_file.endswith('.tif'):
                        img_path = os.path.join(label_folder, img_file)
                        self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")  # Convert TIFF to RGB
        image = self.preprocess(image)
        text = self.tokenizer([label])[0]  # Tokenize label (folder name)
        return image, text

# Function to train for one fold
def train_one_fold(model, train_loader, val_loader, device, optimizer, criterion, fold):
    model.train()
    for epoch in range(10):  # Set the number of epochs
        running_loss = 0.0
        for batch_idx, (images, texts) in enumerate(train_loader):
            images = images.to(device)
            texts = texts.to(device)

            # Forward pass: get image and text embeddings
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            # Compute similarity (dot product) and softmax over similarity scores
            logits = (image_features @ text_features.T).softmax(dim=-1)

            # Create labels for contrastive loss (identity matrix)
            labels = torch.arange(len(images)).to(device)

            # Compute loss
            loss = criterion(logits, labels)
            running_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Fold [{fold}], Epoch [{epoch+1}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, texts in val_loader:
            images = images.to(device)
            texts = texts.to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            logits = (image_features @ text_features.T).softmax(dim=-1)
            labels = torch.arange(len(images)).to(device)

            loss = criterion(logits, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Fold [{fold}] Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

# Main cross-validation fine-tuning script
def main(args):
    # Define dataset directory
    dataset_dir = args.dataset_dir

    # Load CLIP model and preprocessing functions
    device = args.device
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Prepare dataset
    dataset = PlanktonDataset(root_dir=dataset_dir, preprocess=preprocess, tokenizer=clip.tokenize)

    # 3-Fold Cross-Validation setup
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Training fold {fold+1}...")

        # Create data loaders for training and validation sets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)

        # Optimizer and loss function (contrastive or cross-entropy)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        criterion = torch.nn.CrossEntropyLoss()

        # Train and validate for this fold
        val_loss = train_one_fold(model, train_loader, val_loader, device, optimizer, criterion, fold+1)
        fold_losses.append(val_loss)

    # Average validation loss across all folds
    avg_loss = sum(fold_losses) / len(fold_losses)
    print(f"Average Validation Loss across 3 folds: {avg_loss:.4f}")

    # Save the fine-tuned model after all folds
    torch.save(model.state_dict(), "fine_tuned_clip_cv.pth")
    print("Fine-tuned model saved as 'fine_tuned_clip_cv.pth'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CLIP with 3-Fold Cross-Validation on custom TIFF dataset.")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the dataset folder")
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Device to run the model on. Options: 'cuda', 'cpu', 'mps' (for Macs)")
    args = parser.parse_args()
    main(args)
