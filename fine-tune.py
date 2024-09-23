import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import clip
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


# Function to convert TIFF images to RGB (with proper file handling and error catching)
def convert_tiff_to_rgb(image_path):
    try:
        with Image.open(image_path) as img:
            return img.convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None


# Custom dataset to load only TIFF images
class TiffImageDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        # List all TIFF files in the dataset path
        self.samples = [
            (os.path.join(root, file), os.path.basename(root))  # Full path and class name (folder name)
            for root, _, files in os.walk(dataset_path)
            for file in files if file.lower().endswith(('.tiff', '.tif'))
        ]
        # Create a mapping of class names to indices
        self.class_names = list(sorted({label for _, label in self.samples}))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_name = self.samples[idx]
        image = convert_tiff_to_rgb(image_path)

        if image is None:  # Handle any problematic image
            print(f"Skipping problematic image: {image_path}")
            return self.__getitem__((idx + 1) % len(self.samples))  # Skip bad image

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[class_name]  # Convert class name to a numeric label
        return image, label


# Define transformations for TIFF images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

# Load dataset
dataset_path = '/Users/neftekhari/Documents/corrected-elastic-egss/particle-data/New-taxonomy-copy'
tiff_dataset = TiffImageDataset(dataset_path=dataset_path, transform=transform)

# Split dataset into train, validation, and test sets
indices = list(range(len(tiff_dataset)))
train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
train_indices, val_indices = train_test_split(train_indices, test_size=0.1, random_state=42)

train_dataset = torch.utils.data.Subset(tiff_dataset, train_indices)
val_dataset = torch.utils.data.Subset(tiff_dataset, val_indices)
test_dataset = torch.utils.data.Subset(tiff_dataset, test_indices)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)


# Loss function (contrastive loss)
def contrastive_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=device)
    loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
    return (loss_i + loss_t) / 2


# Training loop
num_epochs = 5  # Adjust as needed
best_val_loss = float('inf')
best_model_path = "best_clip_contrastive.pth"

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        texts = clip.tokenize([tiff_dataset.class_names[label] for label in labels]).to(device)

        optimizer.zero_grad()

        # Get image and text embeddings
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Compute logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Compute loss
        loss = contrastive_loss(logits_per_image, logits_per_text)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            texts = clip.tokenize([tiff_dataset.class_names[label] for label in labels]).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            loss = contrastive_loss(logits_per_image, logits_per_text)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved Best Model at Epoch {epoch + 1}")


# Testing on three test sets (assuming you have three different test datasets)
def test_model(test_loader, description):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

