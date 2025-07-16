import os
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Dataset Setup with MTCNN
# ---------------------------
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, mtcnn, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.mtcnn = mtcnn
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB')
        face = self.mtcnn(img)

        if face is None:
            # Return black image if no face detected
            face = torch.zeros((3, 224, 224))

        if self.transform:
            face = self.transform(face)

        return face, label

# --------------------------
# 2. Prepare Dataset
# --------------------------
def load_dataset(root_dir, mtcnn):
    classes = sorted(os.listdir(root_dir))
    label_map = {cls: i for i, cls in enumerate(classes)}

    image_paths = []
    labels = []

    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image_paths.append(img_path)
            labels.append(label_map[cls])

    return image_paths, labels, label_map

# --------------------------
# 3. Train and Evaluate
# --------------------------
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=5):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# --------------------------
# 4. Main Function
# --------------------------
def main():
    dataset_dir = r"D:\elc_2nd"  # Replace with your actual dataset path
    batch_size = 16
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtcnn = MTCNN(image_size=224, margin=20)

    # Load dataset
    image_paths, labels, label_map = load_dataset(dataset_dir, mtcnn)
    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, stratify=labels)

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = FaceDataset(train_paths, train_labels, mtcnn, transform)
    val_dataset = FaceDataset(val_paths, val_labels, mtcnn, transform)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size)
    }

    # Load pretrained ResNet
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(label_map))
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train
    train_model(model, dataloaders, criterion, optimizer, device, num_epochs)

    # Save model
    torch.save(model.state_dict(), "resnet_driver_drowsiness.pth")
    print("✅ Model saved as 'resnet_driver_drowsiness.pth'")

if __name__ == "__main__":
    main()
