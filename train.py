import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, random_split
import pickle

def main():
    # ✅ Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ✅ Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    # ✅ Dataset Path
    dataset_path = r"D:\cattle breed regonition\archive\Final breed conference 2211\train"

    dataset =datasets.ImageFolder(root=dataset_path, transform=transform)

    print("Classes:", dataset.classes)
    print("Total Images:", len(dataset))

    # ❌ Safety check
    if len(dataset.classes) < 2:
        raise ValueError("❌ Dataset must have multiple class folders!")

    # ✅ Split dataset (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # ✅ DataLoaders (Windows FIX)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # ✅ Model (Transfer Learning)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    num_classes = len(dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)

    # ✅ Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ✅ Training
    num_epochs = 20
    print("\n🚀 Training Started...\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # ✅ Testing
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"\n🎯 Test Accuracy: {test_acc:.2f}%")

    # ✅ Save model
    torch.save(model.state_dict(), "cattle_model4.pth")

    # ✅ Save class labels
    with open("classes.pkl", "wb") as f:
        pickle.dump(dataset.classes, f)

    print("✅ Model & Classes Saved!")

# ✅ IMPORTANT for Windows
if __name__ == "__main__":
    main()