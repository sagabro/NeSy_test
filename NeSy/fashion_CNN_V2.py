import random
from pathlib import Path
import requests
import torch
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

class fashion_modelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_shape: int, output_shape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_shape, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=hidden_shape, out_channels=hidden_shape, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_shape, hidden_shape, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(hidden_shape, hidden_shape, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_shape * 7 * 7, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.block2(self.block1(x)))


def train_step(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device: torch.device):
    model.train()
    model.to(device)
    train_loss, train_acc = 0.0, 0.0
    for X, y in data_loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def test_step(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, accuracy_fn, device: torch.device):
    test_loss, test_acc = 0.0, 0.0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            test_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")


def predict_image(model, dataset, class_names, device):
    random_idx = random.randint(0, len(dataset) - 1)
    img, label = dataset[random_idx]
    model.eval()
    model.to(device)
    with torch.inference_mode():
        img = img.unsqueeze(dim=0).to(device)
        pred_logit = model(img)
        pred_prob = torch.softmax(pred_logit, dim=1)
        pred_label_idx = pred_prob.argmax(dim=1).item()
        pred_label = class_names[pred_label_idx]
    print(f"Predicted label: {pred_label}, Actual label: {class_names[label]}")
    return pred_label


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    helper_path = Path("helper_functions.py")
    if not helper_path.is_file():
        print("Downloading helper_functions.py")
        request = requests.get("https://githubusercontent.com")
        helper_path.write_bytes(request.content)

    from helper_functions import accuracy_fn

    train_data = datasets.FashionMNIST(root="NeSy", train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root="NeSy", train=False, download=True, transform=ToTensor())
    class_names = train_data.classes

    BATCH_SIZE = 512
    NUM_WORKERS = 4
    use_cuda = torch.cuda.is_available()

    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    torch.manual_seed(42)
    modelV2 = fashion_modelV2(1, 10, len(class_names)).to(device)

    optimizer_v2 = torch.optim.Adam(params=modelV2.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 200
    for epoch in tqdm(range(epochs)):
        print(f"\nEpoch: {epoch + 1}\n---------")
        train_step(modelV2, train_dataloader, loss_fn, optimizer_v2, accuracy_fn, device)
        test_step(modelV2, test_dataloader, loss_fn, accuracy_fn, device)

    torch.save(modelV2.state_dict(), "fashion_model_v2.pth")

    loaded_model = fashion_modelV2(input_shape=1, hidden_shape=10, output_shape=len(class_names))
    loaded_model.load_state_dict(torch.load("fashion_model_v2.pth", weights_only=True))
    loaded_model.to(device)
    loaded_model.eval()

    predict_image(loaded_model, test_data, class_names, device)