import torch
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from timeit import default_timer as timer
from tqdm.auto import tqdm
import random

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Using CPU")

if Path("../helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("../helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import accuracy_fn


train_data = datasets.FashionMNIST(
    root="NeSy",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="NeSy",
    train=False,
    download=True,
    transform=ToTensor(),
)

BATCH_SIZE = 32

train_dataloader = DataLoader(train_data,
        batch_size=BATCH_SIZE,
        shuffle=True)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))

class_names = train_data.classes

flatten_model = nn.Flatten()
X = train_features_batch[0]

output = flatten_model(X)

def model_eval(model: nn.Module, data_loader: torch.utils.data.DataLoader,loss_fn: nn.Module,accuracy_fn, device: torch.device = device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X,y = X.to(device), y.to(device)
            test_pred = model(X)
            loss += loss_fn(test_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device: torch.device = device):
    model.train()
    model.to(device)
    train_loss,train_acc = 0,0
    for batch, (X,y) in enumerate(data_loader):

        X,y = X.to(device), y.to(device)

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

def test_step(model: torch.nn.Module ,data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn, device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:

            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            test_loss += loss_fn(y_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")

epochs = 10

class fashion_modelV2(nn.Module):
    def __init__(self,input_shape: int , hidden_shape: int, output_shape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_shape, kernel_size = 3, stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=hidden_shape, out_channels=output_shape, kernel_size = 3, stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope=0.01),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_shape, hidden_shape, 3, padding = 1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(hidden_shape, hidden_shape, 3, padding = 1),
            nn.LeakyReLU(negative_slope=0.01),
            MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_shape*7*7, out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        return self.classifier(self.block2(self.block1(x)))

torch.manual_seed(42)
modelV2 = fashion_modelV2(1,10,len(class_names)).to(device)

optimizer_v2 = torch.optim.SGD(params=modelV2.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

for epoch in tqdm(range(epochs)):
    train_step(modelV2,
               train_dataloader,
               loss_fn,
               optimizer_v2,
               accuracy_fn,
               device)
    test_step(modelV2,
              test_dataloader,
              loss_fn,
              accuracy_fn,
              device)

torch.save(modelV2.state_dict(), "fashion_model_v2.pth")

loaded_model = fashion_modelV2(
    input_shape=1,
    hidden_shape=10,
    output_shape=len(class_names)
)

loaded_model.load_state_dict(torch.load("fashion_model_v2.pth"))
loaded_model.to(device)
loaded_model.eval()

def predict_image(model, dataset, class_names, device):
    random_idx = random.randint(0, len(dataset) - 1 )
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

predict_image(modelV2, test_data, class_names, device)