import mlflow
import mlflow.pytorch
from mlflow_setup import configure_mlflow

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import BratsDataset
from model import UNet

configure_mlflow()

params = {
    "epochs": 10,
    "batch_size": 2,
    "lr": 1e-4,
    "model": "UNet",
    "optimizer": "Adam"
}

with mlflow.start_run():
    mlflow.log_params(params)

    # Датасет
    train_dataset = BratsDataset(split='train')
    val_dataset = BratsDataset(split='val')
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = UNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    for epoch in range(params["epochs"]):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

    mlflow.pytorch.log_model(model, "model")
