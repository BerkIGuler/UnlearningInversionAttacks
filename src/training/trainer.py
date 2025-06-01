import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.training.models import ConvNet

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
        self.model = ConvNet(width=config["model"]["width"],
                             num_classes=config["model"]["num_classes"],
                             num_channels=config["model"]["num_channels"]).to(self.device)

    def train_model(self, train_loader, epochs, lr, weight_decay, scheduler=True):
        """Train from scratch. Return trained model."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        if scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}: Loss = {running_loss / len(train_loader):.4f}")
            if scheduler:
                scheduler.step()

        return self.model

    def fine_tune_model(self, train_loader, epochs, lr):
        """Fine-tune pretrained model."""
        return self.train_model(train_loader, epochs=epochs, lr=lr, weight_decay=0, scheduler=False)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
