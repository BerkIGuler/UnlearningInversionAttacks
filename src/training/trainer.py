import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.training.models import ConvNet


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        try:
            self.device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
        except KeyError:
            self.device = torch.device(config["fine_tune"]["device"] if torch.cuda.is_available() else "cpu")
        self.model = ConvNet(width=config["model"]["width"],
                             num_classes=config["model"]["num_classes"],
                             num_channels=config["model"]["num_channels"]).to(self.device)

    def evaluate_model(self, val_loader):
        """Evaluate model on validation set and return per-class accuracy."""
        self.model.eval()
        num_classes = self.config["model"]["num_classes"]
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_correct[label] += (predicted[i] == targets[i]).item()
                    class_total[label] += 1

        class_accuracies = {}
        overall_correct = sum(class_correct)
        overall_total = sum(class_total)

        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy = class_correct[i] / class_total[i]
                class_accuracies[f"Class_{i}"] = accuracy
            else:
                class_accuracies[f"Class_{i}"] = 0.0

        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
        class_accuracies["Overall"] = overall_accuracy

        return class_accuracies

    def train_model(self, train_loader, val_loader, epochs, lr, weight_decay, scheduler=True):
        """Train from scratch. Return trained model."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        if scheduler:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1, end_factor=0.01, total_iters=epochs)

        eval_every_n = self.config["train"]["eval_every_n"]

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

            if (epoch + 1) % eval_every_n == 0:
                accuracies = self.evaluate_model(val_loader)
                print(f"Validation Results after Epoch {epoch + 1}:")
                for class_name, accuracy in accuracies.items():
                    print(f"  {class_name}: {accuracy:.4f}")
                self.model.train()

            if scheduler:
                scheduler.step()

        return self.model

    def fine_tune_model(self, train_loader, val_loader, epochs, lr):
        """Fine-tune pretrained model."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0)

        eval_every_n = self.config["fine_tune"]["eval_every_n"]

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch + 1}/{epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Fine-tune Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

            if (epoch + 1) % eval_every_n == 0:
                accuracies = self.evaluate_model(val_loader)
                print(f"Validation Results after Fine-tune Epoch {epoch + 1}:")
                for class_name, accuracy in accuracies.items():
                    print(f"  {class_name}: {accuracy:.4f}")
                self.model.train()

        return self.model

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)