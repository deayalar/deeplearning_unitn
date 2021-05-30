import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

PRETRAINED_MODELS = {
    "resnet18": { "load": lambda : models.resnet18(pretrained=True), "feature_size": 512},
    "resnet50": { "load": lambda : models.resnet50(pretrained=True), "feature_size": 2048}
    # More pretrained models here e.g. alexnet, vgg16, etc
}

class FinetunedModel(nn.Module):
    def __init__(self, architecture, n_classes):
        super(FinetunedModel, self).__init__()
        self.architecture = architecture

        self.backbone = PRETRAINED_MODELS[architecture]["load"]()
        self.feature_size = PRETRAINED_MODELS[architecture]["feature_size"]
        print(f"Backbone feature size: {self.feature_size}")
        self.finetune(self.backbone, n_classes)

    def finetune(self, model, n_classes):
        model_name = model.__class__.__name__
        if model_name.lower().startswith("resnet"):
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_size, n_classes)
            )

    def forward(self, x, get_features=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if get_features:
            return x
        x = self.classifier(x)
        return x


class ReIdModel(nn.Module):
    "Model based on LeNet for person re-identification"
    def __init__(self, n_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.BatchNorm2d(120),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 120, out_features=120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
