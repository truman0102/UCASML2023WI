import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification


class ViT(nn.Module):
    def __init__(self, num_classes=10, path=""):
        super(ViT, self).__init__()

        vit = ViTForImageClassification.from_pretrained(path)
        self.model = nn.Sequential()
        for name, module in vit.named_children():
            if name.split(".")[0] == "vit":
                self.model.add_module(name, module)
        del vit
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.SELU(), nn.Linear(256, num_classes)
        )  # 线性层，用于分类，可以修改

    def forward(self, x):
        x = self.model(x).last_hidden_state[:, 0, :]  # (B, 768)
        x = self.classifier(x)  # (B, 10)
        return x

    def freeze(self):
        for name, param in self.named_parameters():
            if name.split(".")[0] == "vit":
                param.requires_grad = False


if __name__ == "__main__":
    model = ViT(path="../model/ViT")
    model.freeze()
    print(model(torch.randn(2, 3, 224, 224)).shape)
