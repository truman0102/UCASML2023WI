import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification
from torchvision.models import resnet50


class ResNet50(nn.Module):
    def __init__(self, num_classes=10, pretrained=""):
        super(ResNet50, self).__init__()

        resnet = resnet50(weights=None)
        resnet.load_state_dict(torch.load(pretrained))
        self.model = nn.Sequential()
        for name, module in resnet.named_children():
            if name.split(".")[0] != "fc":
                self.model.add_module(name, module)
        del resnet
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256), nn.SELU(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def freeze(self):
        for name, param in self.named_parameters():
            if name.split(".")[0] == "model":
                param.requires_grad = False


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
            if name.split(".")[0] == "model":
                param.requires_grad = False


if __name__ == "__main__":
    # model = ViT(path="../model/ViT")
    # model.freeze()
    # print(model(torch.randn(2, 3, 224, 224)).shape)
    model = ResNet50(pretrained="../model/ResNet/resnet50-19c8e357.pth")
    model.freeze()
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print(model(torch.randn(2, 3, 224, 224)).shape)
