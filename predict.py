# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

classes_num = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pred_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

idx_to_class = {0: 'apple', 1: 'banana', 2: 'grape', 3: 'orange', 4: 'pear'}


# 定义公共层: conv + bn + relu
def CBR(in_channels, out_channels):
    cbr = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return cbr


# 定义VGG
class VGG(nn.Module):
    def __init__(self, block_nums):
        super(VGG, self).__init__()

        self.block1 = self._make_layers(in_channels=3, out_channels=64, block_num=block_nums[0])
        self.block2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.block3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.block4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.block5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, classes_num)
        )

    def _make_layers(self, in_channels, out_channels, block_num):
        blocks = []
        blocks.append(CBR(in_channels, out_channels))
        for i in range(1, block_num):
            blocks.append(CBR(out_channels, out_channels))
        blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        out = nn.functional.softmax(x, dim=1)

        return out


def VGG16():
    block_nums = [2, 2, 3, 3, 3]
    model = VGG(block_nums)
    return model


def predict(model, image_name):
    test_image = Image.open(image_name)
    # 由于训练的时候还有一个参数，是batch_size,而推理的时候没有，所以我们为了保持维度统一，就得使用.unsqueeze(0)来拓展维度
    test_image_tensor = pred_transform(test_image).unsqueeze(0)

    test_image_tensor = test_image_tensor.to(device)

    with torch.no_grad():
        model.eval()
        out = model(test_image_tensor)
        print(out)
        score, cls = torch.max(out, 1)
        print(score)

    return idx_to_class[cls.item()], score.item()


def main(model_path, image_name):
    model = VGG16().to(device)
    model.load_state_dict(torch.load(model_path))
    cls, score = predict(model, image_name)
    return cls, score


if __name__ == '__main__':
    model_name = "./model/vgg16.pth"
    image_path = "./data/val_data/grape/4.jpg"
    cls, score = main(model_name, image_path)
    print("Classify: %s, score: %s" % (cls, score))
