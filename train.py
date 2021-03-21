# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(7)
classes_num = 5


def data_handle():
    # 定义Transform
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # 获取图片地址
    image_root = args.data_path

    # 获取数据
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(image_root, "train_data"),
        transform=data_transform["train"]
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(image_root, "val_data"),
        transform=data_transform["val"]
    )

    kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    # 装载数据
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    return train_loader, val_loader


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
        out = self.classifier(x)

        return out


def VGG16():
    block_nums = [2, 2, 3, 3, 3]
    model = VGG(block_nums)
    return model


# model = VGG16()
# print(model)


# 训练
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    step_total = len(train_loader)
    for step, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, label)
        pred = output.argmax(dim=1)
        correct = pred.eq(label).sum().item()
        acc = 100. * correct / len(label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Train Epoch:[{}/{}], Step:[{}/{}], loss:{:.4f}, accuracy:{:.2f}%".format(
            epoch, args.epochs, step + 1, step_total, loss.item(), acc
        ))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            test_loss += nn.CrossEntropyLoss(reduction="sum")(output, label).item()
            # pred = output.argmax(dim=1, keepdim=True)
            pred = output.argmax(dim=1)
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print("Test set: loss:{:.4f}, accuracy:{:.2f}%".format(
        test_loss, acc
    ))


def main():
    train_loader, val_loader = data_handle()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = VGG16().to(device)
    if os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path))
        print(f"Loaded {args.save_path}!")
    else:
        print("No Param!")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader)

    torch.save(model.state_dict(), "./model/vgg16.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch VGG16")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./model/vgg16.pth"
    )
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))

    main()
