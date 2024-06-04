import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from models.MobileNetV1 import *
from models.MobileNetV2 import *
import argparse

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_transforms():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    
    return transform_train, transform


def load_data(batchsize, data_dir, transform_train, transform):
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=False, num_workers=8)
    
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=True, num_workers=8)
    
    return train_loader, test_loader


def get_validation_data(test_loader):
    valid_data_iter = iter(test_loader)
    valid_images, valid_labels = next(valid_data_iter)
    valid_size = valid_labels.size(0)
    return valid_images, valid_labels, valid_size


def write_data_file(epoch_list, loss_list, accu_list, filepath):
    optRecord = {
        "epoch": epoch_list,
        "train_loss": loss_list,
        "accuracy": accu_list}
    dfRecord = pd.DataFrame(optRecord)
    dfRecord.to_csv(filepath, index=False, encoding="utf_8_sig")


def train_model(model, train_loader, valid_images, valid_labels, valid_size, device, num_epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
    
    epoch_list, loss_list, accu_list = [], [], []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            optimizer.zero_grad()
            inputs, labels = data
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        with torch.no_grad():
            outputs_valid = model(valid_images.to(device))
        pred_labels = torch.max(outputs_valid, dim=1)[1]
        accuracy = torch.eq(pred_labels, valid_labels.to(device)).sum().item() / valid_size * 100
        print("Epoch {}: train loss={:.4f}, accuracy={:.2f}%".format(epoch, running_loss, accuracy))
        
        epoch_list.append(epoch)
        loss_list.append(running_loss)
        accu_list.append(accuracy)
    
    return epoch_list, loss_list, accu_list


def save_model(model, save_path):
    model_cpu = model.cpu()
    model_path = save_path + ".pth"
    torch.save(model.state_dict(), model_path)
    print("模型保存到: %s" % model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mobilenetv1', choices=['mobilenetv1', 'mobilenetv2'],
                        help='Specify which model to train: [mobilenetv1] [mobilenetv2]')
    args = parser.parse_args()

    device = get_device()
    data_dir = "./data"
    transform_train, transform = create_transforms()
    batchsize = 128

    # 加载数据
    train_loader, test_loader = load_data(batchsize, data_dir, transform_train, transform)
    valid_images, valid_labels, valid_size = get_validation_data(test_loader)
    # 选择模型
    if args.model == 'mobilenetv2':
        model = MobileNetV2(output_size=10).to(device)
    else:
        model = MobileNetV1(num_classes=10).to(device)    

    # 训练模型
    num_epochs = 100
    epoch_list, loss_list, accu_list = train_model(
        model, train_loader, valid_images, valid_labels, valid_size, device, num_epochs
    )

    # 保存训练记录
    write_data_file(epoch_list, loss_list, accu_list, "./result/data/mobilenetv2_training.csv")

    # 保存模型
    save_model(model, "./result/model/mobilenetv2")
