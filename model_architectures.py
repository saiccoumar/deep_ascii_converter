import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import os
import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
# from transformers import BertTokenizer
from PIL import Image
import math
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

### Neural Network

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x

### RESNET

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=95, grayscale=True):
        self.inplanes = 64
        in_dim = 1 if grayscale else 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

### CNN 

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x

class CBRD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBRD, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DBRD(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.5):
        super(DBRD, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_shape=(1, 10, 10), num_classes=95):
        height = input_shape[1]
        width = input_shape[2]
        super(CNN, self).__init__()
        self.gaussian_noise = GaussianNoise(0.3)
        self.cbrd1_1 = CBRD(input_shape[0], 64)
        self.cbrd1_2 = CBRD(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cbrd2_1 = CBRD(64, 128)
        self.cbrd2_2 = CBRD(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cbrd3_1 = CBRD(128, 256)
        self.cbrd3_2 = CBRD(256, 256)
        self.cbrd3_3 = CBRD(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        if height == 10:
            self.dbrd1 = DBRD(256 * 1, 4096)
        else: 
            self.dbrd1 = DBRD(256 * 64, 4096)
        self.dbrd2 = DBRD(4096, 4096)
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.gaussian_noise(x)

        x = self.cbrd1_1(x)
        x = self.cbrd1_2(x)
        x = self.pool1(x)

        x = self.cbrd2_1(x)
        x = self.cbrd2_2(x)
        x = self.pool2(x)

        x = self.cbrd3_1(x)
        x = self.cbrd3_2(x)
        x = self.cbrd3_3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dbrd1(x)
        x = self.dbrd2(x)
        x = self.fc(x)

        probas = F.softmax(x, dim=1)
        return x, probas

class EdgeDetectionEncoder(nn.Module):
    def __init__(self):
        super(EdgeDetectionEncoder, self).__init__()
        
        # Define the convolutional layers for edge detection
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Define the activation function
        self.relu = nn.ReLU()
        
        # Define pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Define upsample layer
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Define the transpose convolutional layers for decoding
        self.conv_transpose1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        # Encoder for edge detection
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        encoded = self.pool(x)  # Encoded representation
        
        # Decoder
        x = self.relu(self.conv_transpose1(encoded))
        x = self.relu(self.conv_transpose2(x))
        x = self.conv_transpose3(x)
        edge_map = torch.sigmoid(x)  # Edge map
        
        return encoded, edge_map


### MobileNetV2
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = (stride == 1) and (in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, grayscale=False):
        super(MobileNetV2, self).__init__()
        in_channels = 1 if grayscale else 3
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = [nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )]
        in_channels = 32
        for t, c, n, s in inverted_residual_setting:
            out_channels = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(in_channels, out_channels, stride, t))
                in_channels = out_channels

        self.features.append(nn.Sequential(
            nn.Conv2d(in_channels, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ))

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Autoencoder10(nn.Module):
    def __init__(self):
        super(Autoencoder10, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 10, 10)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # Output: (8, 10, 10)
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Flatten(),                                          # Output: (8 * 10 * 10 = 800)
            nn.Linear(8 * 10 * 10, 8),                          # Output: (128)
            nn.ReLU(),
            nn.Linear(8, 4),                                    # Output: (64)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 4),                                    # Output: (128)
            nn.ReLU(),
            nn.Linear(4, 8),                                    # Output: (128)
            nn.ReLU(),
            nn.Linear(8, 8 * 10 * 10),                          # Output: (800)
            nn.ReLU(),
            nn.Unflatten(1, (8, 10, 10)),                         # Output: (8, 10, 10)
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 10, 10)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # Output: (1, 10, 10)
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=1, stride=1)    
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    

class Autoencoder64(nn.Module):
    def __init__(self):
        super(Autoencoder64, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(64),                                  # Batch Normalization
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), # Output: (32, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(32),                                  # Batch Normalization
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), # Output: (16, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(16),                                  # Batch Normalization
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # Output: (8, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(8),                                   # Batch Normalization
            nn.Flatten(),                                         # Output: (8 * 64 * 64 = 32768)
            nn.Linear(8 * 64 * 64, 128),                         # Output: (128)
            nn.ReLU(),
            nn.Linear(128, 64),                                   # Output: (64)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),                                   # Output: (128)
            nn.ReLU(),
            nn.Linear(128, 128),                                   # Output: (128)
            nn.ReLU(),
            nn.Linear(128, 512),                                   # Output: (8 * 8 * 8)
            nn.ReLU(),
            nn.Unflatten(1, (8, 8, 8)),                            # Output: (8, 8, 8)
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (16, 16, 16)
            nn.BatchNorm2d(16),                                    # Batch Normalization
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (32, 32, 32)
            nn.BatchNorm2d(32),                                    # Batch Normalization
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (64, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),                    # Output: (1, 64, 64)
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=1, stride=1)                 # Output: (1, 64, 64) – no change to dimensions
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

