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
from transformers import BertTokenizer
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
        self.dbrd1 = DBRD(256 * 1 * 1, 4096)
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

class ImageTextDataset(Dataset):
    def __init__(self, image_dir, text_dir, transform=None, tokenizer=None, max_length=50):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.text_files = [f for f in os.listdir(text_dir) if os.path.isfile(os.path.join(text_dir, f))]
        
        assert len(self.image_files) == len(self.text_files), "Mismatch between image and text files count"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        text_file = image_file.replace('.png', '.txt')  # Assuming images are in .jpg and texts in .txt
        
        image_path = os.path.join(self.image_dir, image_file)
        text_path = os.path.join(self.text_dir, text_file)
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        with open(text_path, 'r') as file:
            text = file.read().strip()

        # Tokenize and encode the text
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return image, input_ids, attention_mask
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # print(self.device)
        if mask is not None:
            mask = mask.to(self.device)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, device):
        super(EncoderLayer, self).__init__()
        self.device = device
        self.self_attn = MultiHeadAttention(d_model, num_heads, device = device)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, device):
        super(DecoderLayer, self).__init__()
        self.device = device
        self.self_attn = MultiHeadAttention(d_model, num_heads, device = device)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, device = device)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
class CustomResNet(nn.Module):
    def __init__(self, d_model):
        super(CustomResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Modify the first convolutional layer to accept 1 channel input
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False),
            *list(resnet.children())[1:-2]  # Exclude the first convolutional layer from the original ResNet
        )
        self.batchnorm = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.adaptive_avg_pool(x)
        x = self.flatten(x)
        return x
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device):
        super(Transformer, self).__init__()

        # self.visual_backbone = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(d_model),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten()
        # )
        self.visual_backbone = CustomResNet(d_model)
        
        self.device = device
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, device=device) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, device=device) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = torch.ones(src.size(0), 1, 1, 1).bool()  # No mask for images
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device=self.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src_img, tgt):
        src_mask, tgt_mask = self.generate_mask(src_img, tgt)
        src_mask.to(self.device)
        tgt_mask.to(self.device)

        src_embedded = self.visual_backbone(src_img).unsqueeze(1)  # Adding sequence dimension
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale with 1 channel
    transforms.Resize((256, 256)),                # Resize to 256x256 pixels
    transforms.ToTensor(),                        # Convert to tensor
    transforms.Normalize(mean=[0.485], std=[0.229]) # Normalize with mean and std for 1 channel
    # transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]) # Normalize with mean and std for 1 channel
])


# Text tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
