import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import numpy
import numpy as np
import pandas as pd
import json

import os
import math
import copy
import random

from efficientnet_pytorch import EfficientNet
import pytorch_lightning as pl

class Config:
    weights2Cal ={
        'apple': 0.52,
        'banana': 0.89,
        'bread': 3.15,
        'bun':2.23,
        'doughnut':4.34,
        'egg': 1.43,
        'fired_dough_twist': 24.16,
        'grape': 0.69,
        'lemon': 0.29,
        'litchi': 0.66,
        'mango': 0.60,
        'mooncake': 18.83,
        'orange': 0.63,
        'peach': 0.57,
        'pear': 0.39,
        'plum': 0.46,
        'qiwi': 0.61,
        'sachima': 21.45,
        'tomato': 0.27
    }
    
    classes = sorted(weights2Cal.keys())
    cls2idx = {}
    idx2cls = {}
    for idx in range(len(classes)):
        cls2idx[classes[idx]] = idx
        idx2cls[idx] = classes[idx]
    num_classes = len(classes)
    # Hyper Parameters
    IMAGE_SIZE = 256
    
def initialize_weights(layer):
    # Initialize weights using Kaiming init, better than Xavier.
    for module in layer.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
class ModelConfig:
    num_classes = 2
    drop_prob = 0.1
    act_type = 'relu'
    gate_attention = True
    attention_type = 'se'
    reduction = 16
    
    bam_dilate = 3
    groups = 16
    resnest = True
    
    bottleneck_type = 'resnext'
    bottleneck_reduction = 4
    
    decoder_attention = False
    use_FPN = False

class Mish(pl.LightningModule):
    def __init__(self):
        pass
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
def replace_mishes(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.SiLU, nn.ReLU)):
            setattr(model, name, Mish())
class Act(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.act_type = ModelConfig.act_type
        if self.act_type == 'relu':
            self.act = nn.ReLU(inplace = True)
        elif self.act_type == 'mish':
            self.act = Mish()
        else:
            self.act = nn.SiLU(inplace = True)
    def forward(self, x):
        return self.act(x)

class ConvBlock(pl.LightningModule):
    def __init__(self, in_features, out_features, kernel_size, padding, groups, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size = kernel_size, padding = padding, groups = groups, stride = stride, bias = False)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = Act()
        initialize_weights(self)
    def forward(self, x):
        return self.bn(self.act(self.conv(x)))
class SqueezeExcite(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Squeeze = nn.Linear(self.in_features, self.inner_features)
        self.act = Act()
        self.Excite = nn.Linear(self.inner_features, self.in_features)
    def forward(self, x):
        mean = torch.squeeze(self.global_avg(x))
        squeeze = self.act(self.Squeeze(mean))
        excite = torch.sigmoid(self.Excite(squeeze)).unsqueeze(-1).unsqueeze(-1) * x
        return excite
class ECASqueezeExcite(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.kernel_size = 5
        self.padding = 2
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv1d(1, 1, kernel_size = self.kernel_size, padding = self.padding, bias = False)
        initialize_weights(self)
    def forward(self, x):
        mean = torch.squeeze(self.global_avg(x), dim = -1).transpose(-1, -2) # (B, 1, C)
        conv = torch.sigmoid(self.conv(mean)).transpose(-1, -2).unsqueeze(-1) # (B, C, 1, 1)
        return conv * x
class SCSqueezeExcite(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Squeeze = nn.Linear(self.in_features, self.inner_features)
        self.act = Act()
        self.Excite = nn.Linear(self.inner_features, self.in_features)
        
        self.kernel_size = 3
        self.padding = 1
        
        self.Conv_Squeeze = nn.Conv2d(self.in_features, 1, kernel_size = self.kernel_size, padding = self.padding, bias = False)
        initialize_weights(self)
    def forward(self, x):
        mean = torch.squeeze(self.global_avg(x))
        squeeze = self.act(self.Squeeze(mean))
        excite = torch.sigmoid(self.Excite(squeeze)).unsqueeze(-1).unsqueeze(-1) * x
        
        conv_excite = torch.sigmoid(self.Conv_Squeeze(x)) * x
        excited = (excite + conv_excite) / 2
        return excited
class Attention(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.attention_type = ModelConfig.attention_type
        assert self.attention_type in ['eca', 'se', 'scse', 'none']
        if self.attention_type == 'eca':
            self.layer = ECASqueezeExcite()
        elif self.attention_type == 'se':
            self.layer = SqueezeExcite(self.in_features, self.inner_features)
        elif self.attention_type == 'scse':
            self.layer = SCSqueezeExcite(self.in_features, self.inner_features)
        else:
            self.layer = nn.Identity()
        self.gate_attention = ModelConfig.gate_attention
        if self.gate_attention:
            self.gamma = nn.Parameter(torch.zeros((1), device = self.device) - 10)
    def forward(self, x):
        excited = self.layer(x)
        if self.gate_attention:
            gamma = torch.sigmoid(self.gamma)
            return gamma * excited + (1 - gamma) * x
        return excited
class SwinTransformerAttention(pl.LightningModule):
    def __init__(self, length, in_features, inner_features, num_heads):
        super().__init__()
        self.length = length 
        self.in_features = in_features
        self.inner_features = inner_features
        self.num_heads = num_heads
    
        self.Keys = nn.Linear(self.in_features, self.inner_features * self.num_heads)
        self.Queries = nn.Linear(self.in_features, self.inner_features * self.num_heads)
        self.Values = nn.Linear(self.in_features, self.inner_features * self.num_heads)
        self.Linear = nn.Linear(self.inner_features * self.num_heads, self.in_features)
        
        
        self.pos_enc = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((self.num_heads, self.length, self.length), device = self.device)))
    def forward(self, x):
        B, L, _ = x.shape
        assert L == self.length
        K = self.Keys(x)
        V = self.Values(x)
        Q = self.Queries(x) # (B, L, Heads * Inner_features)
        
        K = K.view(B, L, self.num_heads, self.inner_features)
        V = V.view(B, L, self.num_heads, self.inner_features)
        Q = Q.view(B, L, self.num_heads, self.inner_features)
        
        K = K.transpose(1, 2).view(-1, L, self.inner_features)
        V = V.transpose(1, 2).view(-1, L, self.inner_features)
        Q = Q.transpose(1, 2).view(-1, L, self.inner_features) # (BH, L, inner_features)
        
        pos_enc = torch.repeat_interleave(self.pos_enc, B, dim = 0)
        att_mat = F.softmax(Q @ K.transpose(1, 2) / math.sqrt(self.inner_features) + pos_enc, dim = -1) # (BH, L, L)
        att_scores = att_mat @ V # (BH, L, I)
        
        att_scores = att_scores.view(B, self.num_heads, L, self.inner_features)
        att_scores = att_scores.transpose(1, 2).view(B, L, -1)
        return self.Linear(att_scores)
class SwinTransformerEncoder(pl.LightningModule):
    def window(self, x):
        # X: Tensor(B, C, H, W) 
        B, C, H, W = x.shape
        windowed = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        windowed = x.permute(0, 2, 4, 3, 5, 1)
        windowed = torch.view(-1, self.window_size * self.window_size, C)
        return windowed
    def unwindow(self, x):
        B, _, C = x.shape
        B = B // (self.H // self.window_size) // (self.W // self.window_size)
        
        unwindow = x.view(B, self.H // self.window_size, self.W // self.window_size, self.window_size, self.window_size, C)
        unwindow = unwindow.permute(0, 5, 1, 3, 2, 4) 
        unwindow = unwindow.view(B, C, self.H, self.W)
        return unwindow
        
    def shift(self, x):
        return torch.roll(x, (-(self.window_size // 2), -(self.window_size // 2)))
    def unshift(self, x):
        return torch.roll(x, (self.window_size // 2, self.window_size // 2))
    def __init__(self, H, W, in_features, inner_features, out_features, num_heads, window_size = 4):
        super().__init__()
        self.H = H
        self.W = W
        self.length = self.H * self.W
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.pos_enc = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1, self.in_features, self.H, self.W), device = self.device)))
        self.norm1 = nn.LayerNorm((self.in_features, self.H, self.W))
        self.att1 = SwinTransformerAttention(self.window_size ** 2, self.in_features, self.inner_features, self.num_heads)
        self.norm2 = nn.LayerNorm((self.length, self.in_features))
        self.linear2 = nn.Linear(self.in_features, self.in_features)
        
        self.norm3 = nn.LayerNorm((self.in_features, self.H, self.W))
        self.att3 = SwinTransformerAttention(self.window_size ** 2, self.in_features, self.inner_features, self.num_heads)
        self.norm4 = nn.LayerNorm((self.length, self.in_features))
        self.linear4 = nn.Linear(self.in_features, self.out_features)
    def forward(self, x):
        # X: Tensor(B, C, H, W)
        x = x + self.pos_enc # (B, C, H, W)
        norm1 = self.norm1(x) # (B, C, H, W)
        windowed = self.window(norm1) # (-1, window_size ** 2, C)
        att1 = self.att1(windowed)
        # Unwindow
        unwindowed = self.unwindow(att1) + x# (B, C, H, W)
        unwindowed = unwindowed.view(B, C, self.W * self.H).transpose(1, 2) # (B, HW, C)
        norm2 = self.norm2(unwindowed) # (B, HW, C)
        linear2 = self.linear2(norm2) + unwindowed
    
        linear2 = linear2.transpose(1, 2).view(B, C, self.H, self.W)
        norm3 = self.norm3(linear2) # (B, C, H, W)
        windowed = self.window(norm3) 
        windowed = self.shift(windowed)
        
        att3 = self.att3(windowed)
        att3 = self.unshift(att3)
        
        unwindowed = self.unwindow(att3) + linear2 # (b, C, H, W)
        
        unwindowed = unwindowed.view(B, C, self.H * self.W).transpose(1, 2) # (B, L, C)
        norm4 = self.norm4(unwindowed) 
        linear4 = self.linear4(norm4) + unwindowed# (B, L, C) 
    
        output = linear4.transpose(1, 2).view(B, C, self.H, self.W)
        return output
        
        
        
class SplitAttention(pl.LightningModule):
    # split attention like in ResNest.
    def __init__(self, in_features, inner_features, cardinality):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.cardinality = cardinality
        assert self.in_features % self.cardinality == 0
        
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = ConvBlock(self.in_features, self.inner_features * self.cardinality, 1, 0, 1, 1)
        self.conv2 = nn.Conv2d(self.inner_features * self.cardinality, self.in_features * self.cardinality, kernel_size = 1, groups = self.cardinality, bias = False)
        initialize_weights(self.conv2)
        
        self.gate_attention = ModelConfig.gate_attention
        if self.gate_attention:
            self.gamma = nn.Parameter(torch.tensor(-10.0, device = self.device))
        
    def forward(self, x):
        # X: Tensor(B, C, H, W, Cardinality)
        B, C, H, W, Cardinality = x.shape
        summed = torch.sum(x, dim = -1)
        pooled = self.AvgPool(summed) # (B, C * Cardinality, 1, 1)
        
        conv1 = self.conv1(pooled)
        conv2 = self.conv2(conv1) # (B, inner_features * cardinality, 1, 1)  
        conv2 = conv2.view(B, self.in_features, self.cardinality)# (B, inner_features, cardinality, 1, 1)
        conv2 = F.softmax(conv2.unsqueeze(2).unsqueeze(2), dim = -1) # (B, inner_features, 1, 1, cardinality)
        
        excited = x * conv2
        if self.gate_attention:
            gamma = torch.sigmoid(self.gamma)
            return gamma * excited + (1 - gamma) * x
        return excited
class GhostConv(pl.LightningModule):
    def __init__(self, in_features, out_features, kernel_size, padding):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.inner_features = self.out_features // 2
        
        self.squeeze = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.cheap = ConvBlock(self.inner_features, self.inner_features, self.kernel_size, self.padding, self.inner_features, 1)
    def forward(self, x):
        squeeze = self.squeeze(x)
        cheap = self.cheap(squeeze) # (B, C, H, W)
        
        return torch.cat([squeeze, cheap], dim = 1) 

class AstrousConvBlock(pl.LightningModule):
    def __init__(self, in_features, out_features, kernel_size, padding, groups, stride, dilation):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size = kernel_size, padding = padding, groups = groups, stride = stride, dilation = dilation, bias = False)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = Act()
        initialize_weights(self)
    def forward(self, x):
        return self.bn(self.act(self.conv(x)))
        
class BAM(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Squeeze = nn.Linear(self.in_features, self.inner_features)
        self.Act = Act()
        self.Excite = nn.Linear(self.inner_features, self.in_features)
        self.bam_dilate = ModelConfig.bam_dilate
        
        self.ConvSqueeze = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.DA = AstrousConvBlock(self.inner_features, self.inner_features, 3, self.bam_dilate, self.inner_features, 1, self.bam_dilate)
        self.ConvExcite = nn.Conv2d(self.inner_features, 1, kernel_size = 1, bias = False)
        initialize_weights(self.ConvExcite)
        
        self.gate_attention = ModelConfig.gate_attention
        if self.gate_attention:
            self.gamma = nn.Parameter(torch.tensor(-10.0, device = self.device))
    def forward(self, x):
        pooled = torch.squeeze(self.global_avg(x))
        squeeze = self.Act(self.Squeeze(pooled))
        excite = torch.sigmoid(self.Excite(squeeze).unsqueeze(-1).unsqueeze(-1)) * x
        
        squeeze = self.ConvSqueeze(x)
        DA = self.DA(squeeze)
        convExcite = self.ConvExcite(DA) * x
        
        excited = (convExcite + excite) / 2
        if self.gate_attention:
            gamma = torch.sigmoid(self.gamma)
            return gamma * excited + (1 - gamma) * x
        return excited
    
# -----------------BottleNeck Blocks -------------------#
class GhostBottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.reduction = ModelConfig.reduction
        
        self.ghost1 = GhostConv(self.in_features, self.inner_features, 1, 0)
        self.att = Attention(self.inner_features, self.inner_features // self.reduction)
        self.ghost2 = GhostConv(self.inner_features, self.in_features, 3, 1)
        
        self.gamma = nn.Parameter(torch.tensor(-10.0, device = self.device))
    def forward(self, x):
        ghost1 = self.ghost1(x)
        att1 = self.att(ghost1)
        ghost2 = self.ghost2(att1)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * ghost2 + (1 - gamma) * x
class GhostDownSampler(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        self.stride = stride
        self.reduction = ModelConfig.reduction
        
        self.AvgPool = nn.AvgPool2d(kernel_size = 3, padding = 1, stride = self.stride)
        self.ghostPool = GhostConv(self.in_features, self.out_features, 1, 0)
        
        self.Ghost1 = GhostConv(self.in_features, self.inner_features, 1, 0)
        self.DW = ConvBlock(self.inner_features, self.inner_features, 3, 1, self.inner_features, 1)
        self.att = Attention(self.inner_features, self.inner_features // self.reduction)
        self.Ghost2 = GhostConv(self.inner_features, self.out_features, 1, 0)
        
        self.gamma = nn.Parameter(torch.tensor(-10.0, device = self.device))
    def forward(self, x):
        pooled = self.AvgPool(x)
        ghostPool = self.ghostPoool(pooled)
        
        ghost1 = self.Ghost1(x)
        dw = self.DW(ghost1)
        att = self.att(dw)
        ghost2 = self.Ghost2(att)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * ghostPool + (1 - gamma) * ghost2
class ResNextBottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features, cardinality, resnest = False):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.cardinality = cardinality
        self.resnest = resnest
        self.reduction = ModelConfig.reduction
        
        self.Squeeze = ConvBlock(self.in_features, self.inner_features * self.cardinality, 1, 0, 1, 1)
        self.Process = ConvBlock(self.inner_features * self.cardinality, self.inner_features * self.cardinality, 3, 1, self.cardinality, 1)
        if self.resnest:
            #  Split Attention
            self.att = SplitAttention(self.inner_features, self.inner_features // self.reduction, self.cardinality)
        else:
            self.att = Attention(self.inner_features * self.cardinality, self.inner_features * self.cardinality // self.reduction)
        self.Expand = ConvBlock(self.inner_features * self.cardinality, self.in_features, 1, 0, 1, 1)
        
        self.gamma = nn.Parameter(torch.tensor(-10.0, device = self.device))
    def forward(self, x):
        squeeze = self.Squeeze(x)
        process = self.Process(squeeze)
        if self.resnest:
            B, C, H, W = process.shape
            process = process.view(B, C // self.cardinality, self.cardinality, H, W)
            process = process.transpose(2, 3).transpose(3, 4)
            
            att = self.att(process)
            att = att.transpose(3, 4).transpose(2, 3).view(B, -1, H, W)
        else:
            att = self.att(process)
        expand = self.Expand(att)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * expand + (1 - gamma) * x
class ResNextDownSampler(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride, cardinality, resnest = False):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        self.stride = stride
        self.cardinality = cardinality
        self.resnest = resnest 
        self.reduction = ModelConfig.reduction
        
        self.avgPool = nn.AvgPool2d(kernel_size = 3, padding = 1, stride = self.stride)
        self.ConvPool = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        
        self.Squeeze = ConvBlock(self.in_features, self.inner_features * self.cardinality, 1, 0, 1, 1)
        self.Process = ConvBlock(self.cardinality * self.inner_features, self.inner_features * self.cardinality, 3, 1, self.cardinality, 1)
        if self.resnest:
            self.att = SplitAttention(self.inner_features, self.inner_features // self.reduction, self.cardinality)
        else:
            self.att = Attention(self.inner_features * self.cardinality, self.inner_features * self.cardinality // self.reduction)
        self.Expand = ConvBlock(self.inner_features * self.cardinality, self.in_features, 1, 0, 1, 1)
    
        self.gamma = nn.Parameter(torch.tensor(-10.0, device = self.device))
    def forward(self, x):
        pooled = self.avgPool(x)
        convPool = self.ConvPool(pooled)
        
        squeeze = self.Squeeze(x)
        process = self.Process(squeeze)
        if self.resnest:
            B, C, H, W = process.shape
            process = process.view(B, C // self.cardinality, self.cardinality, H, W)
            process = process.transpose(2, 3).transpose(3, 4)
            att = self.att(process)
            att = att.transpose(3, 4).transpose(2, 3).view(B, -1, H, W) 
        else:
            att = self.att(process)
        expand = self.Expand(att)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * expand + (1 - gamma) * convPool
class InverseBottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.reduction = ModelConfig.reduction
        
        self.Expand = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.DW = ConvBlock(self.inner_features, self.inner_features, 3, 1, self.inner_features, 1)
        self.Attention = Attention(self.inner_features, self.inner_features // self.reduction)
        self.Squeeze = ConvBlock(self.inner_features, self.in_features, 1, 0, 1, 1)
        
        self.gamma = nn.Parameter(torch.tensor(-10.0, device = self.device))
    def forward(self, x):
        expand = self.Expand(x)
        dw = self.DW(expand)
        att = self.Attention(dw)
        squeeze = self.Squeeze(att)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * squeeze + (1 - gamma) * x
class InverseDownSampler(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        self.stride = stride 
        self.reduction = ModelConfig.reduction
    
        self.AvgPool = nn.AvgPool2d(kernel_size = 3, padding = 1, stride = self.stride)
        self.ConvPool = ConvBlock(self.in_features, self.out_features, 1, 0, 1, 1)
    
        self.Expand = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.DW = ConvBlock(self.inner_features, self.inner_features, 3, 1, self.inner_features, 1)
        self.Attention = Attention(self.inner_features, self.inner_features // self.reduction)
        self.Squeeze = ConvBlock(self.inner_features, self.out_features, 1, 0, 1, 1)
        
        self.gamma = nn.Parameter(torch.tensor(-10.0, device = self.device))
    def forward(self, x):
        pooled = self.AvgPool(x)
        convPool = self.ConvPool(pooled)
        
        expand = self.Expand(x)
        dw = self.DW(expand)
        att = self.Attention(dw)
        squeeze = self.Squeeze(att)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * squeeze + (1 - gamma) * convPool
class ChooseBottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.bottleneck_type = ModelConfig.bottleneck_type
        
        if self.bottleneck_type == 'ghost':
            self.layer = GhostBottleNeck(self.in_features, self.inner_features)
        elif self.bottleneck_type == 'inverse':
            self.layer = InverseBottleNeck(self.in_features, self.inner_features)
        else:
            self.cardinality = ModelConfig.groups
            self.resnest = ModelConfig.resnest
            self.layer = ResNextBottleNeck(self.in_features, self.inner_features, self.cardinality, resnest = self.resnest)
    def forward(self, x):
        return self.layer(x)
        
class ChooseDownsampler(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        self.stride = stride
        self.bottleneck_type = ModelConfig.bottleneck_type
    
        if self.bottleneck_type == 'ghost':
            self.layer = GhostDownSampler(self.in_features, self.inner_features, self.out_features, self.stride)
        elif self.bottleneck_type == 'inverse':
            self.layer = InverseDownSampler(self.in_features, self.inner_features, self.out_features, self.stride)
        else:
            self.cardinality = ModelConfig.groups
            self.resnest = ModelConfig.resnest
            self.layer = ResNextDownSampler(self.in_features, self.inner_features, self.out_features, self.stride, self.cardinality, resnest = self.resnest)
    def forward(self, x):
        return self.layer(x)
class ASPP(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride):
        '''
        ASPP Module:
        - 1x1 Conv
        - 3x3 Conv, Astrous 2
        - 3x3 Conv, Astrous 3
        - 3x3,Conv, Astrous 5
        - 3x3 Pool
        '''
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        self.stride = stride 
        
        self.conv1 = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, self.stride)
        self.conv2 = AstrousConvBlock(self.in_features, self.inner_features, 3, 2, 1, self.stride, 2)
        self.conv3 = AstrousConvBlock(self.in_features, self.inner_features, 3, 3, 1, self.stride, 3)
        self.conv4 = AstrousConvBlock(self.in_features, self.inner_features, 3, 5, 1, self.stride, 5)
        self.conv5 = nn.AvgPool2d(kernel_size = 3, padding = 1, stride = self.stride)
        
        self.ConvProj = ConvBlock(self.inner_features * 4 + self.in_features, self.out_features, 1, 0, 1, 1)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(x)
        
        concat = torch.cat([conv1, conv2, conv3, conv4, conv5], dim = 1)
        return self.ConvProj(concat)
        
class DualChannelBlock(pl.LightningModule):
    # DC-Unet Block, However only 1 channels to reduce model parameters
    def __init__(self, in_features, inner_features, out_features, stride):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        self.stride = stride
        
        self.Conv1 = ConvBlock(self.in_features, self.inner_features, 3, 1, 1, 1)
        self.Conv2 = ConvBlock(self.inner_features, self.inner_features, 3, 1, self.inner_features, 1)
        
        self.Conv3 = ConvBlock(self.inner_features * 2 + self.in_features, self.out_features, 3, 1, 1, self.stride)
    def forward(self, x):
        conv1 = self.Conv1(x)
        conv2 = self.Conv2(conv1)
        cat = torch.cat([x, conv1, conv2], dim = 1) 
        conv3 = self.Conv3(cat)
        return conv3
class EncoderAlpha(pl.LightningModule):
    def freeze(self, layers):
        for layer in layers:
            for parameter in layer.parameters():
                parameter.requires_grad = False
    def freeze_cls(self):
        self.freeze([self.conv1, self.bn1, self.block0, self.block1, self.block2, self.block3, self.block4, self.block5])
    def __init__(self):
        super().__init__()
        self.model_name = 'efficientnet-b0'
        self.model = EfficientNet.from_name(self.model_name)
        
        # Extract Layers
        self.conv1 = self.model._conv_stem # 32, 128
        self.bn1 = self.model._bn0
        self.act1 = self.model._swish
        
        self.block0 = self.model._blocks[0] # 16, 128
        self.block1 = nn.Sequential(*self.model._blocks[1:3]) # 24, 64 
        self.block2 = nn.Sequential(*self.model._blocks[3: 5]) # 40, 32
        self.block3 = nn.Sequential(*self.model._blocks[5: 8]) # 80, 16
        self.block4 = nn.Sequential(*self.model._blocks[8: 11]) # 112, 16
        self.block5 = nn.Sequential(*self.model._blocks[11: 15]) # 192, 8
        self.block6 = self.model._blocks[15] # 320, 8
        
        # Freeze Initial Layers
        self.freeze([self.conv1, self.bn1, self.block0, self.block1])
    
        self.reduction = ModelConfig.bottleneck_reduction
        
        self.Dropout6 = nn.Dropout2d(ModelConfig.drop_prob)
        self.Attention6 = BAM(320, 320 // self.reduction)
        
        self.block7 = nn.Sequential(*[
            ChooseBottleNeck(320, 320 // self.reduction) for i in range(2)
        ] + [
            DualChannelBlock(320, 320 // self.reduction, 320, 1),
            ASPP(320, 320 // self.reduction, 512, 2)
        ])
        
        self.Dropout7 = nn.Dropout2d(ModelConfig.drop_prob)
        self.Attention7 = BAM(512, 512 // self.reduction)
        
        # large encoder, small decoder(Since it's being stripped away)
    def forward_cls(self, x):
        features0 = self.bn1(self.act1(self.conv1(x)))
        block0 = self.block0(features0)
        block1 = self.block1(block0)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        
        block6 = self.Attention6(self.Dropout6(block6))
        block7 = self.block7(block6)
        block7 = self.Attention7(self.Dropout7(block7))
        return block7
class FeaturesAlpha(pl.LightningModule):
    def freeze(self, layers):
        for layer in layers:
            for parameter in layer.parameters():
                parameter.requires_grad = False
    def __init__(self):
        super().__init__()
        self.model = EncoderAlpha()
        # Freeze Entire Encoder
        self.model.freeze_cls()
        # One Last Layer, seperated by Task.
        self.in_dim = 512
        self.out_dim = 1024
        self.reduction = ModelConfig.reduction
        self.cls_att8 = nn.Identity()#BAM(self.in_dim, self.in_dim // self.reduction)
        self.reg_att8 = nn.Identity()#BAM(self.in_dim, self.in_dim // self.reduction)
        self.vol_att8 = nn.Identity()#BAM(self.in_dim, self.in_dim // self.reduction)
        self.drop_att8 = nn.Identity()#nn.Dropout2d(0.1)
        
        self.proj_cls = ConvBlock(self.in_dim, self.out_dim, 1, 0, 1, 1)
        self.proj_reg = ConvBlock(self.in_dim, self.out_dim, 1, 0, 1, 1)
        self.proj_vol = ConvBlock(self.in_dim, self.out_dim, 1, 0, 1, 1)
        
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        features = self.model.forward_cls(x)
    
        cls_att8 = self.cls_att8(self.drop_att8(features))
        reg_att8 = self.reg_att8(self.drop_att8(features))
        vol_att8 = self.vol_att8(self.drop_att8(features))
        
        proj_cls = self.proj_cls(cls_att8)
        proj_reg = self.proj_reg(reg_att8)
        proj_vol = self.proj_vol(vol_att8)
        
        pooled_cls = torch.squeeze(self.global_avg(proj_cls))
        pooled_reg = torch.squeeze(self.global_avg(proj_reg))
        pooled_vol = torch.squeeze(self.global_avg(proj_vol))
        
        return pooled_cls, pooled_reg, pooled_vol
class CrossStitchUnit(pl.LightningModule):
    # Merges Features from Both Branches to better MultiTask Learn
    def __init__(self):
        super().__init__()
        self.alpha1_1 = nn.Parameter(torch.tensor(1., device = self.device))
        self.alpha1_2 = nn.Parameter(torch.tensor(0., device = self.device))
        self.alpha1_3 = nn.Parameter(torch.tensor(0., device = self.device))
        
        self.alpha2_1 = nn.Parameter(torch.tensor(0., device = self.device))
        self.alpha2_2 = nn.Parameter(torch.tensor(1., device = self.device))
        self.alpha2_3 = nn.Parameter(torch.tensor(0., device = self.device))
        
        self.alpha3_1 = nn.Parameter(torch.tensor(0., device = self.device))
        self.alpha3_2 = nn.Parameter(torch.tensor(0., device = self.device))
        self.alpha3_3 = nn.Parameter(torch.tensor(1., device = self.device))
        
    def forward(self, CLS, REG, VOL):
        alpha1_1 = torch.sigmoid(self.alpha1_1)
        alpha1_2 = torch.sigmoid(self.alpha1_2)
        alpha1_3 = torch.sigmoid(self.alpha1_3)
        
        alpha2_1 = torch.sigmoid(self.alpha2_1)
        alpha2_2 = torch.sigmoid(self.alpha2_2)
        alpha2_3 = torch.sigmoid(self.alpha2_3)
        
        alpha3_1 = torch.sigmoid(self.alpha3_1)
        alpha3_2 = torch.sigmoid(self.alpha3_2)
        alpha3_3 = torch.sigmoid(self.alpha3_3)
        
        new_cls = alpha1_1 * CLS + alpha1_2 * REG + alpha1_3 * VOL
        new_reg = alpha2_1 * CLS + alpha2_2 * REG + alpha2_3 * VOL
        new_vol = alpha3_1 * CLS + alpha3_2 * REG + alpha3_3 * VOL
        
        return new_cls, new_reg, new_vol
class LinBNReLU(pl.LightningModule):
    def __init__(self, in_features, out_features, drop_prob = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop_prob = drop_prob
        self.layer = nn.Sequential(*[
            nn.Linear(self.in_features, self.out_features),
            nn.ReLU(inplace = True),
            nn.BatchNorm1d(self.out_features),
            nn.Dropout(self.drop_prob)
        ])
    def forward(self, x):
        return self.layer(x)
class BaseLineHeadAlpha(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.in_dim = 1024
        self.num_classes = Config.num_classes
        self.fc_cls = nn.Linear(self.in_dim, self.num_classes)
        self.fc_reg = nn.Linear(self.in_dim, 1)
        self.fc_vol = nn.Linear(self.in_dim, 1)
    def forward(self, CLS, REG, VOL):
        return self.fc_cls(CLS), torch.squeeze(self.fc_reg(REG)), torch.squeeze(self.fc_vol(VOL))
class CrossStitchAlpha(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.in_dim = 1024
        self.num_classes = Config.num_classes
        self.cross_stitch = CrossStitchUnit()
        
        self.fc_cls1 = LinBNReLU(self.in_dim, 512)
        self.fc_reg1 = LinBNReLU(self.in_dim, 512)
        self.fc_vol1 = LinBNReLU(self.in_dim, 512)
    
        self.cross_stitch2 = CrossStitchUnit()
        
        self.fc_cls2 = nn.Linear(512, self.num_classes)
        self.fc_reg2 = nn.Linear(512, 1) # Predict Weight and Class of the item.
        self.fc_vol2 = nn.Linear(512, 1)
    def forward(self, classification, regression, volume):
        classification, regression, volume = self.cross_stitch(classification, regression, volume)
        
        if len(classification.shape) < 2:
            classification = classification.unsqueeze(0)
        if len(regression.shape) < 2:
            regression = regression.unsqueeze(0)
        if len(volume.shape) < 2:
            volume = volume.unsqueeze(0)
            
        fc_cls1 = self.fc_cls1(classification)
        fc_reg1 = self.fc_reg1(regression)
        fc_vol1 = self.fc_vol1(volume)
        
        fc_cls1, fc_reg1, fc_vol1 = self.cross_stitch2(fc_cls1, fc_reg1, fc_vol1)
        
        fc_cls2 = self.fc_cls2(fc_cls1)
        fc_reg2 = torch.squeeze(self.fc_reg2(fc_reg1))
        fc_vol2 = torch.squeeze(self.fc_vol2(fc_vol1))
        return fc_cls2, fc_reg2, fc_vol2
class ModelAlpha(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.featureExtractor = FeaturesAlpha()
        self.crossStitch = CrossStitchAlpha()#BaseLineHeadAlpha()
    def forward(self, x):
        features= self.featureExtractor(x)
        return self.crossStitch(*features)
class WeightWatcherAlpha(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weights2Cal ={
            'apple': 0.52,
            'banana': 0.89,
            'bread': 3.15,
            'bun':2.23,
            'doughnut':4.34,
            'egg': 1.43,
            'fired_dough_twist': 24.16,
            'grape': 0.69,
            'lemon': 0.29,
            'litchi': 0.66,
            'mango': 0.60,
            'mooncake': 18.83,
            'orange': 0.63,
            'peach': 0.57,
            'pear': 0.39,
            'plum': 0.46,
            'qiwi': 0.61,
            'sachima': 21.45,
            'tomato': 0.27
        }
        
        self.cls2idx = {list(self.weights2Cal.keys())[idx]: idx for idx in range(len(self.weights2Cal))}
        self.idx2cls = {idx: list(self.weights2Cal.keys())[idx] for idx in range(len(self.weights2Cal))}
        self.weight_path = './deep_learning/f1.pth'
        self.model = ModelAlpha()
        self.load_state_dict(torch.load(self.weight_path, map_location = self.device))
    def forward(self, x):
        self.eval()
        with torch.no_grad():
            cls_idx, weights, volume = self.model(x)
            cls_idx = F.softmax(cls_idx, dim = -1)
            _, cls_idx = torch.max(cls_idx, dim = -1)
            cls_idx = cls_idx.item()
            weights = weights.item()
            volume = volume.item()

            class_val = self.idx2cls[cls_idx]
            
            calories = self.weights2Cal[class_val] * weights
            if class_val == 'fired_dough_twist':
                # Typo in dataset, simple correction
                class_val = 'fried_dough_twist'
            return class_val, weights, volume, calories
def load_model():
    model = WeightWatcherAlpha()
    return model
MODEL = load_model()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225] # (3,)
IMAGENET = [np.array(mean), np.array(std)]
def normalize(image):
    image = image - IMAGENET[0]
    image = image / IMAGENET[1]
    return image
def process_image(image):
    image = np.array(image).astype(np.float32)
    image = image / 255.0

    image = torch.tensor(normalize(image)).unsqueeze(0)
    image = image.transpose(-1, -2).transpose(-2, -3)
    cls_val, weight, volume, calories = MODEL(image.float())
    return cls_val, weight, volume, calories
    