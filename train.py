import sys

sys.path.append('../')

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import *
from datasets import AffectNet_Twoviews
from models import MyModel,Discriminator_high,Discriminator_mid
from utils import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
debug_mode = False
os.makedirs('save', exist_ok=True)

csv_dir = Path('/data/csv')
face_dir = Path('/data/AffectNet Database/face_rgb_256')
micro_dir = Path('/data/CASME')

df_train = pd.read_csv(csv_dir / 'training.csv')
df_valid = pd.read_csv(csv_dir / 'validation.csv')
df_micro = pd.read_csv(csv_dir / 'micro_expression.csv')


train_set = Triple_let(df_train, df_micro, 'train', face_dir, micro_dir)
valid_set = Triple_let(df_valid, df_micro, 'valid', face_dir, micro_dir)

batch_size = 32
train_loader = DataLoader(train_set, batch_size, shuffle=True,num_workers=8)
valid_loader = DataLoader(valid_set, batch_size, shuffle=False,num_workers=8)
# 按batch_size将原数据集分割并放入dataloader里，shuffle表示打乱，num_workers表示子进程数目

data = {
    'train': train_loader,
    'valid': valid_loader
}

train_option = {
    'criterion': nn.CrossEntropyLoss(reduction="none").to(device),
    'criterion_cpu': nn.CrossEntropyLoss(),
    'opt_class': optim.Adam,
    'weight_decay': 1e-4,
    'C2': 1,
    'C3':1e-3,
    'C4':1e-3,
    'C5':1e-3,
    'C6': 1e-4
}

model_D1 = Discriminator_mid().to(device)
model_D2 = Discriminator_high().to(device)


lrD1 = 1e-5
lrD2 = 1e-5

model_clean = MyModel(resnet34).to(device)
model_occ = MyModel(resnet34).to(device)


model_fit(model_occ, model_clean, model_D1,model_D2, 2e-6, lrD1,lrD2,8, data, train_option, device, print_interval=80)
model_fit(model_occ, model_clean, model_D1,model_D2, 1e-6, lrD1,lrD2,2, data, train_option, device, print_interval=80)