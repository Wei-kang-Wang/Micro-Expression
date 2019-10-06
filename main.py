import sys

sys.path.append('../')
# 添加路径到python默认搜索路径中，是个list，此处添加的是上两级目录

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
from util.DatasetDefinition import AffectNet_Twoviews
from util.ModelDefinition import MyModel,Discriminator_high,Discriminator_mid
from util.HelperFunction import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#只使用第0块GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
debug_mode = False
os.makedirs('save', exist_ok=True)
# 创建save目录

csv_dir = Path('/data/csv')
face_dir = Path('/data/AffectNet Database/face_rgb_256')
micro_dir = Path('/data/CASME')

df_train = pd.read_csv(csv_dir / 'training.csv') # 路径不能含有中文
df_valid = pd.read_csv(csv_dir / 'validation.csv')
df_micro = pd.read_csv(csv_dir / 'micro_expression.csv')

#train_set = AffectNet_Twoviews(df_train, 'train', img_dir,occ_dir)
#valid_set = AffectNet_Twoviews(df_valid, 'valid', img_dir,occ_dir)
# There two are training and validation sets, whose function is imported from DatasetDefinition.py
# train and valid sets are all three-triplet sets: (clear image, occlusion image, label)

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
# resnet34是pytorch自带的类

#model_clean.load_state_dict( torch.load('/home/xiabin/AffectNet/pretrained/clean_58.91.pkl') )
#model_occ.load_state_dict( torch.load('/home/xiabin/AffectNet/pretrained/occ_56.28.pkl') )
# 加载预训练模型

model_fit(model_occ, model_clean, model_D1,model_D2, 2e-6, lrD1,lrD2,8, data, train_option, device, print_interval=80)
model_fit(model_occ, model_clean, model_D1,model_D2, 1e-6, lrD1,lrD2,2, data, train_option, device, print_interval=80)