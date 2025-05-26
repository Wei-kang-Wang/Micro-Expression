import cv2
import random
import numpy as np
from PIL import Image
from pathlib import Path
import os
from torch.utils.data import Dataset
from torchvision import transforms





def is_grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    d1 = img[:, :, 1].max() - img[:, :, 1].min()
    d2 = img[:, :, 2].max() - img[:, :, 2].min()
    return d1 < 10 and d2 < 10


class Triple_let(Dataset):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_dir = Path('/data/CASME_test')

    def __init__(self, df1, df2, face_str, micro_str):
        self.df1 = df1
        self.df2 = df2
        self.dataset_str = dataset_str
        self.face_str = face_str
        self.micro_str = micro_str
        self.face_label0_set = list(df1[df1['expression']==0].index)
        self.face_label1_set = list(df1[df1['expression']==1].index)
        self.face_label2_set = list(df1[df1['expression']==2].index)
        self.face_label3_set = list(df1[df1['expression']==3].index)
        self.face_label4_set = list(df1[df1['expression']==4].index)
        self.face_label5_set = list(df1[df1['expression']==5].index)
        self.face_label6_set = list(df1[df1['expression']==6].index)

        self.micro_label0_set = list(df2[df2['expression']==0].index)
        self.micro_label1_set = list(df2[df2['expression']==1].index)
        self.micro_label2_set = list(df2[df2['expression']==2].index)
        self.micro_label3_set = list(df2[df2['expression']==3].index)
        self.micro_label4_set = list(df2[df2['expression']==4].index)
        self.micro_label5_set = list(df2[df2['expression']==5].index)
        self.micro_label6_set = list(df2[df2['expression']==6].index)

        if dataset_str in ['train']:
            self.preprocess1 = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(224, scale=[0.7, 1.0]),
                transforms.RandomHorizontalFlip()
            ])

        elif dataset_str in ['valid', 'test']:
            self.preprocess1 = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224)
                ])

        self.preprocess2 = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=[0.6, 1.0]),
            transforms.RandomHorizontalFlip()
            ])

        self.preprocess3 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
            ])


    def __len__(self):
        return len(self.df1)

    def __getitem__(self, index):
        face_name = self.df1['subDirectory_filePath'][index]
        face_path = self.face_dir / face_name       
        face = Image.open(face_path)
        face = self.preprocess1(face)
        face = np.array(face)
        y = self.df1['expression'][index]

        if self.dataset_str == 'train':

            if y == 0:
                expression_index = self.micro_label0_set[random.randint(1,len(self.micro_label0_set))-1]
            if y == 1:
                expression_index = self.micro_label1_set[random.randint(1,len(self.micro_label1_set))-1]
            if y == 2:
                expression_index = self.micro_label2_set[random.randint(1,len(self.micro_label2_set))-1]
            if y == 3:
                expression_index = self.micro_label3_set[random.randint(1,len(self.micro_label3_set))-1]
            if y == 4:
                expression_index = self.micro_label4_set[random.randint(1,len(self.micro_label4_set))-1]
            if y == 5:
                expression_index = self.micro_label5_set[random.randint(1,len(self.micro_label5_set))-1]
            if y == 6:
                expression_index = self.micro_label6_set[random.randint(1,len(self.micro_label6_set))-1]

            micro_name = self.df2['subDirectory_filePath'][expression_index]
            micro_path = self.micro_dir / micro_name
            micro = Image.open(micro_path)
            micro = self.preprocess1(micro)
            micro = self.preprocess3(micro)  
        
        else:
            micro = Image.open(os.path.join(self.test_dir,face_name))

        micro = self.preprocess3(micro) 
        
        return micro,face, y

    def denorm(self, X):
        X = X.numpy().transpose([1, 2, 0])
        X = X * self.std + self.mean
        X = np.clip(X, 0, 1)
        return X




class AffectNet_Twoviews(Dataset):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_dir = Path('/data2/AffectNet Database/test')

    def __init__(self, df, dataset_str, img_dir, occ_dir):
        self.df = df
        self.dataset_str = dataset_str
        self.img_dir = img_dir
        self.occ_dir = occ_dir
        self.label0_set = list(df[df['expression']==0].index)
        self.label1_set = list(df[df['expression']==1].index)
        self.label2_set = list(df[df['expression']==2].index)
        self.label3_set = list(df[df['expression']==3].index)
        self.label4_set = list(df[df['expression']==4].index)
        self.label5_set = list(df[df['expression']==5].index)
        self.label6_set = list(df[df['expression']==6].index)

        if dataset_str in ['train']:
            self.preprocess1 = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(224, scale=[0.7, 1.0]),
                transforms.RandomHorizontalFlip()
            ])

        elif dataset_str in ['valid', 'test']:
            self.preprocess1 = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224)
            ])

        self.preprocess2 = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=[0.6, 1.0]),
            transforms.RandomHorizontalFlip()
        ])

        self.preprocess3 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, index):
        img_name = self.df['subDirectory_filePath'][index]
        img_path = self.img_dir / img_name       
        X_clear = Image.open(img_path)
        X_clear = self.preprocess1(X_clear)
        X_clear = np.array(X_clear)
        if self.dataset_str == 'train':
            X = np.copy(X_clear)
            file_list = list(self.occ_dir.iterdir())
            i = random.randint(0, len(file_list) - 1)
            occ_path = file_list[i]
            occ = Image.open(occ_path)
            occ = self.preprocess2(occ)
            occ = np.array(occ)

            temp = cv2.cvtColor(X, cv2.COLOR_RGB2YCrCb)
            d1 = temp[:, :, 1].max() - temp[:, :, 1].min()
            d2 = temp[:, :, 2].max() - temp[:, :, 2].min()
            if d1 < 10 and d2 < 10:
                occ_gray = cv2.cvtColor(occ[:, :, :3], cv2.COLOR_RGB2GRAY)
                occ[:, :, 0] = occ_gray
                occ[:, :, 1] = occ_gray
                occ[:, :, 2] = occ_gray

            c = occ[:, :, 3] > 150
            X[c] = occ[:, :, :3][c]
        else:
            X = Image.open(os.path.join(self.test_dir,img_name))
        X = self.preprocess3(X)
        y = self.df['expression'][index]
        
        if y == 0:
            clear_index = self.label0_set[random.randint(1,len(self.label0_set))-1]
        elif y==1:
            clear_index = self.label1_set[random.randint(1,len(self.label1_set))-1]
        elif y==2:
            clear_index = self.label2_set[random.randint(1,len(self.label2_set))-1]
        elif y==3:
            clear_index = self.label3_set[random.randint(1,len(self.label3_set))-1]
        elif y==4:
            clear_index = self.label4_set[random.randint(1,len(self.label4_set))-1]
        elif y==5:
            clear_index = self.label5_set[random.randint(1,len(self.label5_set))-1]
        elif y==6:
            clear_index = self.label6_set[random.randint(1,len(self.label6_set))-1]
        img_name = self.df['subDirectory_filePath'][clear_index]
        img_path = self.img_dir / img_name
        X_clear = Image.open(img_path)
        X_clear = self.preprocess1(X_clear)
        X_clear = self.preprocess3(X_clear)   
        
        return X,X_clear, y


    def __len__(self):
        return len(self.df)


    def denorm(self, X):
        X = X.numpy().transpose([1, 2, 0])
        X = X * self.std + self.mean
        X = np.clip(X, 0, 1)
        return X