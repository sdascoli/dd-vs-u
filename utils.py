# some useful functions
import os
import shutil
import math
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

def get_data(task, n_batches, bs, d, noise, var=.5, n_classes=None, teacher=None):
    with torch.no_grad():
        dataset = []
        if task=='classification':
            for i in range(n_batches):
                vectors = torch.ones(n_classes, d)   
                for ivec, vector in enumerate(vectors):
                    vectors[ivec] = rot(vector, 2*np.pi*ivec/n_classes)
                labels = torch.randint(n_classes,(bs,))
                x = torch.ones(bs,d)
                y = torch.ones(bs)
                for j, label in enumerate(labels):
                    x[j] *= vectors[label] + var * torch.randn(d)
                    y[j] = label if np.random.random()>noise else np.random.randint(n_classes)           
                dataset.append((x,y.long()))
        elif task=='regression':
            for i in range(n_batches):
                x = torch.randn(bs,d)
                y = teacher(x) + noise*torch.randn((bs,1))
                dataset.append((x,y))
    return dataset

def rot(x, th):
    with torch.no_grad(): 
        rotation = torch.eye(len(x))
        rotation[:2,:2] = torch.Tensor([[np.cos(th),np.sin(th)],[-np.sin(th), np.cos(th)]]) 
        return rotation @ x

def who_am_i():
    import subprocess
    whoami = subprocess.run(['whoami'], stdout=subprocess.PIPE)
    whoami = whoami.stdout.decode('utf-8')
    whoami = whoami.strip('\n')
    return whoami

def copy_py(dst_folder):
    # and copy all .py's into dst_folder
    if not os.path.exists(dst_folder):
        print("Folder doesn't exist!")
        return
    for f in os.listdir():
        if f.endswith('.py'):
            shutil.copy2(f, dst_folder)

