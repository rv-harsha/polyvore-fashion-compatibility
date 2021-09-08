# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision.utils as tv_utils
from torchsummary import summary
from torchvision.models import resnet50

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import argparse
import time
import sys
import copy
from tqdm import tqdm
import os.path as osp

from utils import Config
from datetime import datetime
import seaborn as sns
sns.set_style('darkgrid')

np.random.seed(0)
torch.manual_seed(0)


# %%
from data import get_dataloader
dataloaders = get_dataloader()

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Display sample images loaded by dataloader
dataiter = iter(dataloaders["train"])
batch = next(dataiter)
concatenated = torch.cat((batch[0],batch[1]),0)
imshow(tv_utils.make_grid(batch[0]))
print("Shape -->",batch[0].shape)


# %%
class CompatibilityClassifier(nn.Module):
    def __init__(self):
        super(CompatibilityClassifier, self).__init__()
        self.cnn1 = nn.Sequential(
            
            nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=2), #(6,256) / (6,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=.2),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  #(256,128) / (32, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=.2),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  #(128,64) / (64, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=.2),

            # Many more convolutional layers need to be added to increase accuracy

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 1, kernel_size=56, stride=1, padding=0)    #(64,1)  / (32,1)  
        )
        
    def forward(self, x1, x2):
        x = torch.cat((x1, x2),1)
        output = self.cnn1(x)
        output = torch.sigmoid(output)
        output = output.reshape(-1)
        return output


# %%
model = CompatibilityClassifier()
# model = resnet50(pretrained=True)

# # weight = model.conv1.weight.clone()
# model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False) #here 4 indicates 4-channel input

# # model.conv1.weight[:, :3] = weight
# # model.conv1.weight[:, 3] = model.conv1.weight[:, 0]

# # for param in model.parameters():
# #     param.requires_grad = False

# num_ftrs = model.fc.in_features

# model.fc = nn.Sequential(
#     nn.Linear(num_ftrs, 2),
#     nn.Sigmoid()
#     )

device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
model.to(device)

# To write the model summary and network architecture to a text file
original_stdout = sys.stdout  # Save a reference to the original standard output
with open(osp.join(Config['root_path'], 'compatibility-model.txt'), 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    # print('------------ MODEL SUMMARY ---------------\n')
    # summary(model, input_size=((3, 224, 224),(3, 224, 224)), batch_size=-1)
    # print('\n----------------------------------------\n')
    print('------------ MODEL ---------------\n')
    print(model)
    print('\n---------------- END OF FILE ------------\n')
    sys.stdout = original_stdout  # Reset the standard output to its original value

criterion = nn.BCELoss()

# Optmize only the parameters in the FC Layer
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# %%
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}


# %%
from data import get_dataloader
dataloaders = get_dataloader()

train_loader = dataloaders["train"]
val_loader = dataloaders["valid"]
test_loader = dataloaders["test"]


# %%
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'logs/polyvore/{ts}')
best_model = model
best_acc = 0.0
for e in tqdm(range(1, 30)):
    
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    # exp_lr_scheduler.step()
    for i, data in enumerate(train_loader,0):
           
        img0, img1 , labels = data
        labels = labels.squeeze(1)
        img0, img1 , labels = img0.to(device), img1.to(device) , labels.to(device)
        optimizer.zero_grad()
        outputs = model(img0,img1)
        # outputs = model(torch.cat((img0,img1),1))

        pred = torch.tensor([1 if x > 0.5 else 0 for x in outputs ]).to(device) 
        train_correct = pred.eq(labels.view_as(pred)).sum().item()
        
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item() 
        train_epoch_acc += train_correct

    train_epoch_loss /=  len(train_loader)
    train_epoch_acc /= (len(train_loader) * train_loader.batch_size)

    writer.add_scalars('compatibility/loss', {'train': train_epoch_loss}, e+1)
    writer.add_scalars('compatibility/accuracy', {'train': train_epoch_acc}, e+1)

    # VALIDATION
    with torch.no_grad():
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        for i, data_t in enumerate(val_loader,0):
            img0, img1 , label_t = data_t
            label_t = label_t.squeeze(1)
            img0, img1 , label_t = img0.to(device), img1.to(device) , label_t.to(device)
            # outputs_t = model(torch.cat((img0,img1),1))
            outputs_t = model(img0,img1)
            pred_t = torch.tensor([1 if x > 0.5 else 0 for x in outputs_t ]).to(device)
            val_loss = criterion(outputs_t, label_t)
            val_epoch_loss += val_loss.item()
            val_epoch_acc += pred_t.eq(label_t.view_as(pred_t)).sum().item()

    val_epoch_loss /= len(val_loader)
    val_epoch_acc /= (len(val_loader) * val_loader.batch_size)

    writer.add_scalars('compatibility/loss', {'test': val_epoch_loss}, e+1)
    writer.add_scalars('compatibility/accuracy', {'test': val_epoch_acc}, e+1)

    if val_epoch_acc > best_acc:
        best_acc = val_epoch_acc
        best_model = model

        torch.save(best_model, osp.join(
            Config['root_path'], Config['checkpoint_path'], 'compatibility-model.pth'))
        print('Model saved at: {}'.format(osp.join(
            Config['root_path'], Config['checkpoint_path'], 'compatibility-model.pth')))

    loss_stats['train'].append(train_epoch_loss)
    loss_stats['val'].append(val_epoch_loss)
    accuracy_stats['train'].append(train_epoch_acc)
    accuracy_stats['val'].append(val_epoch_acc)
    print(f' Epoch {e+0:02}: | Train Loss: {train_epoch_loss:.5f} | Val Loss: {val_epoch_loss:.5f} | Train Acc: {train_epoch_acc:.3f}| Val Acc: {val_epoch_acc:.3f}')
writer.close()


# %%
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')


# %%
# Testing

with torch.no_grad(): 
    model.eval()
    total_test_loss = 0
    total_test_acc = 0
    for i, data_ in enumerate(test_loader,0):
        img0, img1 , label_ = data_
        label_ = label_.squeeze(1)
        img0, img1 , label_ = img0.to(device), img1.to(device) , label_.to(device)
        # outputs_ = model(torch.cat((img0,img1),1))
        outputs_ = model(img0,img1)
        pred_ = torch.tensor([1 if x > 0.5 else 0 for x in outputs_ ]).to(device)
        test_loss = criterion(outputs_, label_)
        total_test_loss += test_loss.item()
        total_test_acc += pred_.eq(label_.view_as(pred_)).sum().item()
total_test_loss /= len(test_loader)
total_test_acc /= (len(test_loader) * test_loader.batch_size)
print(f'Test Loss: {total_test_loss:.5f} | Test Acc: {total_test_acc:.3f}')


