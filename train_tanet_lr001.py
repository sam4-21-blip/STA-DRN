# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from TSSTANet.tsstanet import tanet
import torch.optim as optim
import torch.utils.data
from dataloader.main_dataloader import MainDataset as Dataset
from dataloader.main_dataloader import ValDataset
from dataloader import transforms3d
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

torch.multiprocessing.set_sharing_strategy('file_system')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 2
EPOCHS = 200
LOG_STEP = 10
SCORE_RANGE = 63

if not os.path.exists('model'):
    os.makedirs('model')

optimizer_name = 'Adam'
lr = 0.001
frame_len = 64
features = 16
sigma = 1

Net = tanet(layers=[2, 2, 2, 2], in_channels=3, num_classes=1, k=2, features=features)
Net = Net.to(DEVICE)

optimizer = getattr(optim, optimizer_name)(Net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 2)

MSE_loss_func = nn.MSELoss()
MAE_loss_func = nn.L1Loss()

df = pd.read_csv('./datasets/avec14/label.csv')
image_path_list = df['path'].values
label_list = df['label'].values

train_image_path_list = image_path_list[:100]
train_label_list = label_list[:100]
val_image_path_list = image_path_list[100:200]
val_label_list = label_list[100:200]

train_transform = transforms.Compose([
    transforms3d.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms3d.RandomHorizontalFlip()
])

train_data = Dataset(img_path=train_image_path_list, label_value=train_label_list, dataset='avec14',
                     frame_len=frame_len, img_size=224, input_channel=3, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True, num_workers=2)

val_data = ValDataset(img_path=val_image_path_list, label_value=val_label_list, dataset='avec14',
                      frame_len=32, img_size=112, input_channel=3, transform=None)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2)

best_MAE = float('inf')
total_step = math.ceil(100 / BATCHSIZE)

for epoch in range(EPOCHS):
    Net.train()
    RMSE_loss = []
    MAE_loss = []
    for step, (train_img, train_label) in enumerate(train_loader):
        predict = Net(train_img.to(DEVICE))
        predict = predict * SCORE_RANGE
        predict = predict.view(predict.size(0))
        train_label = train_label + np.random.normal(0, sigma, train_label.shape[0])
        train_label = train_label.float().to(DEVICE)
        loss = MSE_loss_func(predict, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        RMSE_loss.append(MSE_loss_func(predict, train_label).item())
        MAE_loss.append(MAE_loss_func(predict, train_label).item())
        mean_mae_loss = np.mean(MAE_loss)
        mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))
        if (step + 1) % LOG_STEP == 0:
            print('Epoch: {:d}  Step: {:d} / {:d} | train MAE: {:.4f} | train RMSE: {:.4f} | LR: {:.6f}'.format(
                epoch, step + 1, total_step, mean_mae_loss, mean_rmse_loss, optimizer.param_groups[0]['lr']))

    scheduler.step()
    Net.eval()
    RMSE_loss = []
    MAE_loss = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for step, (val_img_pack, val_label) in enumerate(val_loader):
            predict_list = []
            for val_img in val_img_pack:
                predict = Net(val_img.to(DEVICE))
                predict = torch.relu(predict) * SCORE_RANGE
                predict = predict.mean()
                predict_list.append(predict.item())
                del predict
                torch.cuda.empty_cache()
            predict = torch.tensor(np.mean(predict_list)).unsqueeze(dim=0)
            RMSE_loss.append(MSE_loss_func(predict, val_label))
            MAE_loss.append(MAE_loss_func(predict, val_label))
            if (step + 1) % 10 == 0:
                print('Step: {:d} | val label: {:.4f} | val predict: {:.4f}'.format(
                    step + 1, val_label.squeeze(), predict.squeeze()))

    mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))
    mean_mae_loss = np.mean(MAE_loss)
    timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
    print('{} val MAE: {:.4f}    val RMSE: {:.4f}'.format(timestamp, mean_mae_loss, mean_rmse_loss))

    with open('training_log_tanet_lr001.csv', 'a') as f:
        f.write(f'{epoch},{mean_mae_loss},{mean_rmse_loss}\n')

    if mean_mae_loss < best_MAE:
        best_MAE = mean_mae_loss
        torch.save(Net.state_dict(), './model/best_tanet_lr001.pth')
        print('Best MAE: {:.4f}, model saved!'.format(best_MAE))