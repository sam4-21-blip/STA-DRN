import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from TSSTANet.tsstanet import tanet, sanet, stanet, stanet_af
from dataloader.main_dataloader import ValDataset
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SCORE_RANGE = 63
frame_len = 32
img_size = 112

df = pd.read_csv('./datasets/avec14/label.csv')
image_path_list = df['path'].values
label_list = df['label'].values
test_image_path_list = image_path_list[200:]
test_label_list = label_list[200:]

test_data = ValDataset(img_path=test_image_path_list, label_value=test_label_list,
                       dataset='avec14', frame_len=frame_len, img_size=img_size,
                       input_channel=3, transform=None)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)

MSE_loss = nn.MSELoss()
MAE_loss = nn.L1Loss()

def test_model(net, weights_path, name):
    net.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    net = net.to(DEVICE)
    net.eval()
    rmse_list, mae_list = [], []
    with torch.no_grad():
        for val_img_pack, val_label in test_loader:
            predict_list = []
            for val_img in val_img_pack:
                predict = net(val_img.to(DEVICE))
                predict = torch.relu(predict) * SCORE_RANGE
                predict = predict.mean()
                predict_list.append(predict.item())
                del predict
                torch.cuda.empty_cache()
            predict = torch.tensor(np.mean(predict_list)).unsqueeze(dim=0)
            rmse_list.append(MSE_loss(predict, val_label).item())
            mae_list.append(MAE_loss(predict, val_label).item())
    mae = np.mean(mae_list)
    rmse = np.sqrt(np.mean(rmse_list))
    print(f'{name:35s} | MAE: {mae:.4f} | RMSE: {rmse:.4f}')
    return mae, rmse

print("=== Test Set Evaluation (rows 200-299) ===\n")
results = []

models = [
    (tanet,     'model/best_tanet.pth',          'TANet       lr=0.005'),
    (tanet,     'model/best_tanet_lr001.pth',     'TANet       lr=0.001'),
    (sanet,     'model/best_sanet.pth',           'SANet       lr=0.005'),
    (sanet,     'model/best_sanet_lr001.pth',     'SANet       lr=0.001'),
    (stanet,    'model/best_stanet.pth',          'STANet      lr=0.005'),
    (stanet,    'model/best_stanet_lr001.pth',    'STANet      lr=0.001'),
    (stanet_af, 'model/best_stanet_af.pth',       'STANet-AF   lr=0.005'),
    (stanet_af, 'model/best_stanet_af_lr001.pth', 'STANet-AF   lr=0.001'),
]

for model_fn, weights, name in models:
    net = model_fn(layers=[2,2,2,2], in_channels=3, num_classes=1, k=2, features=16)
    mae, rmse = test_model(net, weights, name)
    results.append({'model':name, 'mae':mae, 'rmse':rmse})

print('\n=== Summary ===')
for r in sorted(results, key=lambda x: x['mae']):
    print(f"{r['model']:35s} | MAE: {r['mae']:.4f} | RMSE: {r['rmse']:.4f}")
