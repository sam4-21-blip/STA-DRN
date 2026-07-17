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
N_RUNS = 10

df = pd.read_csv('./datasets/avec14/label.csv')
image_path_list = df['path'].values[200:]
label_list = df['label'].values[200:]

MSE_loss = nn.MSELoss()
MAE_loss = nn.L1Loss()

def single_run(net):
    test_data = ValDataset(img_path=image_path_list, label_value=label_list,
                           dataset='avec14', frame_len=32, img_size=112,
                           input_channel=3, transform=None)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)
    mae_list, rmse_list = [], []
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
    return np.mean(mae_list), np.sqrt(np.mean(rmse_list))

def test_model(model_fn, weights_path, name):
    if not os.path.exists(weights_path):
        print(f"skipping {name} — not found")
        return
    net = model_fn(layers=[2,2,2,2], in_channels=3, num_classes=1, k=2, features=16)
    net.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    net = net.to(DEVICE)
    net.eval()
    maes, rmses = [], []
    for i in range(N_RUNS):
        mae, rmse = single_run(net)
        maes.append(mae); rmses.append(rmse)
        print(f"  run {i+1:2d}: MAE {mae:.4f} RMSE {rmse:.4f}")
    print(f"{name}")
    print(f"  MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"  RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}\n")

models = [
    (tanet,     'model/best_tanet.pth',           'TANet lr=0.005'),
    (tanet,     'model/best_tanet_lr001.pth',      'TANet lr=0.001'),
    (sanet,     'model/best_sanet.pth',            'SANet lr=0.005'),
    (sanet,     'model/best_sanet_lr001.pth',      'SANet lr=0.001'),
    (stanet,    'model/best_stanet.pth',           'STANet lr=0.005'),
    (stanet,    'model/best_stanet_lr001.pth',     'STANet lr=0.001'),
    (stanet_af, 'model/best_stanet_af.pth',        'STANet-AF lr=0.005'),
    (stanet_af, 'model/best_stanet_af_lr001.pth',  'STANet-AF lr=0.001'),
]

print(f"=== {N_RUNS}-run test evaluation ===\n")
for model_fn, weights, name in models:
    test_model(model_fn, weights, name)
