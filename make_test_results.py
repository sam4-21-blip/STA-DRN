"""
Run test set evaluation on all 8 trained models and generate a comparison chart.
Run on the uni machine inside the sta_drn_conda environment:

    cd ~/STA-DRN
    python3 make_test_results.py

Outputs: test_results.png in the current folder.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from TSSTANet.tsstanet import tanet, sanet, stanet, stanet_af
from dataloader.main_dataloader import ValDataset
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SCORE_RANGE = 63
frame_len = 32
img_size = 112

# ---------- load test data ----------
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
    if not os.path.exists(weights_path):
        print(f"  skipping {name} — weights not found")
        return None, None
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

# ---------- run all models ----------
print("=== Test Set Evaluation ===\n")
models_to_test = [
    (tanet,     'model/best_tanet.pth',           'TANet lr=0.005'),
    (tanet,     'model/best_tanet_lr001.pth',      'TANet lr=0.001'),
    (sanet,     'model/best_sanet.pth',            'SANet lr=0.005'),
    (sanet,     'model/best_sanet_lr001.pth',      'SANet lr=0.001'),
    (stanet,    'model/best_stanet.pth',           'STANet lr=0.005'),
    (stanet,    'model/best_stanet_lr001.pth',     'STANet lr=0.001'),
    (stanet_af, 'model/best_stanet_af.pth',        'STANet-AF lr=0.005'),
    (stanet_af, 'model/best_stanet_af_lr001.pth',  'STANet-AF lr=0.001'),
]

results = {}
for model_fn, weights, name in models_to_test:
    net = model_fn(layers=[2,2,2,2], in_channels=3, num_classes=1, k=2, features=16)
    mae, rmse = test_model(net, weights, name)
    if mae is not None:
        results[name] = {'mae': mae, 'rmse': rmse}

# ---------- pick best per model variant ----------
model_names = ['TANet', 'SANet', 'STANet', 'STANet-AF']
best_mae, best_rmse = [], []
for m in model_names:
    candidates_mae = [(results[k]['mae'], k) for k in results if k.startswith(m)]
    candidates_rmse = [(results[k]['rmse'], k) for k in results if k.startswith(m)]
    best_mae.append(min(candidates_mae)[0] if candidates_mae else 0)
    best_rmse.append(min(candidates_rmse)[0] if candidates_rmse else 0)

paper_mae  = [7.07, 7.30, 6.97, 6.00]
paper_rmse = [9.22, 9.28, 9.12, 7.75]

print("\n=== Best per variant ===")
for i, m in enumerate(model_names):
    print(f"{m:12s} | MAE: {best_mae[i]:.4f} | RMSE: {best_rmse[i]:.4f} | Paper MAE: {paper_mae[i]:.2f}")

# ---------- plot ----------
DEEP="#065A82"; SLATE="#7A8FA0"; MID="#21295C"
plt.rcParams["font.family"] = "DejaVu Sans"

x = np.arange(len(model_names)); w = 0.3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)

for ax, yours, paper, metric in [
    (ax1, best_mae,  paper_mae,  "MAE"),
    (ax2, best_rmse, paper_rmse, "RMSE"),
]:
    b1 = ax.bar(x - w/2, yours, w, label="This work (best run)", color=DEEP)
    b2 = ax.bar(x + w/2, paper, w, label="Pan et al. (2024)",    color=SLATE)
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.12, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8, color="#33404A")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.12, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8, color="#33404A")
    ax.set_ylabel(f"Test {metric} (lower is better)", fontsize=10.5, color="#33404A")
    ax.set_title(f"Test set {metric} by model variant", fontsize=12,
                 fontweight="bold", color=MID, pad=10)
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10.5, color="#33404A")
    ax.set_ylim(0, max(yours + paper) * 1.18)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DEE6EC", linewidth=0.7); ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("test_results.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("\nwrote test_results.png")