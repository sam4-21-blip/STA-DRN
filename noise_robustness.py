import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from TSSTANet.tsstanet import tanet
from dataloader.main_dataloader import ValDataset
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SCORE_RANGE = 63

df = pd.read_csv('./datasets/avec14/label.csv')
test_paths = df['path'].values[200:]
test_labels = df['label'].values[200:]

# noise levels to test (sigma) - matches paper's range
noise_levels = [0.0, 0.03, 0.06, 0.125, 0.25, 0.375, 0.5]

# load best model once
net = tanet(layers=[2,2,2,2], in_channels=3, num_classes=1, k=2, features=16)
net.load_state_dict(torch.load('model/best_tanet.pth', map_location=DEVICE, weights_only=True))
net = net.to(DEVICE)
net.eval()

def evaluate_with_noise(sigma):
    test_data = ValDataset(img_path=test_paths, label_value=test_labels,
                           dataset='avec14', frame_len=32, img_size=112,
                           input_channel=3, transform=None)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)
    mae_list, rmse_list = [], []
    with torch.no_grad():
        for img_pack, label in test_loader:
            plist = []
            for img in img_pack:
                img = img.to(DEVICE)
                if sigma > 0:
                    noise = torch.randn_like(img) * sigma
                    img = img + noise
                p = net(img)
                p = torch.relu(p) * SCORE_RANGE
                plist.append(p.mean().item())
                del p
                torch.cuda.empty_cache()
            pred = np.mean(plist)
            mae_list.append(abs(pred - label.item()))
            rmse_list.append((pred - label.item())**2)
    return np.mean(mae_list), np.sqrt(np.mean(rmse_list))

results = []
for sigma in noise_levels:
    mae, rmse = evaluate_with_noise(sigma)
    print(f"sigma={sigma:.3f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    results.append({'sigma': sigma, 'mae': mae, 'rmse': rmse})

sigmas = [r['sigma'] for r in results]
maes = [r['mae'] for r in results]
rmses = [r['rmse'] for r in results]

plt.figure(figsize=(8, 5), dpi=150)
plt.plot(sigmas, maes, marker='o', color='#1C7293', linewidth=2, label='MAE')
plt.plot(sigmas, rmses, marker='s', color='#F96167', linewidth=2, label='RMSE')
plt.xlabel('Gaussian noise level (σ)', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Model robustness to input noise (TANet, test set)', fontsize=13, fontweight='bold', color='#21295C')
plt.legend(frameon=False)
plt.grid(color='#E4EBF0', linewidth=0.6)
plt.gca().set_axisbelow(True)
plt.tight_layout()
plt.savefig('noise_robustness.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\nsaved noise_robustness.png")
