import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from TSSTANet.tsstanet import tanet, stanet_af
from dataloader.main_dataloader import ValDataset
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SCORE_RANGE = 63

df = pd.read_csv('./datasets/avec14/label.csv')
test_paths = df['path'].values[200:]
test_labels = df['label'].values[200:]

test_data = ValDataset(img_path=test_paths, label_value=test_labels,
                       dataset='avec14', frame_len=32, img_size=112,
                       input_channel=3, transform=None)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)

def get_predictions(model_fn, weights_path):
    net = model_fn(layers=[2,2,2,2], in_channels=3, num_classes=1, k=2, features=16)
    net.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    net = net.to(DEVICE)
    net.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for img_pack, label in test_loader:
            plist = []
            for img in img_pack:
                p = net(img.to(DEVICE))
                p = torch.relu(p) * SCORE_RANGE
                plist.append(p.mean().item())
                del p
                torch.cuda.empty_cache()
            preds.append(np.mean(plist))
            actuals.append(label.item())
    return np.array(actuals), np.array(preds)

# best model: TANet lr=0.005
actuals, preds = get_predictions(tanet, 'model/best_tanet.pth')

plt.figure(figsize=(6.5, 6.5), dpi=150)
plt.scatter(actuals, preds, color='#1C7293', s=45, alpha=0.7, edgecolors='white', linewidths=0.5)
lims = [0, 63]
plt.plot(lims, lims, '--', color='#5A6B7B', linewidth=1, label='Perfect prediction')
plt.xlim(lims); plt.ylim(lims)
plt.xlabel('Actual BDI-II score', fontsize=12)
plt.ylabel('Predicted BDI-II score', fontsize=12)
plt.title('Predicted vs actual depression scores\n(TANet, test set)', fontsize=13, fontweight='bold', color='#21295C')
plt.legend(frameon=False)
plt.grid(color='#E4EBF0', linewidth=0.6)
plt.gca().set_axisbelow(True)
plt.tight_layout()
plt.savefig('scatter_predictions.png', dpi=150, bbox_inches='tight', facecolor='white')
print("saved scatter_predictions.png")

# print correlation
corr = np.corrcoef(actuals, preds)[0,1]
print(f"Pearson correlation: {corr:.4f}")
