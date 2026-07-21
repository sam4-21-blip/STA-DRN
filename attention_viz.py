"""
Direct attention visualisation for STA-DRN.

Instead of inferring attention with Grad-CAM++, this hooks the STA module
and reads the *actual* spatial and temporal attention the model computes
in its forward pass:

  - spatial attention  -> heatmap over the face (WHERE the model looks)
  - temporal attention -> curve over frames     (WHEN the model looks)

Run:
    cd ~/STA-DRN
    python3 attention_viz.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from TSSTANet.tsstanet import stanet_af
from dataloader import transforms3d

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SCORE_RANGE = 63
frame_len = 32
img_size = 112

# ---------- load model ----------
Net = stanet_af(layers=[2, 2, 2, 2], in_channels=3, num_classes=1, k=2, features=16)
Net.load_state_dict(torch.load('model/best_stanet_af.pth',
                               map_location=DEVICE, weights_only=True))
Net = Net.to(DEVICE)
Net.eval()

# ---------- hook the deepest STA block ----------
# AttentionSpatiotemporalBlock computes attention_s and attention_t internally
# but doesn't expose them, so we recompute them from the block's sub-layers
# using a forward hook on the block's inputs.
sta_block = Net.layer_2[-1].conv2   # AttentionSpatiotemporalBlock

captured = {}

def hook(module, inp, out):
    # inp[0] is the tensor entering the STA block: (B, C, T, H, W)
    x = inp[0]
    with torch.no_grad():
        # --- spatial branch (mirrors the block's own forward) ---
        x_s = module.group_conv_spatial(x)
        x_s = module.bn_spatial(x_s)
        x_s = module.relu(x_s)
        att_s = module.adaptive_pool_spatial(x_s)      # (B, C, 1, H, W)
        att_s = module.fc_spatial(att_s)
        att_s = att_s.view(x_s.size(0), module.k, x_s.size(1)//module.k,
                           1, x_s.size(3), x_s.size(4))
        att_s = module.softmax(att_s)
        # collapse to a single (H, W) spatial map
        captured['spatial'] = att_s.mean(dim=(1,2,3))[0].cpu().numpy()  # (H, W)

        # --- temporal branch ---
        x_t = module.group_conv_temporal(x)
        x_t = module.bn_temporal(x_t)
        x_t = module.relu(x_t)
        att_t = module.adaptive_pool_temporal(x_t)     # (B, C, T, 1, 1)
        att_t = module.fc_temporal(att_t)
        att_t = att_t.view(x_t.size(0), module.k, x_t.size(1)//module.k,
                           x_t.size(2), 1, 1)
        att_t = module.softmax(att_t)
        # collapse to a single (T,) temporal curve
        captured['temporal'] = att_t.mean(dim=(1,2,4,5))[0].cpu().numpy()  # (T,)

sta_block.register_forward_hook(hook)

# ---------- test labels ----------
df = pd.read_csv('./datasets/avec14/label.csv')
image_path_list = df['path'].values[200:]
label_list = df['label'].values[200:]

def get_sample_idx(labels, rng):
    for i, l in enumerate(labels):
        if rng[0] <= l <= rng[1]:
            return i
    return None

samples = {
    'minimal (0-13)':   get_sample_idx(label_list, (0, 13)),
    'mild (14-19)':     get_sample_idx(label_list, (14, 19)),
    'moderate (20-28)': get_sample_idx(label_list, (20, 28)),
    'severe (29-63)':   get_sample_idx(label_list, (29, 63)),
}
print("Selected:", {k: f"label={label_list[v]}" for k,v in samples.items() if v is not None})

os.makedirs('attention_output', exist_ok=True)

def load_video(img_path):
    p = os.path.join('dataset', 'avec14', 'image', img_path)
    names = sorted(os.listdir(p), key=str)
    if len(names) <= frame_len:
        idxs = [(i*len(names)//frame_len) for i in range(frame_len)]
    else:
        mid = len(names)//2
        idxs = list(range(mid-frame_len//2, mid+frame_len//2))
    pack = np.empty((frame_len, img_size, img_size, 3), np.float32)
    for i, fr in enumerate(idxs):
        im = Image.open(os.path.join(p, names[fr])).resize((img_size, img_size))
        pack[i] = np.asarray(im)
    x = np.transpose(pack, (3,0,1,2))
    x = transforms3d.to_tensor(x)
    x = transforms3d.normalize(x)
    return x

# collect temporal curves for a combined plot
temporal_curves = {}

for label_range, idx in samples.items():
    if idx is None:
        continue
    img_path = image_path_list[idx]
    true_label = label_list[idx]
    x = load_video(img_path).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = Net(x)
        pred_score = (torch.relu(pred) * SCORE_RANGE).item()

    spatial = captured['spatial']    # (H, W)
    temporal = captured['temporal']  # (T,)
    print(f"\n{label_range}: true={true_label} pred={pred_score:.1f} "
          f"| spatial {spatial.shape} temporal {temporal.shape}")

    # ---- spatial heatmap overlay on middle frame ----
    fp = os.path.join('dataset', 'avec14', 'image', img_path)
    names = sorted(os.listdir(fp), key=str)
    mid = np.array(Image.open(os.path.join(fp, names[len(names)//2]))
                   .resize((img_size, img_size))).astype(np.float32)/255.0

    sp = spatial - spatial.min()
    if sp.max() > 0:
        sp = sp / sp.max()
    sp = cv2.resize(sp, (img_size, img_size))
    heat = cv2.applyColorMap(np.uint8(255*sp), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    overlay = np.uint8(255*(0.4*heat + 0.6*mid))

    safe = (label_range.replace(' ','_').replace('(','').replace(')','').replace('-','_'))
    Image.fromarray(overlay).save(f'attention_output/spatial_{safe}_label{true_label}.png')

    temporal_curves[f'{label_range} (BDI {true_label})'] = temporal

# ---- combined temporal attention plot ----
plt.figure(figsize=(9,4.5), dpi=150)
for name, curve in temporal_curves.items():
    plt.plot(range(len(curve)), curve, marker='o', markersize=3, label=name)
plt.xlabel('Frame index (within sampled clip)')
plt.ylabel('Temporal attention weight')
plt.title('Temporal attention across frames, by depression severity')
plt.legend(fontsize=8, frameon=False)
plt.grid(color='#E4EBF0', linewidth=0.6)
plt.tight_layout()
plt.savefig('attention_output/temporal_attention.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()

print("\nDone. See attention_output/ — spatial heatmaps + temporal_attention.png")