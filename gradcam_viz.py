"""
Grad-CAM++ for the STA-DRN 3D model — manual implementation.

The off-the-shelf pytorch-grad-cam library assumes 2D feature maps and
chokes on the 3D (C,T,H,W) tensors this network produces. This version
implements Grad-CAM++ directly with forward/backward hooks, averages the
temporal dimension, and overlays the heatmap on the middle frame.

Run:
    cd ~/STA-DRN
    python3 gradcam_viz.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
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

# ---------- target layer: last spatial conv in the deepest block ----------
target_layer = Net.layer_4[-1].conv2.group_conv_spatial

# hooks capture activations and gradients from the target layer
_features = {}


def fwd_hook(module, inp, out):
    _features['activations'] = out.detach()


def bwd_hook(module, grad_in, grad_out):
    _features['gradients'] = grad_out[0].detach()


target_layer.register_forward_hook(fwd_hook)
target_layer.register_full_backward_hook(bwd_hook)

# ---------- test labels ----------
df = pd.read_csv('./datasets/avec14/label.csv')
image_path_list = df['path'].values[200:]
label_list = df['label'].values[200:]


def get_sample_idx(labels, target_range):
    for i, l in enumerate(labels):
        if target_range[0] <= l <= target_range[1]:
            return i
    return None


samples = {
    'minimal (0-13)':   get_sample_idx(label_list, (0, 13)),
    'mild (14-19)':     get_sample_idx(label_list, (14, 19)),
    'moderate (20-28)': get_sample_idx(label_list, (20, 28)),
    'severe (29-63)':   get_sample_idx(label_list, (29, 63)),
}
print("Selected samples:",
      {k: f"label={label_list[v]}" for k, v in samples.items() if v is not None})

os.makedirs('gradcam_output', exist_ok=True)


def load_video_frames(img_path):
    image_path = os.path.join('dataset', 'avec14', 'image', img_path)
    image_names = sorted(os.listdir(image_path), key=str)
    if len(image_names) <= frame_len:
        frames = [(i * len(image_names) // frame_len) for i in range(frame_len)]
    else:
        mid = len(image_names) // 2
        frames = list(range(mid - frame_len // 2, mid + frame_len // 2))
    pack = np.empty((frame_len, img_size, img_size, 3), np.float32)
    for i, fr in enumerate(frames):
        im = Image.open(os.path.join(image_path, image_names[fr])).resize((img_size, img_size))
        pack[i] = np.asarray(im)
    x = np.transpose(pack, (3, 0, 1, 2))
    x = transforms3d.to_tensor(x)
    x = transforms3d.normalize(x)
    return x


def gradcam_plusplus(activations, gradients):
    """
    activations, gradients: (C, T, H, W) tensors from the target layer.
    Returns a normalised (H, W) heatmap.
    """
    # collapse temporal dimension by averaging
    acts = activations.mean(dim=1)   # (C, H, W)
    grads = gradients.mean(dim=1)    # (C, H, W)

    grads_relu = F.relu(grads)
    # Grad-CAM++ alpha weights
    grads_pow2 = grads_relu ** 2
    grads_pow3 = grads_pow2 * grads_relu
    sum_acts = acts.sum(dim=(1, 2), keepdim=True)
    denom = 2 * grads_pow2 + sum_acts * grads_pow3
    denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    alphas = grads_pow2 / denom
    weights = (alphas * grads_relu).sum(dim=(1, 2))  # (C,)

    cam = (weights.view(-1, 1, 1) * acts).sum(dim=0)  # (H, W)
    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.cpu().numpy()


for label_range, idx in samples.items():
    if idx is None:
        print(f"No sample found for {label_range}")
        continue

    img_path = image_path_list[idx]
    true_label = label_list[idx]
    print(f"\nProcessing: {label_range} | true BDI={true_label} | path={img_path}")

    x = load_video_frames(img_path).unsqueeze(0).to(DEVICE)

    # forward + backward to populate hooks
    Net.zero_grad()
    pred = Net(x)
    pred_score = torch.relu(pred) * SCORE_RANGE
    print(f"  Predicted BDI: {pred_score.item():.2f}")
    pred.sum().backward()

    acts = _features['activations'][0]   # (C, T, H, W)
    grads = _features['gradients'][0]     # (C, T, H, W)
    cam = gradcam_plusplus(acts, grads)   # (H, W) in [0,1]

    # overlay on middle frame
    frame_path = os.path.join('dataset', 'avec14', 'image', img_path)
    frame_names = sorted(os.listdir(frame_path), key=str)
    mid_name = frame_names[len(frame_names) // 2]
    rgb = np.array(Image.open(os.path.join(frame_path, mid_name))
                   .resize((img_size, img_size))).astype(np.float32) / 255.0

    cam_resized = cv2.resize(cam, (img_size, img_size))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = 0.5 * heatmap + 0.5 * rgb
    overlay = np.uint8(255 * overlay / overlay.max())

    safe = (label_range.replace(' ', '_').replace('(', '')
            .replace(')', '').replace('-', '_'))
    out = f'gradcam_output/{safe}_label{true_label}_pred{pred_score.item():.1f}.png'
    Image.fromarray(overlay).save(out)
    print(f"  Saved: {out}")

print("\nDone. Check the gradcam_output/ folder.")