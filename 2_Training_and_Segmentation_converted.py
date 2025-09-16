"""
Converted from `2. Training_and_Segmentation.ipynb`.
Markdown cells are included as commented blocks. Run sections interactively as needed.
"""

# <a href="https://colab.research.google.com/github/easare377/Prithvi-EO-Segmentation/blob/main/Training_and_Segmentation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Import essential libraries for data manipulation, model building, and visualization.

# !pip install terratorch

import numpy as np
import os

import requests
from tqdm import tqdm

def download_file(url, dest_path, chunk_size=1024*1024):
    """
    Download a large file from a URL with a progress bar.
    Args:
        url (str): File URL.
        dest_path (str): Destination file path.
        chunk_size (int): Download chunk size in bytes.
    """
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {dest_path}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


train_file_path = r'D:\ssm_footprint_train.tfrecord'
val_file_path = r'D:\ssm_footprint_val.tfrecord'
# Create the directory if it doesn't exist
# train_dir = os.path.dirname(train_file_path)
# if not os.path.exists(train_dir):
#     os.makedirs(train_dir)
# val_dir = os.path.dirname(val_file_path)
# if not os.path.exists(val_dir):
#     os.makedirs(val_dir)

# download_file("https://sco-training.s3.us-east-2.amazonaws.com/ssm_footprint_train.tfrecord", train_file_path)

# download_file("https://sco-training.s3.us-east-2.amazonaws.com/ssm_footprint_val.tfrecord", val_file_path)

def zero_pad_array(input_array, new_shape):
    """
    Zero-pad the input_array to the specified new_shape.
    Args:
        input_array (numpy.ndarray): Input array of shape (height, width, ...).
        new_shape (tuple): Desired new shape (new_height, new_width, ...).
    Returns:
        numpy.ndarray: Zero-padded array of shape (new_height, new_width, ...).
    """
    h, w = input_array.shape[:2]
    new_h, new_w = new_shape[:2]
    pad_h = max(new_h - h, 0)
    pad_w = max(new_w - w, 0)
    pad_values = [(0, pad_h), (0, pad_w)]
    pad_values += [(0, 0)] * (input_array.ndim - 2)
    return np.pad(input_array, pad_values, mode='constant', constant_values=0)


# Implement a PyTorch Dataset for loading images and masks from TFRecord files, including normalization and padding.

import struct
import tensorflow as tf
import torch
from torch.utils.data import Dataset
import random

def _zero_pad_hw(arr_hw_or_hwc, target_hw):
    th, tw = target_hw
    if arr_hw_or_hwc.ndim == 2:
        h, w = arr_hw_or_hwc.shape
        pad = ((max((th-h)//2,0), max(th-h,0) - max((th-h)//2,0)),
               (max((tw-w)//2,0), max(tw-w,0) - max((tw-w)//2,0)))
    else:
        h, w, c = arr_hw_or_hwc.shape
        pad = ((max((th-h)//2,0), max(th-h,0) - max((th-h)//2,0)),
               (max((tw-w)//2,0), max(tw-w,0) - max((tw-w)//2,0)),
               (0,0))
    return np.pad(arr_hw_or_hwc, pad, mode='constant')

def _zero_pad_cthw(X, target_hw):
    # pad only H,W for (C,T,H,W)
    C,T,H,W = X.shape
    img = np.transpose(X, (2,3,0,1)).reshape(H, W, C*T)      # (H,W,C*T)
    img = _zero_pad_hw(img, target_hw)
    H2, W2, CT = img.shape
    img = np.transpose(img, (2,0,1)).reshape(C, T, H2, W2)
    return img

class MineFootprintTFRecordDataset(Dataset):
    # Your 6-band stats; if C != 6 we fall back to per-sample normalization
    MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
    STD  = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)

    _feature_desc = {
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "mask_raw":  tf.io.FixedLenFeature([], tf.string),
        "height":    tf.io.FixedLenFeature([], tf.int64),
        "width":     tf.io.FixedLenFeature([], tf.int64),
        "channels":  tf.io.FixedLenFeature([], tf.int64),
        "timesteps": tf.io.FixedLenFeature([], tf.int64),
        "temporal_coords": tf.io.VarLenFeature(tf.float32),  # length = 2*T
        "location_coords": tf.io.FixedLenFeature([2], tf.float32),
    }

    def __init__(self, tfrecord_file, transform=None, pad_to=(224, 224)):
        super().__init__()
        self.tfrecord_path = os.fspath(tfrecord_file)
        self.transform = transform
        self.pad_to = pad_to
        self._offsets = self._scan_index()
        self._fh = open(self.tfrecord_path, 'rb')

    def _scan_index(self):
        offsets = []
        with open(self.tfrecord_path, 'rb') as f:
            pos = 0
            while True:
                header = f.read(12)                  # 8 len + 4 len_crc
                if not header:
                    break
                rec_len = struct.unpack('<Q', header[:8])[0]
                offsets.append(pos)
                pos += 12 + rec_len + 4             # header + data + data_crc
                f.seek(pos)
        return offsets

    def _read_record(self, offset):
        self._fh.seek(offset)
        header = self._fh.read(12)
        rec_len = struct.unpack('<Q', header[:8])[0]
        data = self._fh.read(rec_len)
        _ = self._fh.read(4)
        return data

    def __len__(self):
        return len(self._offsets)

    def _normalize_cthw(self, X):
        C,T,H,W = X.shape
        if C == len(self.MEAN):
            mean = self.MEAN.reshape(C,1,1,1)
            std  = (self.STD + 1e-6).reshape(C,1,1,1)
            return (X - mean) / std
        # Fallback: per-sample per-channel across (T,H,W)
        m = X.mean(axis=(1,2,3), keepdims=True)
        s = X.std(axis=(1,2,3), keepdims=True) + 1e-6
        return (X - m) / s

    def _apply_albu_over_time(self, X, mask):
        """Apply Albumentations over (H,W) consistently for all time steps:
           reshape (C,T,H,W) -> (H,W,C*T), apply, then back."""
        C,T,H,W = X.shape
        img = np.transpose(X, (2,3,0,1)).reshape(H, W, C*T)   # (H,W,C*T)

        aug = self.transform(image=img, mask=mask)
        img_aug, msk_aug = aug["image"], aug["mask"]

        # Handle ToTensorV2 in pipeline
        if isinstance(img_aug, torch.Tensor):
            # Albumentations returns CHW if ToTensorV2; convert to HWC
            img_aug = img_aug.cpu().numpy().transpose(1,2,0)
        if isinstance(msk_aug, torch.Tensor):
            msk_aug = msk_aug.cpu().numpy()

        H2, W2, CT2 = img_aug.shape
        if CT2 % C != 0:
            raise RuntimeError(f"Aug channels {CT2} not divisible by C={C}")
        T2 = CT2 // C
        X2 = np.transpose(img_aug, (2,0,1)).reshape(C, T2, H2, W2)
        return X2, msk_aug

    def __getitem__(self, idx):
        serialised = self._read_record(self._offsets[idx])
        ex = tf.io.parse_single_example(serialised, self._feature_desc)

        H = int(ex["height"])
        W = int(ex["width"])
        C = int(ex["channels"])
        T = int(ex["timesteps"])

        img = np.frombuffer(ex["image_raw"].numpy(), dtype=np.float32).reshape((C, T, H, W))
        msk = np.frombuffer(ex["mask_raw"].numpy(),  dtype=np.uint8).reshape((H, W))

        img = np.nan_to_num(img, nan=0.0)
        msk = np.nan_to_num(msk.astype(np.float32), nan=0.0).astype(np.uint8)

        # Normalize per channel (broadcast over T,H,W)
        img = self._normalize_cthw(img)

        # Center-pad to target size
        img = _zero_pad_cthw(img, self.pad_to)       # (C,T,H2,W2)
        msk = _zero_pad_hw(msk, self.pad_to)         # (H2,W2)

        # Temporal/Location coords
        temporal_coords = tf.sparse.to_dense(ex["temporal_coords"]).numpy().astype(np.float32).reshape(T, 2)
        location_coords = ex["location_coords"].numpy().astype(np.float32)   # (2,)

        # Optional Albumentations over time (geom only)
        if self.transform is not None:
            img, msk = self._apply_albu_over_time(img, msk)

        # To torch
        img = torch.from_numpy(img).float()                  # (C,T,H,W)
        msk = torch.from_numpy(msk).long()                   # (H,W)
        temporal_coords = torch.from_numpy(temporal_coords)  # (T,2)
        location_coords = torch.from_numpy(location_coords)  # (2,)

        return {
            "image": img,  # (C,T,H,W)
            "temporal_coords": temporal_coords,  # (T,2)
            "location_coords": location_coords,  # (2,)
            "mask": msk      # (H,W)
        }

    def __del__(self):
        try:
            if hasattr(self, "_fh") and self._fh and not self._fh.closed:
                self._fh.close()
        except Exception:
            pass


# -------- collate: pick ONE random timestep length for the whole batch --------
def make_temporal_collate(max_T=10, min_T=1):
    assert 1 <= min_T <= max_T
    def _collate(batch):
        t_sel = 4 #random.randint(min_T, max_T)
        imgs, tcoords, lcoords, masks = [], [], [], []
        for sample in batch:
            X = sample["image"]                 # (C,T,H,W)
            T = X.shape[1]
            X = X[:, -t_sel:, :, :]             # keep LAST t_sel
            imgs.append(X)

            tc = sample["temporal_coords"]      # (T,2)
            tcoords.append(tc[-t_sel:, :])

            lcoords.append(sample["location_coords"])   # (2,)
            masks.append(sample["mask"])         # (H,W)

        imgs   = torch.stack(imgs,   dim=0)      # (B,C,t_sel,H,W)
        tcoords= torch.stack(tcoords,dim=0)      # (B,t_sel,2)
        lcoords= torch.stack(lcoords,dim=0)      # (B,2)
        masks  = torch.stack(masks,  dim=0)      # (B,H,W)
        return {"image": imgs,
                "temporal_coords": tcoords,
                "location_coords": lcoords,
                "mask": masks
               }
    return _collate

def make_temporal_collate(timesteps=4):
    #assert 1 <= min_T <= max_T
    def _collate(batch):
        t_sel = timesteps #random.randint(min_T, max_T)
        imgs, tcoords, lcoords, masks = [], [], [], []
        for sample in batch:
            X = sample["image"]                 # (C,T,H,W)
            T = X.shape[1]
            X = X[:, -t_sel:, :, :]             # keep LAST t_sel
            imgs.append(X)

            tc = sample["temporal_coords"]      # (T,2)
            tcoords.append(tc[-t_sel:, :])

            lcoords.append(sample["location_coords"])   # (2,)
            masks.append(sample["mask"])         # (H,W)

        imgs   = torch.stack(imgs,   dim=0)      # (B,C,t_sel,H,W)
        tcoords= torch.stack(tcoords,dim=0)      # (B,t_sel,2)
        lcoords= torch.stack(lcoords,dim=0)      # (B,2)
        masks  = torch.stack(masks,  dim=0)      # (B,H,W)
        return {"image": imgs,
                "temporal_coords": tcoords,
                "location_coords": lcoords,
                "mask": masks
               }
    return _collate


# Set up an Albumentations transformation pipeline for data augmentation during training.
import albumentations as A
from albumentations.pytorch import ToTensorV2
# Define transformation pipeline
transform = A.Compose([
    A.RandomRotate90(p=0.7),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2(),
])


# Prepare TFRecord Training Dataset and DataLoader
TIMESTEPS = 4
train_dataset = MineFootprintTFRecordDataset(train_file_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True,drop_last=True, collate_fn=make_temporal_collate(TIMESTEPS))

val_dataset = MineFootprintTFRecordDataset(val_file_path, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True, collate_fn=make_temporal_collate(TIMESTEPS))

# Print the shape of the training data (example)
for batch in train_loader:
    x = batch["image"]               # (B,C,t',H,W)
    tcoords = batch["temporal_coords"] # (B,t',2)
    lcoords = batch["location_coords"] # (B,2)
    y = batch["mask"]                  # (B,H,W)
    print("Batch image shape:", x.shape)
    print("Batch temporal coords shape:", tcoords.shape)
    print("Batch location coords shape:", lcoords.shape)
    print("Batch mask shape:", y.shape)
    break  # Just print the first batch


import random as _random, math
import matplotlib.pyplot as plt

def visualize_random_temporal_rgb(
    ds,
    idx: int | None = None,
    rgb_indices=(2, 1, 0),      # (R,G,B) from your band order [BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2]
    scale_divisor: float = 3000.0,
    max_cols: int = 5,
    figsize=(16, 6),
    title: str | None = None,
):
    """
    Visualize RGB for each timestep (including zero-padded frames) from MineFootprintTFRecordDataset.
    """
    assert hasattr(ds, "MEAN") and hasattr(ds, "STD"), "Dataset must expose MEAN and STD."
    if idx is None:
        idx = _random.randrange(len(ds))

    sample = ds[idx]  # {'image': (C,T,H,W) float, 'temporal_coords': (T,2), 'mask': (H,W), ...}

    X = sample["image"]            # torch.Tensor (C,T,H,W)
    tc = sample["temporal_coords"] # torch.Tensor (T,2)

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(tc, torch.Tensor):
        tc = tc.detach().cpu().numpy()

    C, T, H, W = X.shape

    # ---- unnormalize (inverse of (x - mean) / std) ----
    mean = np.asarray(ds.MEAN, dtype=np.float32).reshape(-1, 1, 1, 1)
    std  = (np.asarray(ds.STD,  dtype=np.float32) + 1e-6).reshape(-1, 1, 1, 1)
    if mean.shape[0] != C:
        # Fallback: per-sample stats over (T,H,W) if band count differs
        m = X.mean(axis=(1,2,3), keepdims=True)
        s = X.std(axis=(1,2,3), keepdims=True) + 1e-6
        X_unnorm = X * s + m
    else:
        X_unnorm = X * std + mean

    # ---- prepare plotting grid ----
    ncols = min(T, max_cols)
    nrows = math.ceil(T / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for t in range(T):
        ax = axes[t]
        # Extract RGB per timestep
        r, g, b = rgb_indices
        rgb = np.stack([X_unnorm[r, t], X_unnorm[g, t], X_unnorm[b, t]], axis=-1)  # (H,W,3)

        # Scale to 0..255 via /3000
        rgb = np.clip(rgb / scale_divisor, 0.0, 1.0)
        rgb = (rgb * 255.0).round().astype(np.uint8)

        ax.imshow(rgb)
        ax.axis("off")

        year, doy = tc[t].tolist()
        if year == 0 and doy == 0:
            ax.set_title(f"T{t+1}: padded", fontsize=9)
        else:
            ax.set_title(f"T{t+1}: {int(year)} (doy={int(doy)})", fontsize=9)

    # Hide any extra axes
    for k in range(T, nrows * ncols):
        axes[k].axis("off")

    if title is None:
        title = f"Sample idx={idx}  |  C={C}, T={T}, H  W={H}  {W}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    plt.show()


# Just call it with your dataset
# visualize_random_temporal_rgb(train_dataset)  # random sample


# Define Semantic Segmentation Model using TerraTorch
from terratorch import BACKBONE_REGISTRY
# print just the Prithvi family (any registry)
prithvi_backbones = sorted([n for n in BACKBONE_REGISTRY if "prithvi" in n])
for backbone in prithvi_backbones:
    print(f"{backbone}")


from terratorch.models.backbones.prithvi_mae import PrithviViT, get_3d_sincos_pos_embed
from terratorch.models.decoders import FCNDecoder


# ---------- helpers -----------------------------------------------------------
def set_temporal_frames(encoder: PrithviViT, T: int):
    """Make the encoder expect T frames and rebuild its pos_embed on the right device/dtype."""
    ps_t, ps_h, ps_w = encoder.patch_embed.patch_size
    assert ps_t == 1, f"temporal patch must be 1, got {ps_t}"
    encoder.num_frames = T
    encoder.patch_embed.input_size = (T,) + encoder.img_size
    encoder.patch_embed.grid_size  = [s // p for s, p in zip(encoder.patch_embed.input_size,
                                                             encoder.patch_embed.patch_size)]

    pos = get_3d_sincos_pos_embed(encoder.embed_dim, encoder.patch_embed.grid_size, add_cls_token=True)
    pos = torch.from_numpy(pos).float().unsqueeze(0).to(
        encoder.cls_token.device, dtype=encoder.cls_token.dtype
    )
    # replace buffer (non-persistent so it follows device moves cleanly)
    encoder.register_buffer("pos_embed", pos, persistent=False)


def patch_pos_interp_device_safety(encoder: PrithviViT):
    """
    Some builds return CPU tensors from interpolate_pos_encoding; make sure they land on the buffer's device/dtype.
    """
    _old = encoder.interpolate_pos_encoding

    def _safe(sample_shape):
        out = _old(sample_shape)
        return out.to(encoder.pos_embed.device, dtype=encoder.pos_embed.dtype)

    encoder.interpolate_pos_encoding = _safe  # bind to instance


# ---------- model -------------------------------------------------------------
class PrithviFCNSeg(torch.nn.Module):
    """
    PrithviViT encoder -> token-to-image reshape -> FCNDecoder -> 1x1 head.
    """
    def __init__(
        self,
        encoder: PrithviViT,
        decoder: FCNDecoder,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head    = torch.nn.Conv2d(decoder.out_channels, num_classes, kernel_size=1)

    def forward(self, x, temporal_coords=None, location_coords=None):
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]

        feats = self.encoder.forward_features(
            x, temporal_coords=temporal_coords, location_coords=location_coords
        )  # list[Tensor], each [B, 1 + L, embed_dim]

        maps = self.encoder.prepare_features_for_image_model(feats)  # list[Tensor], e.g. last is [B, E*T, 14, 14]

        decoded = self.decoder(maps)               # (B, decoder_channels, Hdec, Wdec)

        logits = self.head(decoded)                # (B, num_classes, Hdec, Wdec)

        if logits.shape[-2:] != (H, W):
            logits = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        return logits


# ---------- factory -----------------------------------------------------------
def build_prithvi_fcn_segmentation(
    *,
    num_classes: int = 3,
    in_chans: int = 6,
    img_size: int = 224,
    T: int = 10,
    patch_size=(1, 16, 16),
    embed_dim: int = 1024,
    depth: int = 24,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    coords_encoding=("time", "location"),
    decoder_channels: int = 256,
    decoder_num_convs: int = 4,
    device: torch.device | str = "cuda",
):
    # 1) Encoder
    encoder = PrithviViT(
        img_size=img_size,
        patch_size=patch_size,  # (1,16,16) required for temporal encoding
        num_frames=T,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        coords_encoding=list(coords_encoding),
    )

    # Make pos-embed interpolation device-safe and lock T
    patch_pos_interp_device_safety(encoder)
    set_temporal_frames(encoder, T)

    # 2) Decoder: expects a list of embed_dims per stage; encoder.out_channels is already that list
    decoder = FCNDecoder(embed_dim=encoder.out_channels, channels=decoder_channels, num_convs=decoder_num_convs)

    # 3) Compose full model
    model = PrithviFCNSeg(encoder, decoder, num_classes=num_classes)
    return model.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_prithvi_fcn_segmentation(
    num_classes=3,
    in_chans=6,
    img_size=224,
    T=TIMESTEPS,
    decoder_num_convs=5,
    device=device,
)


import torch.nn.functional as F

@torch.no_grad()
def load_prithvi_encoder_weights_for_seg(
    seg_model,                 # instance of PrithviFCNSeg from earlier",
    ckpt_path: str,
    *,
    strict: bool = False,
    resize_patch_if_needed: bool = False,
):
    """
    Load pretrained Prithvi encoder weights into seg_model.encoder.
    """
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    device = next(seg_model.parameters()).device

    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("state_dict", sd)

    enc = seg_model.encoder
    model_sd = enc.state_dict()

    enc_sd = {}
    for k, v in state.items():
        if k.startswith("model.encoder."):
            enc_sd[k.replace("model.encoder.", "")] = v
        elif k.startswith("encoder."):
            enc_sd[k.replace("encoder.", "")] = v
        elif k.startswith("model.backbone.encoder."):
            enc_sd[k.replace("model.backbone.encoder.", "")] = v
        elif k.startswith("backbone.encoder."):
            enc_sd[k.replace("backbone.encoder.", "")] = v
        elif k in model_sd:
            enc_sd[k] = v

    pe = enc_sd.get("pos_embed", None)
    if isinstance(pe, torch.Tensor) and pe.shape != enc.pos_embed.shape:
        enc_sd.pop("pos_embed", None)
        print(f"[weights] dropped pos_embed (ckpt {tuple(pe.shape)} != model {tuple(enc.pos_embed.shape)})")

    if resize_patch_if_needed:
        tgt_w = enc.patch_embed.proj.weight.shape
        ck_w = enc_sd.get("patch_embed.proj.weight", None)
        if isinstance(ck_w, torch.Tensor) and ck_w.shape != tgt_w:
            oi = ck_w.shape[0] * ck_w.shape[1]
            w_oi = ck_w.reshape(oi, 1, ck_w.shape[2], ck_w.shape[3], ck_w.shape[4]).float()
            w_resized = F.interpolate(
                w_oi, size=(tgt_w[2], tgt_w[3], tgt_w[4]),
                mode="trilinear", align_corners=True
            ).reshape(tgt_w).to(dtype=ck_w.dtype)
            enc_sd["patch_embed.proj.weight"] = w_resized
            print(f"[weights] resized patch_embed.proj.weight {tuple(ck_w.shape)} -> {tuple(tgt_w)}")

        ck_b = enc_sd.get("patch_embed.proj.bias", None)
        if isinstance(ck_b, torch.Tensor) and ck_b.shape != enc.patch_embed.proj.bias.shape:
            enc_sd.pop("patch_embed.proj.bias", None)
            print("[weights] dropped patch_embed.proj.bias (shape mismatch)")

    dropped = []
    for k, v in list(enc_sd.items()):
        if k in model_sd and model_sd[k].shape != v.shape:
            dropped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            enc_sd.pop(k, None)
    if dropped:
        print("[weights] dropped for shape mismatch:", dropped[:6], "..." if len(dropped) > 6 else "")

    missing, unexpected = enc.load_state_dict(enc_sd, strict=strict)
    seg_model.to(device)

    print(f"[weights] encoder loaded. missing={len(missing)} unexpected={len(unexpected)} (strict={strict})")
    if missing:    print("  missing:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected: print("  unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")


# Example usage of loading weights (adjust paths as needed)
# load_prithvi_encoder_weights_for_seg(model, r"Z:\SPOT\2023\Asare\Prithvi_models\Prithvi_EO_V2_300M_TL.pt", strict=False, resize_patch_if_needed=False)
# load_prithvi_encoder_weights_for_seg(model, r"Z:\SPOT\2023\Asare\Prithvi_models\Mae\prithvi_mae_state_dict.pt", strict=False, resize_patch_if_needed=False)


from pathlib import Path
model_save_path  = Path("prithvi_state_dict.pt")

print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Total params:", sum(p.numel() for p in model.parameters()))


# Example Model Forward Pass
B, C, T, H, W = 2, 6, TIMESTEPS, 224, 224
x = torch.randn(B, C, T, H, W, device=device)

years = torch.arange(2015, 2015+T, device=device).float().unsqueeze(0).repeat(B, 1)
+tcoords = torch.stack([years, torch.ones_like(years)], dim=-1)      # (B,T,2)
+lcoords = torch.tensor([[7.12, -1.5]] * B, device=device).float()   # (B,2)
+
+with torch.no_grad():
+    logits = model(x, temporal_coords=tcoords, location_coords=lcoords)
+print("logits:", tuple(logits.shape))  # -> (B, num_classes, H, W)
+
+
+# Set Device, Loss Function, and Optimizer
+device = 'cuda' if torch.cuda.is_available() else 'cpu'
+criterion = torch.nn.CrossEntropyLoss()
+optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-5)
+print(device)
+
+
+# Training Loop and utilities
+import torch
+from time import time
+from datetime import timedelta
+from tqdm import tqdm as _tqdm
+
+def _get_logits(out):
+    return out.output if hasattr(out, "output") else out
+
+@torch.no_grad()
+def evaluate(model, criterion, val_loader, device):
+    model.eval()
+    running_loss = 0.0
+    correct = 0
+    total = 0
+
+    for batch in val_loader:
+        images  = batch["image"].to(device)                        # (B,C,T,H,W) or (B,C,H,W)
+        masks   = batch["mask"].to(device).long()                  # (B,H,W)
+        tcoords = batch["temporal_coords"].to(device)              # (B,T,2) or (B,1,2)
+        lcoords = batch["location_coords"].to(device)              # (B,2)
+
+        logits = _get_logits(model(images, temporal_coords=tcoords, location_coords=lcoords))  # (B,K,H,W)
+        loss = criterion(logits, masks)
+
+        running_loss += loss.item() * images.size(0)
+        preds = torch.argmax(logits, dim=1)                        # (B,H,W)
+        correct += (preds == masks).sum().item()
+        total   += masks.numel()
+
+    avg_loss = running_loss / len(val_loader.dataset)
+    pixel_acc = correct / total if total > 0 else 0.0
+    return avg_loss, pixel_acc
+
+def train(model, criterion, optimizer, train_loader, val_loader, device, num_epochs, model_save_path):
+    model.to(device)
+
+    for epoch in range(1, num_epochs + 1):
+        model.train()
+        running_loss = 0.0
+        seen_samples = 0
+
+        pbar = _tqdm(
+            enumerate(train_loader, 1),
+            total=len(train_loader),
+            desc=f"Epoch {epoch}/{num_epochs}",
+            ncols=120,
+            unit="batch",
+            leave=False
+        )
+        start_time = time()
+
+        for step, batch in pbar:
+            images  = batch["image"].to(device)
+            masks   = batch["mask"].to(device).long()
+            tcoords = batch["temporal_coords"].to(device)
+            lcoords = batch["location_coords"].to(device)
+
+            bsz = images.size(0)
+            optimizer.zero_grad()
+
+            logits = _get_logits(model(images, temporal_coords=tcoords, location_coords=lcoords))
+            loss = criterion(logits, masks)
+            loss.backward()
+            optimizer.step()
+
+            running_loss += loss.item() * bsz
+            seen_samples += bsz
+            avg_loss_so_far = running_loss / seen_samples
+
+            elapsed = time() - start_time
+            batches_left = len(train_loader) - step
+            eta = timedelta(seconds=int(elapsed / max(step,1) * batches_left))
+            pbar.set_postfix_str(f"train_loss: {avg_loss_so_far:.4f} - ETA: {eta}")
+
+        train_epoch_loss = running_loss / seen_samples
+
+        # ---------- Validation ----------
+        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
+
+        # ---------- Save ----------
+        torch.save(model.state_dict(), model_save_path)
+
+        print(f"\n📌 Epoch {epoch}/{num_epochs} "
+              f"— train_loss: {train_epoch_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4%}")
+
+
+# To run training:
+# train(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=50, model_save_path=model_save_path)
+
+
+torch.save(model.state_dict(), model_save_path)
+
+
+# Prediction and Visualization Utilities
+import matplotlib.pyplot as _plt
+
+DEFAULT_ENCODING = {
+    3: [255, 255,   0],  # yellow
+    2: [180,  96,   0],  # brown-ish
+    1: [251,  72, 196],  # magenta
+    0: [  0,   0,   0],  # background
+}
+
+def _overlay(rgb, mask, colour_encoding, alpha=0.4):
+    """Blend segmentation mask on top of rgb in [0,1]."""
+    h, w = mask.shape
+    colour_arr = np.zeros((h, w, 3), dtype=np.float32)
+    for cls_id, colour in colour_encoding.items():
+        colour_arr[mask == cls_id] = np.array(colour, dtype=np.float32) / 255.0
+    return (1 - alpha) * rgb + alpha * colour_arr
+
+@torch.no_grad()
+def visualise_random_prediction_temporal(
+    model,
+    dataset,
+    *,
+    timesteps: int = 10,
+    device: str = "cuda",
+    colour_encoding: dict[int, list[int]] | None = None,
+    rgb_divisor: float = 3000.0,
+    alpha: float = 0.6,
+    rgb_indices = (2, 1, 0),   # (R,G,B) from [BLUE,GREEN,RED,...] => [2,1,0]
+):
+    colour_encoding = colour_encoding or DEFAULT_ENCODING
+    model.eval().to(device)
+
+    # ---- pick a random sample
+    idx = random.randrange(len(dataset))
+    sample = dataset[idx]
+
+    # Expecting temporal tensors
+    img = sample["image"]              # (C,T,H,W) torch.Tensor (z-scored)
+    tcoords = sample["temporal_coords"]  # (T,2) torch.Tensor
+    lcoords = sample["location_coords"]  # (2,)  torch.Tensor
+    gt = sample["mask"].numpy()        # (H,W) numpy/uint8
+
+    assert img.ndim == 4, f"Expected image (C,T,H,W), got {tuple(img.shape)}"
+    C, T_full, H, W = img.shape
+    t_use = min(timesteps, T_full)
+
+    # ---- select last `timesteps`
+    img_sel = img[:, -t_use:, :, :]                    # (C,t_use,H,W)
+    tcoords_sel = tcoords[-t_use:, :]                  # (t_use,2)
+
+    # ---- prepare batch for model
+    x = img_sel.unsqueeze(0).to(device)                # (1,C,t_use,H,W)
+    tc = tcoords_sel.unsqueeze(0).to(device)           # (1,t_use,2)
+    lc = lcoords.unsqueeze(0).to(device)               # (1,2)
+
+    # ---- forward
+    out = model(x, temporal_coords=tc, location_coords=lc)
+    logits = out.output if hasattr(out, "output") else out  # (1,num_classes,H,W) or (1,*,h,w)
+    pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy()  # (H,W)
+
+    # ---- un-normalise ALL selected timesteps for visualisation
+    img_np = img_sel.detach().cpu().numpy()  # (C,t_use,H,W)
+    mean = dataset.MEAN.reshape(-1, 1, 1, 1)
+    std  = dataset.STD.reshape(-1, 1, 1, 1)
+    img_np = img_np * std + mean            # back to reflectance-ish scale
+
+    # ---- build RGB per timestep in [0,1]
+    rgb_seq = []
+    for t in range(t_use):
+        rgb = img_np[list(rgb_indices), t, :, :]          # (3,H,W)
+        rgb = np.transpose(rgb, (1, 2, 0)) / rgb_divisor  # (H,W,3)
+        rgb = np.clip(rgb, 0.0, 1.0)
+        rgb_seq.append(rgb)
+
+    # ---- overlays for the LAST timestep
+    last_rgb = rgb_seq[-1]
+    gt_overlay   = _overlay(last_rgb, gt,   colour_encoding, alpha)
+    pred_overlay = _overlay(last_rgb, pred, colour_encoding, alpha)
+
+    # ---- Titles showing year/doy, mark PADs
+    tcoords_np = tcoords_sel.cpu().numpy()  # (t_use,2)
+    titles = []
+    for y, d in tcoords_np:
+        if y == 0 and d == 0:
+            titles.append("PAD (0/0)")
+        else:
+            titles.append(f"{int(y)} | DOY {int(d)}")
+
+    # ---- layout: one row of T thumbnails + 2 overlays
+    ncols = t_use + 2
+    _plt.figure(figsize=(3 * ncols, 3.2))
+
+    for i in range(t_use):
+        ax = _plt.subplot(1, ncols, i + 1)
+        ax.imshow(rgb_seq[i])
+        ax.set_title(titles[i], fontsize=9)
+        ax.axis("off")
+
+    ax = _plt.subplot(1, ncols, t_use + 1)
+    ax.imshow(gt_overlay)
+    ax.set_title("GT (last frame)", fontsize=10)
+    ax.axis("off")
+
+    ax = _plt.subplot(1, ncols, t_use + 2)
+    ax.imshow(pred_overlay)
+    ax.set_title("Pred (last frame)", fontsize=10)
+    ax.axis("off")
+
+    _plt.suptitle(f"Sample #{idx} — showing last {t_use} timesteps (target is LAST)", y=1.02)
+    _plt.tight_layout()
+    _plt.show()
+
+
+# Visualize Random Prediction (example)
+val_dataset = MineFootprintTFRecordDataset(val_file_path, transform=transform)
+val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True,drop_last=True)
+
+# visualise_random_prediction_temporal(model, dataset=val_dataset, timesteps=TIMESTEPS, device=device, colour_encoding=DEFAULT_ENCODING, rgb_divisor=3000.0, alpha=0.8, rgb_indices=(2,1,0))
