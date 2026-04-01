"""
prediction.py — Flayer 5-Fold Ensemble Classifier
===================================================
Self-contained prediction module for RSNA Intracranial Aneurysm Detection.
Uses the Flayer 5-fold EfficientNetV2 CenterNet ensemble (from the 9th-place solution).

Usage:
    from prediction import FlayerClassifier, load_volumes
    clf = FlayerClassifier(flayer_dir="/path/to/flayer/weights")
    clf.load()
    probs = clf.predict(volume_uint8)  # → dict with per-vessel + aneurysm probs
"""

import os, gc, warnings
import numpy as np
import cv2
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast
from pathlib import Path
from typing import List, Tuple, Dict

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────
# Competition Labels
# ─────────────────────────────────────────────────────────────────
LABEL_COLS = [
    'Left Infraclinoid ICA', 'Right Infraclinoid ICA',
    'Left Supraclinoid ICA', 'Right Supraclinoid ICA',
    'Left MCA', 'Right MCA',
    'Anterior Comm', 'Left ACA', 'Right ACA',
    'Left Post Comm', 'Right Post Comm',
    'Basilar Tip', 'Other Posterior',
    'Aneurysm Present',
]
LABEL_COLS_FULL = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]
VESSEL_LABELS = LABEL_COLS_FULL[:-1]
ANEURYSM_IDX = 13  # last label


# ═════════════════════════════════════════════════════════════════
# 1. DICOM Preprocessor
# ═════════════════════════════════════════════════════════════════
class FlayerDICOMPreprocessor:
    """Convert DICOM series → fixed-size (D, H, W) uint8 volume."""

    def __init__(self, target_shape=(64, 448, 448)):
        self.target_depth, self.target_height, self.target_width = target_shape

    def load_dicom_series(self, series_path):
        dicom_files = []
        for root, _, files in os.walk(series_path):
            for f in files:
                if f.endswith(".dcm"):
                    dicom_files.append(os.path.join(root, f))
        if not dicom_files:
            raise ValueError(f"No DICOM files in {series_path}")
        datasets = []
        for fp in dicom_files:
            try:
                datasets.append(pydicom.dcmread(fp, force=True))
            except Exception:
                continue
        if not datasets:
            raise ValueError(f"No valid DICOMs in {series_path}")
        return datasets

    def extract_slice_info(self, datasets):
        infos = []
        for i, ds in enumerate(datasets):
            inst = getattr(ds, "InstanceNumber", i)
            try:
                pos = getattr(ds, "ImagePositionPatient", None)
                z = float(pos[2]) if pos and len(pos) >= 3 else float(inst)
            except Exception:
                z = float(i)
            infos.append({"dataset": ds, "z_position": z})
        return infos

    def get_windowing_params(self, ds):
        mod = str(getattr(ds, "Modality", "") or "").upper()
        if mod and mod != "CT":
            return None, None
        def _f(v):
            if v is None: return None
            try: return float(v[0]) if isinstance(v, (list, tuple)) else float(v)
            except: return None
        c, w = _f(getattr(ds, "WindowCenter", None)), _f(getattr(ds, "WindowWidth", None))
        return (c, w) if c is not None and w is not None and w > 0 else (None, None)

    def apply_windowing_or_normalize(self, img, center, width):
        if center is not None and width is not None:
            lo, hi = center - width/2, center + width/2
            return ((np.clip(img, lo, hi) - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
        p1, p99 = np.percentile(img, [1, 99])
        if p99 > p1:
            return ((np.clip(img, p1, p99) - p1) / (p99 - p1 + 1e-6) * 255).astype(np.uint8)
        mn, mx = img.min(), img.max()
        if mx > mn:
            return ((img - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)
        return np.zeros_like(img, dtype=np.uint8)

    def extract_pixel_array(self, ds):
        img = ds.pixel_array.astype(np.float32)
        if img.ndim == 3: img = img[img.shape[0] // 2]
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        slope, intercept = getattr(ds, "RescaleSlope", 1), getattr(ds, "RescaleIntercept", 0)
        if slope != 1 or intercept != 0:
            img = img * float(slope) + float(intercept)
        return img

    def resize_volume_3d(self, volume):
        target = (self.target_depth, self.target_height, self.target_width)
        if volume.shape == target:
            return volume.astype(np.uint8)
        with torch.no_grad():
            vol = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            resized = F.interpolate(vol, size=target, mode="trilinear", align_corners=False)
            return resized.squeeze().clamp(0, 255).cpu().numpy().astype(np.uint8)

    def process_series(self, series_path):
        datasets = self.load_dicom_series(series_path)
        first_img = datasets[0].pixel_array
        if len(datasets) == 1 and first_img.ndim == 3:
            return self._process_3d(datasets[0])
        return self._process_2d(datasets)

    def _process_3d(self, ds):
        vol = ds.pixel_array.astype(np.float32)
        s, i = getattr(ds, "RescaleSlope", 1), getattr(ds, "RescaleIntercept", 0)
        if s != 1 or i != 0: vol = vol * float(s) + float(i)
        wc, ww = self.get_windowing_params(ds)
        slices = [self.apply_windowing_or_normalize(vol[j], wc, ww) for j in range(vol.shape[0])]
        return self.resize_volume_3d(np.stack(slices))

    def _process_2d(self, datasets):
        infos = sorted(self.extract_slice_info(datasets), key=lambda x: x["z_position"])
        wc, ww = self.get_windowing_params(infos[0]["dataset"])
        processed = []
        for sd in infos:
            img = self.extract_pixel_array(sd["dataset"])
            img = self.apply_windowing_or_normalize(img, wc, ww)
            if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            processed.append(cv2.resize(img, (self.target_width, self.target_height)))
        return self.resize_volume_3d(np.stack(processed))


# ═════════════════════════════════════════════════════════════════
# 2. CenterNet3D — Flayer Model Architecture
# ═════════════════════════════════════════════════════════════════
class CenterNet3DInfer(nn.Module):
    """3D CenterNet with EfficientNetV2 backbone for vessel-level classification."""
    def __init__(self, model_name, size, n_classes, base_ch=32):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False,
                                          features_only=True, out_indices=(-2,))
        with torch.no_grad():
            feat_ch = self.backbone(torch.randn(1, 3, size, size))[-1].shape[1]
        self.head_conv = nn.Sequential(
            nn.Conv3d(feat_ch, base_ch, 3, padding=1),
            nn.BatchNorm3d(base_ch), nn.ReLU(inplace=True))
        self.heatmap_head = nn.Conv3d(base_ch, n_classes, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x2d = x.squeeze(1)
        feats = [self.backbone(x2d[:, i:i+1].expand(-1, 3, -1, -1))[-1] for i in range(D)]
        return {'heatmap': self.heatmap_head(self.head_conv(torch.stack(feats, dim=2)))}


# ═════════════════════════════════════════════════════════════════
# 3. FlayerClassifier — Main Interface
# ═════════════════════════════════════════════════════════════════
class FlayerClassifier:
    """
    Flayer 5-fold EfficientNetV2 CenterNet ensemble.

    Usage:
        clf = FlayerClassifier(flayer_dir="/path/to/weights")
        clf.load()
        result = clf.predict(volume_uint8)
        print(result['aneurysm_prob'])    # float
        print(result['vessel_probs'])     # dict {vessel_name: float}
        print(result['raw_probs'])        # np.array shape (14,)
    """

    MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
    SIZE = 448
    N_FOLD = 5

    def __init__(self, flayer_dir: str):
        self.flayer_dir = flayer_dir
        self.models = {}
        self.transform = None

    def load(self):
        """Load all 5-fold models."""
        self.transform = A.Compose([
            A.Resize(self.SIZE, self.SIZE), A.Normalize(), ToTensorV2()])
        for fold in range(self.N_FOLD):
            ckpt = os.path.join(self.flayer_dir,
                                f"{self.MODEL_NAME}_fold{fold}_best.pth")
            if not os.path.exists(ckpt):
                print(f"  ⚠️ fold {fold}: not found"); continue
            model = CenterNet3DInfer(self.MODEL_NAME, self.SIZE,
                                     len(VESSEL_LABELS), 32).to(device)
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state.get('model', state), strict=False)
            model.eval()
            self.models[fold] = model
        print(f"  ✅ Flayer: {len(self.models)} folds loaded → {device}")

    def predict(self, volume_uint8: np.ndarray) -> dict:
        """
        Run ensemble prediction on a preprocessed volume.

        Args:
            volume_uint8: (D, H, W) uint8 volume

        Returns:
            dict with 'aneurysm_prob', 'vessel_probs', 'raw_probs', 'fold_probs'
        """
        img_hwd = volume_uint8.transpose(1, 2, 0)
        t = self.transform(image=img_hwd)['image']
        if not torch.is_tensor(t): t = torch.from_numpy(t)
        t5d = t.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sum_logits, sum_w = None, 0.0
        fold_probs = []
        with torch.inference_mode():
            with autocast(enabled=(device.type == 'cuda')):
                for fold, model in self.models.items():
                    heatmap = model(t5d)['heatmap']
                    # Global max pool → per-class logit + overall
                    logits, _ = heatmap.view(1, heatmap.shape[1], -1).max(dim=2)
                    overall = logits.max(dim=1, keepdim=True).values
                    logits = torch.cat([logits, overall], dim=1).squeeze(0)
                    if sum_logits is None:
                        sum_logits = torch.zeros_like(logits)
                    sum_logits.add_(logits); sum_w += 1.0
                    fold_probs.append(logits.sigmoid().float().cpu().numpy())

        if sum_logits is None:
            probs = np.full(len(LABEL_COLS), 0.5)
        else:
            probs = torch.sigmoid(sum_logits / sum_w).float().cpu().numpy()

        vessel_probs = {LABEL_COLS[i]: float(probs[i]) for i in range(13)}
        return {
            'aneurysm_prob': float(probs[ANEURYSM_IDX]),
            'vessel_probs': vessel_probs,
            'raw_probs': probs,
            'fold_probs': fold_probs,
        }


# ═════════════════════════════════════════════════════════════════
# 4. Convenience: load_volumes
# ═════════════════════════════════════════════════════════════════
def load_volumes(series_dir, max_volumes=30, target_shape=(64, 448, 448)):
    """Load DICOM series → list of (D, H, W) uint8 volumes."""
    uids = sorted([u for u in os.listdir(series_dir)
                   if os.path.isdir(os.path.join(series_dir, u))])[:max_volumes]
    if not uids:
        raise RuntimeError(f"No series in {series_dir}")
    print(f"[Data] Loading {len(uids)} volumes...")
    preprocessor = FlayerDICOMPreprocessor(target_shape=target_shape)
    volumes = []
    for i, uid in enumerate(uids):
        try:
            volumes.append(preprocessor.process_series(os.path.join(series_dir, uid)))
            if (i+1) % 5 == 0: print(f"  {i+1}/{len(uids)} loaded")
        except Exception as e:
            print(f"  SKIP {uid[-12:]}: {e}")
        gc.collect()
    print(f"[Data] {len(volumes)} volumes, shape={volumes[0].shape}")
    return volumes
