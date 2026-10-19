"""Mammography image loading and study assembly (Sec. 3.2.2, 4.1).

Standard four-view protocol (CC_R, MLO_R, CC_L, MLO_L). Each view is
resized to 512 x 512 pixels and encoded by a frozen DINOv2 ViT-B/14
(patch 14), yielding 36 x 36 = 1,296 patch tokens plus one learnable
[CLS] token per view.
"""
import numpy as np

VIEWS = ("CC_R", "MLO_R", "CC_L", "MLO_L")
IMG_SIZE = 512


def load_dicom_array(path):
    import pydicom
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        img = img.max() - img
    return img


def normalize(img, clip=(1.0, 99.8)):
    """Percentile clipping then scaling to [-1, 1]."""
    lo, hi = np.percentile(img[img > 0], clip) if (img > 0).any() else (0, 1)
    img = np.clip(img, lo, hi)
    img = (img - lo) / max(hi - lo, 1e-6)
    return img * 2.0 - 1.0


def resize_view(img, size=IMG_SIZE):
    import cv2
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def assemble_study(dicom_paths: dict) -> dict:
    """dicom_paths: {view: path}; returns {view: 512x512 array in [-1,1]}.
    Missing views -> None."""
    return {
        v: (normalize(resize_view(load_dicom_array(dicom_paths[v])))
            if v in dicom_paths else None)
        for v in VIEWS
    }
