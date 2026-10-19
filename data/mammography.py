"""Mammography image loading and study assembly (Sec. 3.1, 4.1).

Standard four-view protocol (CC + MLO of both breasts). Views are resized
so the long side fits the encoder budget and normalized to [-1, 1].
"""
import numpy as np

VIEWS = ("RCC", "RMLO", "LCC", "LMLO")


def load_dicom_array(path):
    import pydicom
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        img = img.max() - img
    return img


def normalize(img, clip=(1.0, 99.8)):
    lo, hi = np.percentile(img[img > 0], clip) if (img > 0).any() else (0, 1)
    img = np.clip(img, lo, hi)
    img = (img - lo) / max(hi - lo, 1e-6)
    return img * 2.0 - 1.0


def resize_long_side(img, target=1792):
    import cv2
    h, w_ = img.shape
    s = target / max(h, w_)
    return cv2.resize(img, (round(w_ * s), round(h * s)), interpolation=cv2.INTER_AREA)


def assemble_study(dicom_paths: dict) -> dict:
    """dicom_paths: {view: path}; returns {view: normalized array}. Missing views -> None."""
    return {
        v: (normalize(resize_long_side(load_dicom_array(dicom_paths[v])))
            if v in dicom_paths else None)
        for v in VIEWS
    }
