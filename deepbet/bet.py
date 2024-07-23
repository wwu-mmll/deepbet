import torch
import warnings
import fill_voids
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F

from deepbet.utils import (DATA_PATH, dilate, normalize, load_model, apply_mask, check_file, is_file_broken,
                           reoriented_nifti, keep_largest_connected_component)


def run_bet(input_paths, brain_paths=None, mask_paths=None, tiv_paths=None, threshold=.5, n_dilate=0, no_gpu=False,
            skip_broken=True, progress_bar_func=None, **kwargs):
    assert not (brain_paths is None and mask_paths is None and tiv_paths is None), 'No destination filepaths given'
    bet = BrainExtraction(no_gpu=no_gpu, **kwargs)
    progress_bar = tqdm(enumerate(input_paths), disable=len(input_paths) == 1, total=len(input_paths))
    for i, in_path in progress_bar:
        if skip_broken and is_file_broken(in_path):
            warnings.warn(f'Skipped file {in_path} (use "run_bet(..., skip_broken=False)" for error messages)', Warning)
        else:
            check_file(in_path)
            brain_path = None if brain_paths is None else brain_paths[i]
            mask_path = None if mask_paths is None else mask_paths[i]
            tiv_path = None if tiv_paths is None else tiv_paths[i]
            bet.run(in_path, brain_path, mask_path, tiv_path, threshold, n_dilate)
            if progress_bar_func is not None:
                progress_bar_func(progress_bar)


class BrainExtraction:
    def __init__(self, no_gpu=False, model_path=None, bbox_model_path=None):
        model_path = f'{DATA_PATH}/models/model.pt' if model_path is None else model_path
        bbox_model_path = f'{DATA_PATH}/models/bbox_model.pt' if bbox_model_path is None else bbox_model_path
        self.model = load_model(model_path, no_gpu)
        self.bbox_model = load_model(bbox_model_path, no_gpu)
        self.bbox = None

    def run(self, input, brain_path=None, mask_path=None, tiv_path=None, threshold=.5, n_dilate=0):
        img = nib.load(input) if isinstance(input, str) else input
        x = nib.as_closest_canonical(img).get_fdata(dtype=np.float32)
        x = x[..., 0] if len(img.shape) == 4 else x
        mask = self.run_model(x.copy())
        mask = mask > threshold
        mask = self.postprocess(mask, n_dilate)
        x = apply_mask(x, mask)
        x = x[..., None] if len(img.shape) == 4 else x
        mask = mask[..., None] if len(img.shape) == 4 else mask
        tiv = 1e-3 * mask.sum() * np.prod(img.header.get_zooms()[:3])
        img = reoriented_nifti(x, img.affine, img.header)
        mask = reoriented_nifti(mask, img.affine, img.header)
        mask.header.set_data_dtype(np.uint8)
        self.save(img, mask, tiv, brain_path, mask_path, tiv_path)
        return img, mask, tiv

    def run_model(self, x, small_shape=(128, 128, 128), shape=(256, 256, 256), bbox_margin=.1):
        x = torch.from_numpy(x).to(next(self.model.parameters()).device)
        x = torch.nan_to_num(x)
        mask = torch.zeros_like(x)
        x_small = F.interpolate(x[None, None], small_shape, mode='nearest-exact')[0, 0]
        low, high = x_small.quantile(.005), x_small.quantile(.995)
        with torch.no_grad():
            mask_small = self.bbox_model(normalize(x_small, low, high)[None, None])[0, 1]
        mask_small = keep_largest_connected_component((mask_small > .5).float().cpu().numpy())
        mask_small = torch.from_numpy(mask_small).to(next(self.model.parameters()).device)
        self.bbox = self.get_bbox_with_margin(mask_small, mask.shape, bbox_margin)
        x = F.interpolate(x[self.bbox][None, None], shape, mode='nearest-exact')[0, 0]
        with torch.no_grad():
            mask_bbox = self.model(normalize(x, low, high)[None, None])[0, 1]
        mask[self.bbox] = F.interpolate(mask_bbox[None, None], mask[self.bbox].shape, mode='nearest-exact')[0, 0]
        return mask.cpu().numpy()

    def postprocess(self, mask, n_dilate=0):
        mask[self.bbox] = keep_largest_connected_component(mask[self.bbox])
        mask[self.bbox] = fill_voids.fill(mask[self.bbox])
        return dilate(mask, n_dilate)

    def get_bbox_with_margin(self, mask_small, shape, margin):
        margin = margin * torch.ones(3)
        scale_factor = torch.tensor(shape) / torch.tensor(mask_small.shape)
        center, size = self.get_bbox(mask_small)
        center, size = scale_factor * center, scale_factor * size
        size = (1 + 2 * margin) * size
        center, size = center.round(), size.round()
        bbox = [[int(c - s / 2), int(c + s / 2)] for c, s in zip(center, size)]
        return tuple([slice(max(0, b[0]), min(s, b[1]), 1) for b, s in zip(bbox, shape)])

    @staticmethod
    def get_bbox(x, threshold=.02):
        rs = [torch.where(x.mean(dims) > threshold)[0] for dims in [(1, 2), (0, 2), (0, 1)]]
        assert all([r.numel() > 0 for r in rs]), 'Not enough foreground to calculate bounding box'
        center = [(r.max() + 1 + r.min()) / 2 for r in rs]
        size = [(r.max() + 1) - r.min() for r in rs]
        return torch.tensor(center), torch.tensor(size)

    @staticmethod
    def save(img, mask, tiv, img_path, mask_path, tiv_path):
        if img_path is not None:
            img.to_filename(img_path)
        if mask_path is not None:
            mask.to_filename(mask_path)
        if tiv_path is not None:
            pd.Series([tiv], name='tiv_cm3').to_csv(tiv_path)
