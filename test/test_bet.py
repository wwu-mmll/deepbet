import torch
import unittest
import numpy as np
import nibabel as nib

from deepbet import BrainExtraction
from deepbet.utils import DATA_PATH


def dice_score(x1, x2):
    inter = (x1 * x2).sum()
    union = (x1 + x2).sum()
    return ((2. * inter) / union).mean()


class TestBrainExtraction(unittest.TestCase):
    def test_run(self):
        mask = nib.load(f'{DATA_PATH}/niftis/mask.nii.gz')
        x_mask = nib.as_closest_canonical(mask).get_fdata(dtype=np.float32)

        bet = BrainExtraction(no_gpu=False)
        img1, mask1, tiv1 = bet.run(f'{DATA_PATH}/niftis/t1w.nii.gz')

        x_mask_pred = nib.as_closest_canonical(mask1).get_fdata(dtype=np.float32)
        dice = dice_score(x_mask, x_mask_pred)
        print(f'Dice Score: {100 * dice:.1f}%')

        self.assertTrue(dice > .9)

    def test_run_model(self):
        img = nib.load(f'{DATA_PATH}/niftis/t1w.nii.gz')
        x = nib.as_closest_canonical(img).get_fdata(dtype=np.float32)
        mask = nib.load(f'{DATA_PATH}/niftis/mask.nii.gz')
        x_mask = nib.as_closest_canonical(mask).get_fdata(dtype=np.float32)

        bet = BrainExtraction(no_gpu=False)
        x_mask_pred = bet.run_model(x)
        x_mask_pred = (x_mask_pred > .5).astype(np.float32)
        dice = dice_score(x_mask, x_mask_pred)
        print(f'Dice Score (just model): {100 * dice:.1f}%')

        self.assertTrue(dice > .9)

    def test_postprocess(self):
        mask = nib.load(f'{DATA_PATH}/niftis/mask.nii.gz')
        x_mask = nib.as_closest_canonical(mask).get_fdata(dtype=np.float32)
        x_mask = (x_mask > .5).astype(np.float32)

        x = x_mask.copy()
        central_cube = tuple([slice(s // 2, s // 2 + 2, 1) for s in x.shape])
        x[central_cube] = 0

        x[0, 0, 0] = 1.

        bet = BrainExtraction(no_gpu=False)
        bet.bbox = tuple([slice(0, s, 1) for s in x.shape])
        x_mask_pred = bet.postprocess(x)

        self.assertTrue(x[0, 0, 0] == 0.)  # small component removed successful
        self.assertTrue(np.all(x[central_cube] == 1.))  # hole filled successful
        self.assertTrue(np.array_equal(x_mask, x_mask_pred))  # total postprocess successful

    def test_get_bbox_with_margin(self, cube_length=5, lows=(2, 3, 4), factor=10, margin=.1):
        small_shape = np.array([15, 15, 15])
        x = torch.zeros(*list(small_shape))
        x[lows[0]:lows[0] + cube_length, lows[1]:lows[1] + cube_length, lows[2]:lows[2] + cube_length] = 1

        total_margin = cube_length * factor * margin
        bbox_bounds = [(factor * low - total_margin, factor * (low + cube_length) + total_margin) for low in lows]
        expected_bbox = tuple([slice(int(bb[0]), int(bb[1]), 1) for bb in bbox_bounds])

        bet = BrainExtraction(no_gpu=False)
        bbox = bet.get_bbox_with_margin(x, shape=list(small_shape * factor), margin=margin)

        self.assertTrue(bbox == expected_bbox)

    def test_get_bbox(self):
        x = torch.zeros(7, 7, 7)
        x[1:5, 2:5, 1:7] = .01
        x[2:4, 3:4, 2:6] = 1.

        expected_center = torch.tensor([3.0, 3.5, 4.0])
        expected_size1 = torch.tensor([2, 1, 4])
        center, size = BrainExtraction.get_bbox(x)

        self.assertTrue(torch.all(torch.eq(center, expected_center)))
        self.assertTrue(torch.all(torch.eq(size, expected_size1)))

        expected_size2 = torch.tensor([4, 3, 6])
        center, size = BrainExtraction.get_bbox(x, threshold=.001)

        self.assertTrue(torch.all(torch.eq(center, expected_center)))
        self.assertTrue(torch.all(torch.eq(size, expected_size2)))


if __name__ == '__main__':
    unittest.main()
