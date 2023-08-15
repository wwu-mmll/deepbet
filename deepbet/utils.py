import cc3d
import torch
import numpy as np
import nibabel as nib
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif hasattr(torch.backends, 'cuda') and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def load_model(path, use_gpu=False):
    model = torch.jit.load(path)
    model.eval()
    return model.to(DEVICE if use_gpu else torch.device('cpu'))


def normalize(x, low, high):
    x = (x - low) / (high - low)
    x = x.clamp(min=0, max=1)
    x = (x - x.mean()) / x.std()
    return .226 * x + .449


def dilate(x, n_layer):
    for _ in range(abs(int(n_layer))):
        graph = cc3d.voxel_connectivity_graph(x.astype(int) + 1, connectivity=6)
        x[graph != 63] = 1 if n_layer > 0 else 0
    return x


def keep_largest_connected_component(mask):
    labels = cc3d.connected_components(mask)
    largest_label = np.bincount(labels[labels != 0]).argmax()
    mask[labels != largest_label] = 0.
    return mask


def apply_mask(x, mask):
    x = x.flatten('F')
    x[~mask.flatten('F')] = 0.
    return x.reshape(mask.shape, order='F')


def reoriented_nifti(array, affine, header):
    ornt_ras = [[0, 1], [1, 1], [2, 1]]
    ornt = nib.io_orientation(affine)
    ornt_inv = nib.orientations.ornt_transform(ornt_ras, ornt)
    return nib.Nifti1Image(nib.apply_orientation(array, ornt_inv), affine, header)
