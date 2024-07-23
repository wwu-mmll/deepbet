import cc3d
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
DATA_PATH = f'{Path(__file__).parents[1].resolve()}/data'
FILETYPES = ('.nii.gz', '.nii', '.img', '.mnc', '.mnc2', '.BRIK', '.REC')
MIN_FILESIZE = 102400
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif hasattr(torch.backends, 'cuda') and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def is_file_broken(fpath):
    return not Path(fpath).is_file() or not fpath.endswith(FILETYPES) or Path(fpath).stat().st_size < MIN_FILESIZE


def check_file(fp):
    assert Path(fp).is_file(), f'Filepath {fp} does not exist'
    assert fp.endswith(FILETYPES), f'File {fp} is not of supported types: {FILETYPES}'
    filesize_kb = Path(fp).stat().st_size // 1024
    min_size_kb = MIN_FILESIZE // 1024
    assert filesize_kb >= min_size_kb, f'File {fp} is probably broken since it is <{min_size_kb}kB ({filesize_kb}kB)'


def load_model(path, no_gpu=False):
    model = torch.jit.load(path)
    model.eval()
    return model.to(torch.device('cpu') if no_gpu else DEVICE)


def normalize(x, low, high):
    x = (x - low) / (high - low)
    x = x.clamp(min=0, max=1)
    x = (x - x.mean()) / x.std()
    return .226 * x + .449
    # return (x - x.mean()) / x.std()


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
    out_header = header if isinstance(header, nib.Nifti1Header) else None
    return nib.Nifti1Image(nib.apply_orientation(array, ornt_inv), affine, out_header)
