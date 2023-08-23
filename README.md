<h1 align="center">
<img src='https://github.com/wwu-mmll/deepbet/assets/55840648/ceff1fcc-e14b-4938-8f03-2cb28712eec1' width='700'>
</h1><br>


This is the official implementation of the [deepbet paper](https://arxiv.org/abs/2308.07003).

deepbet is a neural network based tool which achieves state-of-the-art results for brain extraction of T1w MR images 
of healthy adults while taking ~1 second per image.

## Usage
After installation, there are three ways to use deepbet
1. ```deepbet-gui``` runs the **Graphical User Interface (GUI)**

![deepbet_gui_newest](https://github.com/wwu-mmll/deepbet/assets/55840648/7458ce57-95eb-4f55-bd9e-58aa101932b6)


2. ```deepbet-cli``` runs the **Command Line Interface (CLI)**

![deepbet_cli](https://github.com/wwu-mmll/deepbet/assets/55840648/5191cd9a-bada-4ff0-9f1b-3d499655a8f7)

3. Run deepbet directly in Python

```python
from deepbet import run_bet

input_paths = ['path/to/sub_1/t1.nii.gz', 'path/to/sub_2/t1.nii.gz']
brain_paths = ['path/to/sub_1/brain.nii.gz', 'path/to/sub_2/brain.nii.gz']
mask_paths = ['path/to/sub_1/mask.nii.gz', 'path/to/sub_2/mask.nii.gz']
tiv_paths = ['path/to/sub_1/tiv.csv', 'path/to/sub_2/tiv.csv']
run_bet(input_paths, brain_paths, mask_paths, tiv_paths, threshold=.5, n_dilate=0, no_gpu=False)
```

Besides the `input paths` and the output paths

- `brain_paths`: Destination filepaths of input nifti **files with brain extraction applied**
- `mask_paths`: Destination filepaths of **brain mask nifti files**
- `tiv_paths`: Destination filepaths of **.csv-files containing the total intracranial volume (TIV)** in cm³
    - Simpler than it sounds: TIV = Voxel volume * Number of 1-Voxels in brain mask

you can additionally do

- **Fine adjustments** via `threshold`: deepbet internally predicts values between 0 and 1 for each voxel and then includes each voxel which is above 0.5. 
You can change this threshold (e.g. to 0.1 to include more voxels).
- **Coarse adjustments** via `n_dilate`: Enlarges/shrinks mask by successively adding/removing voxels adjacent to mask surface.

and choose if you want to **use GPU (NVIDIA and Apple M1/M2 support) for speedup**

- `no_gpu`: deepbet automatically uses NVIDIA GPU or Apple M1/M2 if available. If you do not want that set no_gpu=True.

## Installation
```bash
pip install deepbet
conda install -c anaconda pyqt=5.15.7
```

## Citation
If you find this code useful in your research, please consider citing:

    @inproceedings{deepbet,
    Author = {Lukas Fisch, Stefan Zumdick, Carlotta Barkhau, Daniel Emden, Jan Ernsting, Ramona Leenings, Kelvin Sarink, Nils R. Winter, Udo Dannlowski, Tim Hahn},
    Title = {fastbet: Fast brain extraction of T1-weighted MRI using Convolutional Neural Networks},
    Journal  = {Imaging Neuroscience},
    Year = {2023}
    }
