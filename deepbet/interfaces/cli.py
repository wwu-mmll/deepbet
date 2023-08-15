import os
import glob
import argparse
import pandas as pd
from pathlib import Path

from deepbet.bet import run_bet


def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input file or folder', required=False, type=str)
    parser.add_argument('-o', '--output', help='Brain file or folder', required=False, type=str, default=None)
    parser.add_argument('-m', '--mask_output', help='Mask folder', required=False, type=str, default=None)
    parser.add_argument('-v', '--tiv_output', help='TIV folder', required=False, type=str, default=None)
    parser.add_argument('-t', '--threshold', help='Mask probability threshold', required=False, type=float, default=.5)
    parser.add_argument('-d', '--n_dilate', help='No. of dilated/eroded layers', required=False, type=int, default=0)
    parser.add_argument('-g', '--no_gpu', help='If GPU should be avoided', required=False, type=bool, default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-csv', '--paths_csv', help='.csv-file with filepaths', required=False, type=str, default=None)
    args = parser.parse_args()

    assert args.input is not None or args.paths_csv is not None, 'No input filepaths given'
    if args.paths_csv is None:
        files_in = sorted(glob.glob(f'{args.input}/*.ni*')) if os.path.isdir(args.input) else [args.input]
        fnames = pd.Series(files_in).apply(lambda f: Path(f).name.split('.')[0])
        path_out = os.getcwd() if args.output is None and args.mask_output is None else args.output
        files_out = f'{path_out}/' + fnames + '.nii.gz' if os.path.isdir(path_out) else [path_out]
        masks_out = args.mask_output
        if masks_out is not None:
            masks_out = f'{masks_out}/' + fnames + '.nii.gz' if os.path.isdir(masks_out) else [masks_out]
        tivs_out = args.tiv_output
        if tivs_out is not None:
            tivs_out = f'{tivs_out}/' + fnames + '.csv' if os.path.isdir(tivs_out) else [tivs_out]
    else:
        df = pd.read_csv(args.csv_path)
        files_in = df.iloc[:, 0]
        files_out = df.iloc[:, 1]
        masks_out = df.iloc[:, 2]
        tivs_out = df.iloc[:, 3]

    run_bet(files_in, files_out, masks_out, tivs_out, args.threshold, args.n_dilate, args.no_gpu)


if __name__ == '__main__':
    run_cli()
