import glob
import customtkinter as ctk
from pathlib import Path
from functools import partial
from tkinter import PhotoImage

from deepbet import run_bet
from deepbet.utils import DATA_PATH, FILETYPES


class App(ctk.CTk):
    def __init__(self, title='deepbet', padx=2, pady=2, sticky='nsew'):
        super().__init__()

        self.title(title)
        self.grid_kwargs = {'padx': padx, 'pady': pady, 'sticky': sticky}

        self.input_files = []
        self.brain_paths = None
        self.mask_paths = None
        self.tiv_paths = None

        self.input_file_button = ctk.CTkButton(self, text='Select Input Files', command=self.set_input)
        self.input_file_button.grid(row=0, column=0, **self.grid_kwargs)
        self.input_pattern_entry = ctk.CTkEntry(self, placeholder_text='File pattern e.g. /path/to/input/*T1w.nii.gz')
        self.input_pattern_entry.bind('<Return>', command=partial(self.set_input, is_pattern=True))
        self.input_pattern_entry.grid(row=0, column=1, columnspan=3, **self.grid_kwargs)

        self.brain_dir_entry = ctk.CTkEntry(self, placeholder_text='/path/to/output/brains')
        self.brain_dir_entry.bind('<Return>', command=partial(self.set_output_dir, output_type='brain'))
        self.brain_dir_entry.grid(row=1, column=0, **self.grid_kwargs)
        self.mask_dir_entry = ctk.CTkEntry(self, placeholder_text='/path/to/output/masks')
        self.mask_dir_entry.bind('<Return>', command=partial(self.set_output_dir, output_type='mask'))
        self.mask_dir_entry.grid(row=1, column=1, columnspan=2, **self.grid_kwargs)
        self.tiv_dir_entry = ctk.CTkEntry(self, placeholder_text='/path/to/output/tivs')
        self.tiv_dir_entry.bind('<Return>', command=partial(self.set_output_dir, output_type='tiv'))
        self.tiv_dir_entry.grid(row=1, column=3, **self.grid_kwargs)

        self.threshold_label = ctk.CTkLabel(self, text='Threshold:')
        self.threshold_label.grid(row=2, column=0, **self.grid_kwargs)
        self.threshold_entry = ctk.CTkEntry(self)
        self.threshold_entry.insert('0', '0.5')
        self.threshold_entry.grid(row=2, column=1, **self.grid_kwargs)
        self.dilate_label = ctk.CTkLabel(self, text='Dilate:')
        self.dilate_label.grid(row=2, column=2, **self.grid_kwargs)
        self.dilate_entry = ctk.CTkEntry(self)
        self.dilate_entry.insert('0', '0')
        self.dilate_entry.grid(row=2, column=3, **self.grid_kwargs)

        self.status_label = ctk.CTkLabel(self, text='Selected 0 Input Files', text_color='gray')
        self.status_label.grid(row=3, column=0, **self.grid_kwargs)
        self.run_button = ctk.CTkButton(self, state='disabled', text='Run', command=self.run_processing)
        self.run_button.grid(row=3, column=1, columnspan=2, **self.grid_kwargs)
        self.no_gpu_checkbox = ctk.CTkCheckBox(self, state='disabled', text='No GPU')
        self.no_gpu_checkbox.grid(row=3, column=3, **self.grid_kwargs)

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=4, column=0, columnspan=4, **self.grid_kwargs)

    def set_input(self, event=None, is_pattern=False):
        if is_pattern:
            pattern = self.input_pattern_entry.get()
            files = [fp for fp in sorted(glob.glob(pattern)) if fp.endswith(FILETYPES)]
            if len(files) == 0:
                print(f'Found no files based on given pattern {pattern}')
        else:
            type_string = '*' + ' *'.join(FILETYPES)
            files = ctk.filedialog.askopenfilenames(filetypes=[('Image Files', type_string)])
        if files:
            self.input_files = files
            self.status_label.configure(text=f'Selected {len(files)} Input Files')
            for item in [self.input_file_button, self.input_pattern_entry]:
                item.configure(state='disabled')
            for item in [self.brain_dir_entry, self.mask_dir_entry, self.tiv_dir_entry]:
                item.configure(state='normal')

    def set_output_dir(self, event=None, output_type='brain'):
        out_dir = event.widget.get()
        if Path(out_dir).is_dir() and self.input_files is not None:
            filenames = [str(Path(fp).name.split('.')[0]) for fp in self.input_files]
            if output_type == 'brain':
                self.brain_paths = [f'{out_dir}/{fname}.nii.gz' for fname in filenames]
                self.brain_dir_entry.configure(state='disabled')
            elif output_type == 'mask':
                self.mask_paths = [f'{out_dir}/{fname}.nii.gz' for fname in filenames]
                self.mask_dir_entry.configure(state='disabled')
            else:
                self.tiv_paths = [f'{out_dir}/{fname}.csv' for fname in filenames]
                self.tiv_dir_entry.configure(state='disabled')
            self.run_button.configure(state='normal')
        if not Path(out_dir).is_dir():
            print(f'Not a directory: {out_dir}')
        if self.input_files is None:
            print(f'Before setting output directories, input files must be given')

    def update_progress(self, progress_bar):
        self.progress_bar.set(progress_bar.n / progress_bar.total)
        if progress_bar.n > 0:
            self.status_label.configure(text=progress_bar.__str__().split('| ')[-1].replace('it', '🧠'))
        self.update_idletasks()

    def run_processing(self):
        self.status_label.configure(text='Warming up...')
        threshold = float(self.threshold_entry.get())
        n_dilate = int(self.dilate_entry.get())
        no_gpu = self.no_gpu_checkbox.get()
        run_bet(self.input_files, self.brain_paths, self.mask_paths, self.tiv_paths, threshold, n_dilate, no_gpu,
                progress_bar_func=self.update_progress)
        self.destroy()


def run_gui():
    app = App()
    icon = PhotoImage(file=f'{DATA_PATH}/icon.png')
    app.iconphoto(True, icon)
    app.mainloop()


if __name__ == '__main__':
    run_gui()
