import sys
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from deepbet.bet import BrainExtractor


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window title
        self.setWindowTitle('deepbet')

        # Set window size
        self.setFixedSize(700, 350)

        # Set up input file button
        self.input_file_button = QPushButton('Select Input Files', self)
        self.input_file_button.clicked.connect(self.get_input_files)
        self.input_file_button.setGeometry(QRect(250, 25, 200, 30))

        # Set up brain directory checkbox
        self.brain_dir_checkbox = QCheckBox('Save extracted brains', self)
        self.brain_dir_checkbox.setGeometry(QRect(50, 80, 200, 30))
        self.brain_dir_checkbox.stateChanged.connect(self.show_brain_dir_dialog)
        self.brain_dir_checkbox.setDisabled(True)
        self.brain_dir = None

        # Set up brain directory textbox
        self.brain_dir_textbox = QLineEdit(self)
        self.brain_dir_textbox.setGeometry(QRect(210, 80, 200, 30))
        self.brain_dir_textbox.setDisabled(True)

        # Set up mask directory checkbox
        self.mask_dir_checkbox = QCheckBox('Save brain masks', self)
        self.mask_dir_checkbox.setGeometry(QRect(50, 130, 200, 30))
        self.mask_dir_checkbox.stateChanged.connect(self.show_mask_dir_dialog)
        self.mask_dir_checkbox.setDisabled(True)
        self.mask_dir = None

        # Set up brain directory textbox
        self.mask_dir_textbox = QLineEdit(self)
        self.mask_dir_textbox.setGeometry(QRect(210, 130, 200, 30))
        self.mask_dir_textbox.setDisabled(True)

        # Set up tiv directory checkbox
        self.tiv_dir_checkbox = QCheckBox('Save TIV', self)
        self.tiv_dir_checkbox.setGeometry(QRect(50, 180, 200, 30))
        self.tiv_dir_checkbox.stateChanged.connect(self.show_tiv_dir_dialog)
        self.tiv_dir_checkbox.setDisabled(True)
        self.tiv_dir = None

        # Set up tiv directory textbox
        self.tiv_dir_textbox = QLineEdit(self)
        self.tiv_dir_textbox.setGeometry(QRect(210, 180, 200, 30))
        self.tiv_dir_textbox.setDisabled(True)

        # Set up threshold textbox
        self.threshold_label = QLabel(self)
        self.threshold_label.setGeometry(QRect(460, 80, 100, 30))
        self.threshold_label.setText('Threshold')
        self.threshold_textbox = QLineEdit(self)
        self.threshold_textbox.setGeometry(QRect(530, 80, 100, 30))
        self.threshold_textbox.setDisabled(True)
        self.threshold_textbox.setText('0.5')
        text = self.threshold_textbox.text()

        # Set up dilate textbox
        self.dilate_label = QLabel(self)
        self.dilate_label.setGeometry(QRect(470, 130, 100, 30))
        self.dilate_label.setText('Dilate')
        self.dilate_textbox = QLineEdit(self)
        self.dilate_textbox.setGeometry(QRect(530, 130, 100, 30))
        self.dilate_textbox.setDisabled(True)
        self.dilate_textbox.setText('0')

        # Set up no_gpu checkbox
        self.no_gpu_checkbox = QCheckBox(self)
        self.no_gpu_checkbox.setGeometry(QRect(530, 180, 100, 30))
        layout = QVBoxLayout()
        label = QLabel('No GPU', self)
        label.setAlignment(Qt.AlignLeft)
        label.setGeometry(QRect(470, 187, 50, 30))
        layout.addWidget(label)
        layout.addWidget(self.no_gpu_checkbox)
        # self.setLayout(layout)
        self.no_gpu_checkbox.stateChanged.connect(self.set_no_gpu)
        self.no_gpu_checkbox.setDisabled(True)
        self.no_gpu = False

        # Set up run button
        self.run_button = QPushButton('Run', self)
        self.run_button.clicked.connect(self.run_processing)
        self.run_button.setGeometry(QRect(250, 300, 200, 30))
        self.run_button.setDisabled(True)

        # Set up progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(QRect(50, 250, 600, 30))

        # Set up status label
        self.status_label = QLabel(self)
        self.status_label.setGeometry(QRect(300, 220, 200, 30))

    def set_no_gpu(self, state):
        if state == Qt.Checked:
            self.no_gpu = True

    def get_input_files(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter('Image Files (*.nii *.nii.gz)')
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            self.input_files = file_dialog.selectedFiles()
            self.status_label.setText(f'Selected {len(self.input_files)} input files')
            self.progress_bar.setValue(0)
            self.brain_dir_checkbox.setDisabled(False)
            self.brain_dir_textbox.setDisabled(False)
            self.mask_dir_checkbox.setDisabled(False)
            self.mask_dir_textbox.setDisabled(False)
            self.tiv_dir_checkbox.setDisabled(False)
            self.tiv_dir_textbox.setDisabled(False)
            self.threshold_textbox.setDisabled(False)
            self.dilate_textbox.setDisabled(False)
            self.no_gpu_checkbox.setDisabled(False)

    def show_brain_dir_dialog(self, state):
        if state == Qt.Checked:
            dir_dialog = QFileDialog()
            dir_dialog.setFileMode(QFileDialog.Directory)
            if dir_dialog.exec_():
                self.brain_dir = dir_dialog.selectedFiles()[0]
                self.brain_dir_textbox.setText(self.brain_dir)
                self.run_button.setDisabled(False)
                self.threshold_textbox.setDisabled(False)
                self.dilate_textbox.setDisabled(False)
                self.brain_dir_checkbox.setDisabled(False)

    def show_mask_dir_dialog(self, state):
        if state == Qt.Checked:
            dir_dialog = QFileDialog()
            dir_dialog.setFileMode(QFileDialog.Directory)
            if dir_dialog.exec_():
                self.mask_dir = dir_dialog.selectedFiles()[0]
                self.mask_dir_textbox.setText(self.mask_dir)
                self.run_button.setDisabled(False)

    def show_tiv_dir_dialog(self, state):
        if state == Qt.Checked:
            dir_dialog = QFileDialog()
            dir_dialog.setFileMode(QFileDialog.Directory)
            if dir_dialog.exec_():
                self.tiv_dir = dir_dialog.selectedFiles()[0]
                self.tiv_dir_textbox.setText(self.tiv_dir)
                self.run_button.setDisabled(False)

    def run_processing(self):
        threshold = float(self.threshold_textbox.text())
        n_dilate = int(self.dilate_textbox.text())
        bet = BrainExtractor(not self.no_gpu)
        for i, file in enumerate(self.input_files):
            filename = str(Path(file).name).split('.')[0]
            brain_path = None if self.brain_dir is None else f'{self.brain_dir}/{Path(file).name}'
            mask_path = None if self.mask_dir is None else f'{self.mask_dir}/{Path(file).name}'
            tiv_path = None if self.tiv_dir is None else f'{self.tiv_dir}/{filename}.csv'
            bet.run(file, brain_path, mask_path, tiv_path, threshold, n_dilate)
            self.progress_bar.setValue(int(100 * (i + 1) / len(self.input_files)))


if __name__ == '__main__':
    run_gui()
