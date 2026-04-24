import sys
import os
import numpy as np
import nibabel as nib
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QSlider, QHBoxLayout, QComboBox, QMessageBox, QSpinBox, QDialog, QListWidget, QTextEdit)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from utils import (segment_single_file, MODEL_CONFIGS, process_images_for_topology, 
                   create_root_topology_plot)
from PIL import Image
import cv2
from skimage.morphology import skeletonize
from scipy.spatial.distance import euclidean
import pandas as pd
import glob
from tqdm import tqdm

MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'

# Map model file to display name and model_name for runsegmentation
MODEL_OPTIONS = [
    ('best_metric_modelunet_dataset.pth', 'UNet (Default)', 'unet_dataset', 'unet'),
    ('best_metric_modelunet_dataset_2_100k.pth', 'UNet (100k)', 'unet_dataset_2_100k', 'unet'),
    ('best_metric_modelsegresnet_dataset_2_default.pth', 'SegResNet', 'segresnet_dataset_2_default', 'segresnet'),
    ('best_metric_modeldynunet_dataset_2_100k.pth', 'DynUNet', 'dynunet_dataset_2_100k', 'dynunet'),
    ('best_metric_modeldataset_2adamw_100k_num_heads_2.pth', 'UNETR (AdamW, 100k, 2 heads)', 'dataset_2adamw_100k_num_heads_2', 'unetr'),
]

class NiftiViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Soilcore NIfTI Viewer')
        self.resize(900, 900)
        self.nii_img = None
        self.data = None
        self.original_data = None  # Store original data for thresholding
        self.slice_axis = 0
        self.current_slice = 0
        self.seg_output_path = None
        self.image_folder_path = None
        self.selected_folders = []  # List to store multiple selected folders
        self.nifti_file_list = []   # List of NIfTI paths for batch processing
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.info_label = QLabel('No file loaded.')
        layout.addWidget(self.info_label)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel('Segmentation Model:'))
        self.model_combo = QComboBox()
        for _, display, _, _ in MODEL_OPTIONS:
            self.model_combo.addItem(display)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        self.canvas = FigureCanvas(Figure(figsize=(4, 8)))
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()

        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton('Load NIfTI File')
        self.load_btn.clicked.connect(self.load_nifti)
        btn_layout.addWidget(self.load_btn)

        self.run_btn = QPushButton('Run Segmentation')
        self.run_btn.clicked.connect(self.run_segmentation)
        self.run_btn.setEnabled(False)
        btn_layout.addWidget(self.run_btn)

        self.save_btn = QPushButton('Save Thresholded Images')
        self.save_btn.clicked.connect(self.save_thresholded_images)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)

        self.axis_slider = QSlider(Qt.Horizontal)
        self.axis_slider.setMinimum(0)
        self.axis_slider.setMaximum(2)
        self.axis_slider.setValue(0)
        self.axis_slider.setTickPosition(QSlider.TicksBelow)
        self.axis_slider.setTickInterval(1)
        self.axis_slider.valueChanged.connect(self.change_axis)
        btn_layout.addWidget(QLabel('Axis:'))
        btn_layout.addWidget(self.axis_slider)

        layout.addLayout(btn_layout)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.update_slice)
        layout.addWidget(self.slice_slider)

        # Threshold controls
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel('Lower Threshold:'))
        self.lower_threshold_slider = QSlider(Qt.Horizontal)
        self.lower_threshold_slider.setMinimum(0)
        self.lower_threshold_slider.setMaximum(100)
        self.lower_threshold_slider.setValue(0)
        self.lower_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.lower_threshold_slider.setTickInterval(10)
        self.lower_threshold_slider.valueChanged.connect(self.apply_thresholds)
        threshold_layout.addWidget(self.lower_threshold_slider)
        
        self.lower_threshold_label = QLabel('0.00')
        threshold_layout.addWidget(self.lower_threshold_label)
        
        threshold_layout.addWidget(QLabel('Upper Threshold:'))
        self.upper_threshold_slider = QSlider(Qt.Horizontal)
        self.upper_threshold_slider.setMinimum(0)
        self.upper_threshold_slider.setMaximum(100)
        self.upper_threshold_slider.setValue(100)
        self.upper_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.upper_threshold_slider.setTickInterval(10)
        self.upper_threshold_slider.valueChanged.connect(self.apply_thresholds)
        threshold_layout.addWidget(self.upper_threshold_slider)
        
        self.upper_threshold_label = QLabel('1.00')
        threshold_layout.addWidget(self.upper_threshold_label)
        
        layout.addLayout(threshold_layout)

        # Batch NIfTI: load list, run model + thresholds, save each as folder
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel('Batch NIfTI:'))
        self.load_nifti_list_btn = QPushButton('Load NIfTI Files...')
        self.load_nifti_list_btn.clicked.connect(self.load_nifti_list)
        batch_layout.addWidget(self.load_nifti_list_btn)
        self.batch_process_btn = QPushButton('Batch: Run model + thresholds, save as folders')
        self.batch_process_btn.clicked.connect(self.batch_process_and_save)
        self.batch_process_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_process_btn)
        layout.addLayout(batch_layout)

        batch_list_layout = QVBoxLayout()
        batch_list_layout.addWidget(QLabel('NIfTI files for batch:'))
        batch_list_control = QHBoxLayout()
        self.nifti_list_widget = QListWidget()
        self.nifti_list_widget.setMaximumHeight(80)
        batch_list_control.addWidget(self.nifti_list_widget)
        self.remove_nifti_btn = QPushButton('Remove Selected')
        self.remove_nifti_btn.clicked.connect(self.remove_selected_nifti)
        self.remove_nifti_btn.setEnabled(False)
        batch_list_control.addWidget(self.remove_nifti_btn)
        batch_list_layout.addLayout(batch_list_control)
        self.clear_nifti_list_btn = QPushButton('Clear All')
        self.clear_nifti_list_btn.clicked.connect(self.clear_nifti_list)
        self.clear_nifti_list_btn.setEnabled(False)
        batch_list_layout.addWidget(self.clear_nifti_list_btn)
        layout.addLayout(batch_list_layout)

        # Root topology analysis controls
        topology_layout = QHBoxLayout()
        topology_layout.addWidget(QLabel('Root Topology Analysis:'))
        
        self.select_folder_btn = QPushButton('Select Single Folder')
        self.select_folder_btn.clicked.connect(self.select_single_folder)
        topology_layout.addWidget(self.select_folder_btn)
        
        self.select_multiple_folders_btn = QPushButton('Select Multiple Folders')
        self.select_multiple_folders_btn.clicked.connect(self.select_multiple_folders)
        topology_layout.addWidget(self.select_multiple_folders_btn)
        
        topology_layout.addWidget(QLabel('Pixel Range:'))
        self.pixel_range_spin = QSpinBox()
        self.pixel_range_spin.setMinimum(1)
        self.pixel_range_spin.setMaximum(10)
        self.pixel_range_spin.setValue(2)
        topology_layout.addWidget(self.pixel_range_spin)
        
        topology_layout.addWidget(QLabel('Num Ranges:'))
        self.num_ranges_spin = QSpinBox()
        self.num_ranges_spin.setMinimum(3)
        self.num_ranges_spin.setMaximum(10)
        self.num_ranges_spin.setValue(5)
        topology_layout.addWidget(self.num_ranges_spin)
        
        self.analyze_btn = QPushButton('Analyze Root Topology')
        self.analyze_btn.clicked.connect(self.analyze_root_topology)
        self.analyze_btn.setEnabled(False)
        topology_layout.addWidget(self.analyze_btn)
        
        layout.addLayout(topology_layout)

        # Selected folders display
        folders_layout = QVBoxLayout()
        folders_layout.addWidget(QLabel('Selected Soil Core Folders:'))
        
        # Add horizontal layout for list and remove button
        folders_control_layout = QHBoxLayout()
        
        self.folders_list = QListWidget()
        self.folders_list.setMaximumHeight(100)
        folders_control_layout.addWidget(self.folders_list)
        
        # Remove button
        self.remove_folder_btn = QPushButton('Remove Selected')
        self.remove_folder_btn.clicked.connect(self.remove_selected_folder)
        self.remove_folder_btn.setEnabled(False)
        folders_control_layout.addWidget(self.remove_folder_btn)
        
        folders_layout.addLayout(folders_control_layout)
        
        # Clear all button
        self.clear_all_btn = QPushButton('Clear All')
        self.clear_all_btn.clicked.connect(self.clear_all_folders)
        self.clear_all_btn.setEnabled(False)
        folders_layout.addWidget(self.clear_all_btn)
        
        layout.addLayout(folders_layout)

        self.setLayout(layout)

    def select_single_folder(self):
        """Select a single folder containing 2D image files"""
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Image Folder')
        if folder_path:
            self.selected_folders = [folder_path]
            self.image_folder_path = folder_path  # Keep for backward compatibility
            self.update_folders_list()
            self.check_and_enable_analysis()

    def select_multiple_folders(self):
        """Select multiple folders, each representing a soil core"""
        # For multiple folder selection, we'll use a dialog approach
        # Since QFileDialog doesn't support multiple folder selection directly,
        # we'll use a custom approach
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Parent Directory Containing Soil Core Folders')
        if folder_path:
            # Get all subdirectories that contain image files
            import glob
            subdirs = []
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    # Check if this directory contains image files
                    image_files = glob.glob(os.path.join(item_path, "*.png")) + \
                                 glob.glob(os.path.join(item_path, "*.jpg")) + \
                                 glob.glob(os.path.join(item_path, "*.tif"))
                    if image_files:
                        subdirs.append(item_path)
            
            if subdirs:
                self.selected_folders = subdirs
                self.update_folders_list()
                self.check_and_enable_analysis()
                QMessageBox.information(self, 'Folders Selected', 
                                      f'Found {len(subdirs)} soil core folders with image files.')
            else:
                QMessageBox.warning(self, 'No Valid Folders', 
                                  'No subdirectories with image files found in the selected directory.')

    def update_folders_list(self):
        """Update the folders list display"""
        self.folders_list.clear()
        for folder in self.selected_folders:
            folder_name = os.path.basename(folder)
            self.folders_list.addItem(f"{folder_name} ({folder})")
        
        # Enable/disable buttons based on list state
        self.remove_folder_btn.setEnabled(len(self.selected_folders) > 0)
        self.clear_all_btn.setEnabled(len(self.selected_folders) > 0)

    def check_and_enable_analysis(self):
        """Check if we have valid folders and enable analysis button"""
        if self.selected_folders:
            self.analyze_btn.setEnabled(True)
        else:
            self.analyze_btn.setEnabled(False)

    def load_nifti(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open NIfTI file', '', 'NIfTI Files (*.nii *.nii.gz)')
        if file_path:
            self.nii_img = nib.load(file_path)
            self.data = self.nii_img.get_fdata()
            self.original_data = self.data.copy()  # Store original data
            self.slice_axis = 0
            self.current_slice = self.data.shape[self.slice_axis] // 2
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(self.data.shape[self.slice_axis] - 1)
            self.slice_slider.setValue(self.current_slice)
            self.update_info(file_path)
            self.show_slice()
            self.input_path = file_path
            self.run_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.seg_output_path = None
            # Reset thresholds when loading new file
            self.reset_thresholds()

    def load_nifti_list(self):
        """Load multiple NIfTI files for batch processing."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Select NIfTI files', '', 'NIfTI Files (*.nii *.nii.gz);;All Files (*)'
        )
        if paths:
            for p in paths:
                if p not in self.nifti_file_list:
                    self.nifti_file_list.append(p)
            self.update_nifti_list()

    def update_nifti_list(self):
        """Refresh the NIfTI list widget and batch button state."""
        self.nifti_list_widget.clear()
        for path in self.nifti_file_list:
            self.nifti_list_widget.addItem(os.path.basename(path))
        self.remove_nifti_btn.setEnabled(len(self.nifti_file_list) > 0)
        self.clear_nifti_list_btn.setEnabled(len(self.nifti_file_list) > 0)
        self.batch_process_btn.setEnabled(len(self.nifti_file_list) > 0)

    def remove_selected_nifti(self):
        """Remove the selected NIfTI path from the list."""
        row = self.nifti_list_widget.currentRow()
        if row >= 0 and row < len(self.nifti_file_list):
            self.nifti_file_list.pop(row)
            self.update_nifti_list()

    def clear_nifti_list(self):
        """Clear all NIfTI paths from the batch list."""
        self.nifti_file_list.clear()
        self.update_nifti_list()

    def _get_threshold_values_for_batch(self):
        """Get absolute lower/upper threshold values matching the current view (sliders + loaded data range)."""
        if self.original_data is not None:
            data_min = float(self.original_data.min())
            data_max = float(self.original_data.max())
            lower_val = data_min + (data_max - data_min) * self.lower_threshold_slider.value() / 100.0
            upper_val = data_min + (data_max - data_min) * self.upper_threshold_slider.value() / 100.0
            return lower_val, upper_val, True  # use_absolute
        return None, None, False

    def batch_process_and_save(self):
        """For each NIfTI in the list: run segmentation, apply thresholds, save as a folder of PNGs."""
        if not self.nifti_file_list:
            QMessageBox.warning(self, 'No Files', 'Load at least one NIfTI file for batch processing.')
            return
        output_dir = QFileDialog.getExistingDirectory(self, 'Select output base directory')
        if not output_dir:
            return
        # Use the same threshold values as the current view (from loaded data + sliders)
        lower_val_abs, upper_val_abs, use_view_thresholds = self._get_threshold_values_for_batch()
        if not use_view_thresholds:
            reply = QMessageBox.question(
                self, 'No loaded volume',
                'No volume is loaded, so threshold values cannot be taken from the current view.\n'
                'Use slider percentages (0–100%) of each file\'s range instead?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        model_idx = self.model_combo.currentIndex()
        _, _, model_name, _ = MODEL_OPTIONS[model_idx]
        lower_pct = self.lower_threshold_slider.value()
        upper_pct = self.upper_threshold_slider.value()
        failed = []
        saved_folders = []
        try:
            self.batch_process_btn.setEnabled(False)
            self.batch_process_btn.setText('Processing...')
            QApplication.processEvents()
            for i, input_path in enumerate(self.nifti_file_list):
                base_name = os.path.basename(input_path)
                if base_name.lower().endswith('.nii.gz'):
                    base_name = base_name[:-7]
                elif base_name.lower().endswith('.nii'):
                    base_name = base_name[:-4]
                file_output_dir = os.path.join(output_dir, base_name)
                os.makedirs(file_output_dir, exist_ok=True)
                try:
                    seg_path = segment_single_file(
                        input_file=input_path,
                        model_name=model_name,
                        output_dir=file_output_dir,
                        threshold_value=None,
                    )
                    seg_nii = nib.load(seg_path)
                    seg_data = seg_nii.get_fdata()
                    if use_view_thresholds:
                        lower_val = lower_val_abs
                        upper_val = upper_val_abs
                    else:
                        data_min = float(seg_data.min())
                        data_max = float(seg_data.max())
                        lower_val = data_min + (data_max - data_min) * lower_pct / 100.0
                        upper_val = data_min + (data_max - data_min) * upper_pct / 100.0
                    thresholded = np.copy(seg_data)
                    thresholded[thresholded < lower_val] = 0
                    thresholded[thresholded > upper_val] = 0
                    binary_data = np.where(thresholded > 0, 1, 0)
                    data_to_save = np.swapaxes(binary_data, 0, 2)
                    for j in range(data_to_save.shape[1]):
                        slice_data = data_to_save[:, j, :]
                        slice_image = Image.fromarray(np.uint8(slice_data * 255))
                        slice_image.save(os.path.join(file_output_dir, f"{base_name}_{j:03d}.png"))
                    saved_folders.append(file_output_dir)
                except Exception as e:
                    failed.append((os.path.basename(input_path), str(e)))
            msg = f'Saved {len(saved_folders)} folder(s).'
            if use_view_thresholds:
                msg += f'\nThresholds (from current view): {lower_val_abs:.3f} – {upper_val_abs:.3f}'
            if failed:
                msg += '\n\nFailed:\n' + '\n'.join(f'{n}: {e}' for n, e in failed)
            QMessageBox.information(self, 'Batch complete', msg)
        except Exception as e:
            QMessageBox.critical(self, 'Batch error', str(e))
        finally:
            self.batch_process_btn.setEnabled(True)
            self.batch_process_btn.setText('Batch: Run model + thresholds, save as folders')

    def reset_thresholds(self):
        """Reset threshold sliders to default values"""
        if self.original_data is not None:
            data_min = self.original_data.min()
            data_max = self.original_data.max()
            self.lower_threshold_slider.setValue(0)
            self.upper_threshold_slider.setValue(100)
            self.lower_threshold_label.setText(f'{data_min:.2f}')
            self.upper_threshold_label.setText(f'{data_max:.2f}')

    def apply_thresholds(self):
        """Apply lower and upper thresholds to the data"""
        if self.original_data is None:
            return
            
        data_min = self.original_data.min()
        data_max = self.original_data.max()
        
        # Convert slider values (0-100) to actual data range
        lower_val = data_min + (data_max - data_min) * self.lower_threshold_slider.value() / 100
        upper_val = data_min + (data_max - data_min) * self.upper_threshold_slider.value() / 100
        
        # Update labels
        self.lower_threshold_label.setText(f'{lower_val:.2f}')
        self.upper_threshold_label.setText(f'{upper_val:.2f}')
        
        # Apply thresholds
        self.data = np.copy(self.original_data)
        self.data[self.data < lower_val] = 0
        self.data[self.data > upper_val] = 0
        
        # Update display
        self.show_slice()

    def update_info(self, file_path):
        info = f"Loaded: {os.path.basename(file_path)}\nShape: {self.data.shape}\nMin: {self.data.min():.2f}, Max: {self.data.max():.2f}"
        self.info_label.setText(info)

    def change_axis(self, value):
        self.slice_axis = value
        if self.data is not None:
            self.slice_slider.setMaximum(self.data.shape[self.slice_axis] - 1)
            self.current_slice = self.data.shape[self.slice_axis] // 2
            self.slice_slider.setValue(self.current_slice)
            self.show_slice()

    def update_slice(self, value):
        self.current_slice = value
        self.show_slice()

    def show_slice(self):
        if self.data is not None:
            self.ax.clear()
            if self.slice_axis == 0:
                img = self.data[self.current_slice, :, :]
            elif self.slice_axis == 1:
                img = self.data[:, self.current_slice, :]
            else:
                img = self.data[:, :, self.current_slice]
            self.ax.imshow(img.T, cmap='gray', origin='lower')
            self.ax.set_title(f'Slice {self.current_slice} (Axis {self.slice_axis})')
            self.canvas.draw()

    def save_thresholded_images(self):
        """Save the current thresholded data as 2D PNG images"""
        if self.original_data is None:
            QMessageBox.warning(self, 'No Data', 'Please load a NIfTI file first.')
            return
        
        # Get output directory from user
        output_dir = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if not output_dir:
            return
        
        try:
            # Apply current thresholds to get the data to save
            data_min = self.original_data.min()
            data_max = self.original_data.max()
            
            lower_val = data_min + (data_max - data_min) * self.lower_threshold_slider.value() / 100
            upper_val = data_min + (data_max - data_min) * self.upper_threshold_slider.value() / 100
            
            # Apply thresholds to create binary data
            thresholded_data = np.copy(self.original_data)
            thresholded_data[thresholded_data < lower_val] = 0
            thresholded_data[thresholded_data > upper_val] = 0
            
            # Binarize: anything above 0 becomes 1
            binary_data = np.where(thresholded_data > 0, 1, 0)
            
            # Get base filename without extension
            if hasattr(self, 'input_path') and self.input_path:
                base_name = os.path.splitext(os.path.basename(self.input_path))[0]
                if base_name.endswith('.nii'):
                    base_name = base_name[:-4]
            else:
                base_name = "thresholded_data"
            
            # Create subdirectory for this dataset
            dataset_dir = os.path.join(output_dir, base_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Save 2D slices as PNG images
            # Assuming the data is in format (depth, height, width) like in gendata4RV.py
            # We'll save slices along the first axis (depth)
            data_to_save = np.swapaxes(binary_data, 0, 2)  # Swap axes like in gendata4RV.py
            
            for i in range(data_to_save.shape[1]):
                slice_data = data_to_save[:, i, :]
                # Convert to uint8 and scale to 0-255
                slice_image = Image.fromarray(np.uint8(slice_data * 255))
                slice_image.save(os.path.join(dataset_dir, f"{base_name}_{i:03}.png"))
            
            QMessageBox.information(self, 'Save Complete', 
                                  f'Saved {data_to_save.shape[1]} thresholded images to:\n{dataset_dir}\n\n'
                                  f'Threshold range: {lower_val:.2f} - {upper_val:.2f}')
            
        except Exception as e:
            QMessageBox.critical(self, 'Save Error', f'Error saving images: {str(e)}')

    def analyze_root_topology(self):
        """Analyze root topology from 2D images and generate CSV with bar plot"""
        if not self.selected_folders:
            QMessageBox.warning(self, 'No Folders', 'Please select at least one folder first.')
            return
        
        try:
            # Disable button during processing
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText('Processing...')
            QApplication.processEvents()
            
            # Get parameters
            pixels_per_range = self.pixel_range_spin.value()
            num_ranges = self.num_ranges_spin.value()
            
            # Process each folder (soil core)
            all_core_features = []
            core_summaries = []
            saved_files = []  # Track saved files for confirmation
            
            for folder_path in self.selected_folders:
                core_name = os.path.basename(folder_path)
                print(f"Processing soil core: {core_name}")
                
                # Process images for this core
                all_features = process_images_for_topology(
                    folder_path, pixels_per_range, num_ranges
                )
                
                if all_features:
                    # Generate individual CSV for this core - save in parent directory
                    parent_dir = os.path.dirname(folder_path)
                    core_csv_path = os.path.join(parent_dir, f"{core_name}_root_length_by_diameter.csv")
                    import pandas as pd
                    combined_features = pd.concat(all_features, ignore_index=True)
                    
                    try:
                        combined_features.to_csv(core_csv_path, index=False)
                        saved_files.append(core_csv_path)
                        print(f"✓ Saved individual CSV: {core_csv_path}")
                    except Exception as csv_error:
                        print(f"⚠ Error saving CSV for {core_name}: {csv_error}")
                        # Try to save in a different location
                        try:
                            backup_path = os.path.join(os.path.dirname(parent_dir), f"{core_name}_root_length_by_diameter.csv")
                            combined_features.to_csv(backup_path, index=False)
                            saved_files.append(backup_path)
                            print(f"✓ Saved backup CSV: {backup_path}")
                        except Exception as backup_error:
                            print(f"⚠ Failed to save backup CSV for {core_name}: {backup_error}")
                    
                    # Create summary for this core
                    range_columns = [col for col in combined_features.columns if 'Root Length Diameter Range' in col]
                    core_summary = combined_features[range_columns].sum()
                    core_summary_dict = {
                        'soil_core': core_name,
                        'total_images': len(all_features)
                    }
                    core_summary_dict.update(core_summary.to_dict())
                    core_summaries.append(core_summary_dict)
                    
                    all_core_features.extend(all_features)
                    
                    print(f"✓ Processed {core_name}: {len(all_features)} images")
                else:
                    print(f"⚠ No valid segments found in {core_name}")
            
            if all_core_features:
                # Generate summary CSV with all cores
                if len(self.selected_folders) > 1:
                    # Create summary directory in the first folder's parent
                    parent_dir = os.path.dirname(self.selected_folders[0])
                    summary_csv_path = os.path.join(parent_dir, "soil_cores_summary.csv")
                    
                    try:
                        summary_df = pd.DataFrame(core_summaries)
                        summary_df.to_csv(summary_csv_path, index=False)
                        saved_files.append(summary_csv_path)
                        print(f"✓ Saved summary CSV: {summary_csv_path}")
                    except Exception as summary_error:
                        print(f"⚠ Error saving summary CSV: {summary_error}")
                    
                    # Create aggregated plot
                    combined_features = pd.concat(all_core_features, ignore_index=True)
                    plot_path = os.path.join(parent_dir, "soil_cores_aggregated_plot.png")
                    try:
                        fig, ax = create_root_topology_plot(combined_features, pixels_per_range, num_ranges, plot_path)
                        saved_files.append(plot_path)
                        print(f"✓ Saved aggregated plot: {plot_path}")
                    except Exception as plot_error:
                        print(f"⚠ Error saving aggregated plot: {plot_error}")
                    
                    # Show the plot in a new window
                    try:
                        plot_dialog = QDialog(self)
                        plot_dialog.setWindowTitle('Soil Cores Root Topology Analysis Results')
                        plot_dialog.resize(800, 600)
                        
                        from PyQt5.QtWidgets import QVBoxLayout
                        plot_layout = QVBoxLayout()
                        plot_canvas = FigureCanvas(fig)
                        plot_layout.addWidget(plot_canvas)
                        plot_dialog.setLayout(plot_layout)
                        
                        plot_dialog.show()
                    except Exception as dialog_error:
                        print(f"⚠ Error showing plot dialog: {dialog_error}")
                    
                    # Create detailed completion message
                    completion_message = f'Root topology analysis complete!\n\n'
                    completion_message += f'Files saved:\n'
                    for file_path in saved_files:
                        completion_message += f'• {os.path.basename(file_path)}\n'
                    
                    QMessageBox.information(self, 'Analysis Complete', completion_message)
                else:
                    # Single folder case - use existing logic
                    folder_path = self.selected_folders[0]
                    output_csv = os.path.join(folder_path, "..", f"{core_name}_root_length_by_diameter.csv")
                    combined_features = pd.concat(all_core_features, ignore_index=True)
                    
                    try:
                        combined_features.to_csv(output_csv, index=False)
                        saved_files.append(output_csv)
                        print(f"✓ Saved single CSV: {output_csv}")
                    except Exception as csv_error:
                        print(f"⚠ Error saving single CSV: {csv_error}")
                    
                    # Create and display bar plot using utils function
                    plot_path = os.path.join(folder_path, "..", f"{core_name}_root_topology_plot.png")
                    try:
                        fig, ax = create_root_topology_plot(combined_features, pixels_per_range, num_ranges, plot_path)
                        saved_files.append(plot_path)
                        print(f"✓ Saved single plot: {plot_path}")
                    except Exception as plot_error:
                        print(f"⚠ Error saving single plot: {plot_error}")
                    
                    # Show the plot in a new window
                    try:
                        plot_dialog = QDialog(self)
                        plot_dialog.setWindowTitle('Root Topology Analysis Results')
                        plot_dialog.resize(800, 600)
                        
                        from PyQt5.QtWidgets import QVBoxLayout
                        plot_layout = QVBoxLayout()
                        plot_canvas = FigureCanvas(fig)
                        plot_layout.addWidget(plot_canvas)
                        plot_dialog.setLayout(plot_layout)
                        
                        plot_dialog.show()
                    except Exception as dialog_error:
                        print(f"⚠ Error showing plot dialog: {dialog_error}")
                    
                    QMessageBox.information(self, 'Analysis Complete', 
                                          f'Root topology analysis complete!\nCSV saved to: {output_csv}')
            else:
                QMessageBox.warning(self, 'No Data', 'No valid root segments found in any of the selected folders.')
                
        except Exception as e:
            QMessageBox.critical(self, 'Analysis Error', f'Error during analysis: {str(e)}')
        finally:
            # Re-enable button
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText('Analyze Root Topology')

    def run_segmentation(self):
        if not hasattr(self, 'input_path') or not self.input_path:
            QMessageBox.warning(self, 'No Input', 'Please load a NIfTI file first.')
            return
        
        model_idx = self.model_combo.currentIndex()
        _, _, model_name, _ = MODEL_OPTIONS[model_idx]
        
        try:
            # Disable the run button during processing
            self.run_btn.setEnabled(False)
            self.run_btn.setText('Processing...')
            QApplication.processEvents()  # Update the GUI
            
            # Run segmentation using utils function
            output_path = segment_single_file(
                input_file=self.input_path,
                model_name=model_name,
                output_dir=None,  # Will use default outputs/model_name
                threshold_value=None
            )
            
            # Load and display the result
            self.nii_img = nib.load(output_path)
            self.data = self.nii_img.get_fdata()
            self.original_data = self.data.copy()  # Store original segmented data
            self.slice_axis = 0
            self.current_slice = self.data.shape[self.slice_axis] // 2
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(self.data.shape[self.slice_axis] - 1)
            self.slice_slider.setValue(self.current_slice)
            self.update_info(output_path)
            self.show_slice()
            self.seg_output_path = output_path
            
            # Reset thresholds for the new segmented data
            self.reset_thresholds()
            
            # Enable save button after segmentation
            self.save_btn.setEnabled(True)
            
            QMessageBox.information(self, 'Segmentation Complete', f'Segmentation complete. Output saved to {output_path}')
            
        except Exception as e:
            QMessageBox.critical(self, 'Segmentation Error', f'Error during segmentation: {str(e)}')
        finally:
            # Re-enable the run button
            self.run_btn.setEnabled(True)
            self.run_btn.setText('Run Segmentation')

    def remove_selected_folder(self):
        """Remove the selected folder from the list"""
        selected_indexes = self.folders_list.selectedIndexes()
        if selected_indexes:
            selected_folder = self.folders_list.item(selected_indexes[0].row()).text()
            folder_path = selected_folder.split('(')[1].strip(')')
            self.selected_folders.remove(folder_path)
            self.update_folders_list()
            self.check_and_enable_analysis()

    def clear_all_folders(self):
        """Clear all folders from the list"""
        self.selected_folders = []
        self.update_folders_list()
        self.check_and_enable_analysis()

def main():
    app = QApplication(sys.argv)
    viewer = NiftiViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 