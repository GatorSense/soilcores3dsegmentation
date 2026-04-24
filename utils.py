import os
import nibabel as nib
import numpy as np
import torch
import tempfile
import shutil
import cv2
from skimage.morphology import skeletonize
from scipy.spatial.distance import euclidean
import pandas as pd
import glob
from tqdm import tqdm
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, UNet, DynUNet, SegResNet
from PIL import Image
from typing import List, Optional, Dict, Any
from scipy.stats import pearsonr, spearmanr

# Model configuration mapping
MODEL_CONFIGS = {
    'unet_dataset': ('unet', 'best_metric_modelunet_dataset.pth'),
    'unet_dataset_2_default': ('unet', 'best_metric_modelunet_dataset_2_default.pth'),
    'unet_dataset_2_100k': ('unet', 'best_metric_modelunet_dataset_2_100k.pth'),
    'segresnet_dataset_2_default': ('segresnet', 'best_metric_modelsegresnet_dataset_2_default.pth'),
    'dynunet_dataset_2_100k': ('dynunet', 'best_metric_modeldynunet_dataset_2_100k.pth'),
    'dataset_2adamw_100k_num_heads_2': ('unetr', 'best_metric_modeldataset_2adamw_100k_num_heads_2.pth'),
}

def get_test_transforms():
    """Get the standard test transforms for soilcore data"""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
    ])

def create_model(net_type, device):
    """Create and return a model based on network type"""
    if net_type == 'unet':
        class UNetWithSigmoid(UNet):
            def forward(self, x):
                x = super().forward(x)
                return x

        model = UNetWithSigmoid(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm='instance',
        ).to(device)

    elif net_type == 'unetr':
        class UNETRWithSigmoid(UNETR):
            def forward(self, x):
                x = super().forward(x)
                return x

        model = UNETRWithSigmoid(
            in_channels=1,
            out_channels=1,
            img_size=(96, 96, 16),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name="instance",
            res_block=True,
            dropout_rate=0.2,
        ).to(device)
        
    elif net_type == 'dynunet':
        class DynUNetWithSigmoid(DynUNet):
            def forward(self, x):
                x = super().forward(x)
                return x

        model = DynUNetWithSigmoid(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            kernel_size=[3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2],
            filters=(16, 32, 64, 128, 256),
            norm_name="instance",
        ).to(device)
        
    elif net_type == 'segresnet':
        class SegResNetWithSigmoid(SegResNet):
            def forward(self, x):
                x = super().forward(x)
                return x

        model = SegResNetWithSigmoid(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            init_filters=16,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            norm="instance",
        ).to(device)
        
    else:
        raise ValueError(f"Unknown network type: {net_type}")
    
    return model

def load_model(model_name, models_dir='models'):
    """Load a trained model"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    net_type, model_file = MODEL_CONFIGS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(net_type, device)
    model_path = os.path.join(models_dir, model_file)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Try to load with strict=False to handle missing keys
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"SUCCESS: Model loaded successfully with strict=True")
    except RuntimeError as e:
        print(f"WARNING: Strict loading failed, trying with strict=False...")
        print(f"Error: {e}")
        
        # Try with strict=False
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"WARNING: Missing keys: {len(missing_keys)}")
            for key in missing_keys[:5]:  # Show first 5 missing keys
                print(f"  - {key}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"WARNING: Unexpected keys: {len(unexpected_keys)}")
            for key in unexpected_keys[:5]:  # Show first 5 unexpected keys
                print(f"  - {key}")
            if len(unexpected_keys) > 5:
                print(f"  ... and {len(unexpected_keys) - 5} more")
        
        print(f"SUCCESS: Model loaded with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
    
    model.eval()
    
    return model, device

def prepare_single_file_dataset(input_file, temp_dir):
    """Prepare a single file for segmentation by creating a temporary dataset structure"""
    # Create temporary directory structure
    temp_data_dir = os.path.join(temp_dir, "temp_soilcores")
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # Copy the input file to the temporary directory
    filename = os.path.basename(input_file)
    temp_file_path = os.path.join(temp_data_dir, filename)
    shutil.copy2(input_file, temp_file_path)
    
    # Create a simple dataset JSON for the single file
    dataset_json = {
        "test": [
            {
                "image": temp_file_path
            }
        ]
    }
    
    # Save the dataset JSON
    dataset_json_path = os.path.join(temp_data_dir, "dataset_temp.json")
    import json
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    return temp_data_dir, dataset_json_path

def segment_single_file(input_file, model_name, output_dir=None, threshold_value=None):
    """Segment a single NIfTI file using the specified model"""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare the dataset
        temp_data_dir, dataset_json_path = prepare_single_file_dataset(input_file, temp_dir)
        
        # Load model
        model, device = load_model(model_name)
        
        # Get transforms
        test_transforms = get_test_transforms()
        
        # Load and transform the data
        from monai.data import DataLoader, CacheDataset
        test_files = [{"image": os.path.join(temp_data_dir, os.path.basename(input_file))}]
        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_num=1, cache_rate=1.0, num_workers=1)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        
        # Run inference
        with torch.no_grad():
            for batch in test_loader:
                test_input = batch["image"].to(device)
                test_output = sliding_window_inference(
                    inputs=test_input, 
                    roi_size=(96, 96, 16),
                    sw_batch_size=4,
                    predictor=model
                )
                
                # Apply threshold if specified
                if threshold_value is not None:
                    threshold = AsDiscrete(threshold=threshold_value)
                    output_th = threshold(test_output).cpu()
                else:
                    output_th = test_output.cpu()
                
                # Save the result
                if output_dir is None:
                    output_dir = os.path.join('outputs', model_name)
                os.makedirs(output_dir, exist_ok=True)
                
                base_name = os.path.basename(input_file).split('.')[0]
                output_path = os.path.join(output_dir, f'{base_name}.nii.gz')
                
                img = nib.Nifti1Image(output_th[0,0,:,:,:].numpy(), np.eye(4))
                nib.save(img, output_path)
                
                return output_path

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Root topology analysis functions
def load_and_preprocess_image(path):
    """Load and preprocess image for root topology analysis"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary = binary // 255
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    return binary, skeleton

def compute_distance_transform(binary):
    """Compute distance transform for root width estimation"""
    binary_uint8 = binary.astype(np.uint8) * 255
    dist = cv2.distanceTransform(binary_uint8, cv2.DIST_L2, 3)
    dist = dist * 2.0
    return dist

def get_neighbors(x, y, image):
    """Get 8-connected neighbors"""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                if image[ny, nx] > 0:
                    neighbors.append((nx, ny))
    return neighbors

def extract_segments(skeleton):
    """Extract path-like segments from skeleton"""
    from collections import deque
    visited = np.zeros_like(skeleton, dtype=bool)
    height, width = skeleton.shape
    segments = []
    endpoints = []
    branchpoints = []

    # Find endpoints and branchpoints
    for y in range(height):
        for x in range(width):
            if skeleton[y, x] > 0:
                neighbors = get_neighbors(x, y, skeleton)
                if len(neighbors) == 1:
                    endpoints.append((x, y))
                elif len(neighbors) > 2:
                    branchpoints.append((x, y))

    # Extract segments from endpoints
    for ep in endpoints:
        if visited[ep[1], ep[0]]:
            continue
        path = [ep]
        visited[ep[1], ep[0]] = True
        current = ep
        prev = None

        while True:
            neighbors = get_neighbors(*current, skeleton)
            neighbors = [n for n in neighbors if n != prev and not visited[n[1], n[0]]]

            if len(neighbors) != 1:
                break

            next_point = neighbors[0]
            path.append(next_point)
            visited[next_point[1], next_point[0]] = True
            prev, current = current, next_point

            if next_point in endpoints or next_point in branchpoints:
                break

        if len(path) > 1:
            segments.append(path)

    return segments

def segment_length(segment):
    """Calculate length of a segment"""
    return sum(euclidean(p1, p2) for p1, p2 in zip(segment[:-1], segment[1:]))

def compute_segment_diameter(segment, dist_transform, binary):
    """Compute average diameter of a segment"""
    diameters = []
    for x, y in segment:
        if 0 <= y < dist_transform.shape[0] and 0 <= x < dist_transform.shape[1]:
            diameter = float(dist_transform[y, x]) * 2
            diameters.append(diameter)
    return np.mean(diameters) if diameters else 0.0

def extract_segment_features(segments, dist_transform, binary, pixels_per_range=1, num_ranges=5):
    """Extract features for all segments and group by diameter ranges"""
    data = []
    for i, seg in enumerate(segments):
        length = segment_length(seg)
        diameter = compute_segment_diameter(seg, dist_transform, binary)
        data.append({
            "segment_id": i,
            "length_px": length,
            "diameter_px": diameter
        })
    
    if not data:
        return pd.DataFrame()
        
    features_df = pd.DataFrame(data)
    
    # Group by diameter ranges
    bins = [0]
    for i in range(num_ranges):
        bins.append(bins[-1] + pixels_per_range)
    bins[-1] = np.inf
    
    labels = [
        f"Root Length Diameter Range {i+1} (px)"
        for i in range(num_ranges)
    ]
    
    features_df["diameter_bin"] = pd.cut(features_df["diameter_px"], bins=bins, labels=labels, right=False)
    grouped = features_df.groupby("diameter_bin", observed=True)["length_px"].sum()
    
    result_dict = {label: 0.0 for label in labels}
    result_dict.update(grouped.to_dict())
    
    return pd.DataFrame([result_dict])

def process_images_for_topology(folder_path, pixels_per_range, num_ranges):
    """Process all images in folder for root topology analysis"""
    image_files = glob.glob(os.path.join(folder_path, "*.png")) + \
                 glob.glob(os.path.join(folder_path, "*.jpg")) + \
                 glob.glob(os.path.join(folder_path, "*.tif"))
    
    all_features = []
    
    for img_file in tqdm(image_files, desc="Processing images"):
        folder_name = os.path.basename(os.path.dirname(img_file))
        file_name = os.path.basename(img_file)
        
        try:
            binary, skeleton = load_and_preprocess_image(img_file)
            dist = compute_distance_transform(binary)
            segments = extract_segments(skeleton)
            
            if len(segments) > 0:
                features_df = extract_segment_features(
                    segments, dist, binary, pixels_per_range, num_ranges
                )
                if not features_df.empty:
                    features_df["folder"] = folder_name
                    features_df["file"] = file_name
                    all_features.append(features_df)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    return all_features

def create_root_topology_plot(combined_features, pixels_per_range, num_ranges, save_path=None):
    """Create bar plot of root lengths by diameter ranges"""
    import matplotlib.pyplot as plt
    
    # Sum across all images
    range_columns = [col for col in combined_features.columns if 'Root Length Diameter Range' in col]
    total_lengths = combined_features[range_columns].sum()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    x_pos = np.arange(len(total_lengths))
    bars = ax.bar(x_pos, total_lengths.values, alpha=0.7, color='green')
    
    # Customize plot
    ax.set_xlabel('Diameter Range (pixels)')
    ax.set_ylabel('Total Root Length (pixels)')
    ax.set_title(f'Root Length by Diameter Range\n(Pixel Range: {pixels_per_range}, Num Ranges: {num_ranges})')
    
    # Set x-axis labels
    range_labels = [f'{i*pixels_per_range}-{(i+1)*pixels_per_range}' for i in range(num_ranges)]
    range_labels[-1] = f'{range_labels[-1].split("-")[0]}+'  # Last range is open-ended
    ax.set_xticks(x_pos)
    ax.set_xticklabels(range_labels, rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, total_lengths.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    # Save the plot if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax 

# -----------------------------
# New helpers for CLI pipeline
# -----------------------------

def _derive_base_name_from_path(path: str) -> str:
    """Derive a clean base name from a file path, handling .nii and .nii.gz."""
    base_name = os.path.splitext(os.path.basename(path))[0]
    if base_name.endswith('.nii'):
        base_name = base_name[:-4]
    return base_name

def save_thresholded_slices_from_nifti(
    nifti_path: str,
    output_root_dir: str,
    lower_percent: float = 0.0,
    upper_percent: float = 100.0,
    view: str = "horizontal",
) -> str:
    """Convert a NIfTI volume to thresholded binary PNG slices.

    Thresholds are given in percent (0-100) of the volume min-max range.
    View parameter controls slicing direction:
    - "horizontal": slice along the 6cm dimension (6×6 cm slices)
    - "vertical": slice along the 30cm dimension (30×6 cm slices)
    Returns the folder path where the PNG slices were saved.
    """
    nii_img = nib.load(nifti_path)
    data = nii_img.get_fdata()

    data_min = float(data.min())
    data_max = float(data.max())

    lower_val = data_min + (data_max - data_min) * (lower_percent / 100.0)
    upper_val = data_min + (data_max - data_min) * (upper_percent / 100.0)

    thresholded_data = np.copy(data)
    thresholded_data[thresholded_data < lower_val] = 0
    thresholded_data[thresholded_data > upper_val] = 0

    binary_data = np.where(thresholded_data > 0, 1, 0)

    base_name = _derive_base_name_from_path(nifti_path)
    dataset_dir = os.path.join(output_root_dir, base_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Choose slicing direction based on view parameter
    if view.lower() == "horizontal":
        # Horizontal view: slice along the 6cm dimension (6×6 cm slices)
        data_to_save = np.swapaxes(binary_data, 0, 2)
        for i in range(data_to_save.shape[0]):
            slice_data = data_to_save[i, :, :]
            slice_image = Image.fromarray(np.uint8(slice_data * 255))
            slice_image.save(os.path.join(dataset_dir, f"{base_name}_{i:03}.png"))
    elif view.lower() == "vertical":
        # Vertical view: slice along the 30cm dimension (30×6 cm slices)
        data_to_save = np.swapaxes(binary_data, 0, 2)
        for i in range(data_to_save.shape[1]):
            slice_data = data_to_save[:, i, :]
            slice_image = Image.fromarray(np.uint8(slice_data * 255))
            slice_image.save(os.path.join(dataset_dir, f"{base_name}_{i:03}.png"))
    else:
        raise ValueError(f"Invalid view parameter: {view}. Must be 'horizontal' or 'vertical'.")

    return dataset_dir

def segment_multiple_files(
    input_files: List[str],
    model_name: str,
    output_dir: Optional[str] = None,
    threshold_value: Optional[float] = None,
) -> List[str]:
    """Run segmentation for multiple NIfTI files, returning list of output paths."""
    output_paths: List[str] = []
    for input_path in input_files:
        out_path = segment_single_file(
            input_file=input_path,
            model_name=model_name,
            output_dir=output_dir,
            threshold_value=threshold_value,
        )
        output_paths.append(out_path)
    return output_paths

def create_empty_features_dataframe(pixels_per_range: int, num_ranges: int) -> pd.DataFrame:
    """Create an empty features DataFrame with NA values for when no segments are found.
    
    This should match exactly the structure created by extract_segment_features().
    """
    # Create range columns with the exact same naming convention as extract_segment_features
    range_columns = [
        f"Root Length Diameter Range {i+1} (px)"
        for i in range(num_ranges)
    ]
    
    # Create a single row with NA values for all range columns
    empty_data = {col: [np.nan] for col in range_columns}
    empty_data["folder"] = [""]
    empty_data["file"] = [""]
    
    return pd.DataFrame(empty_data)

def analyze_core_folders(
    selected_folders: List[str],
    pixels_per_range: int,
    num_ranges: int,
    view: str = "horizontal",
) -> Dict[str, Any]:
    """Analyze one or many core folders and generate CSVs/plots like the GUI flow.

    Returns a dict with keys:
      - per_core_csvs: list[str]
      - aggregated_csv: str | None
      - aggregated_plot: str | None
    """
    all_core_features: List[pd.DataFrame] = []
    core_summaries: List[Dict[str, Any]] = []
    saved_files: Dict[str, Any] = {
        "per_core_csvs": [],
        "aggregated_csv": None,
        "aggregated_plot": None,
    }

    for folder_path in selected_folders:
        core_name = os.path.basename(folder_path)
        all_features = process_images_for_topology(folder_path, pixels_per_range, num_ranges)

        parent_dir = os.path.dirname(folder_path)
        core_csv_path = os.path.join(parent_dir, f"{core_name}_root_length_by_diameter_{view}.csv")
        
        if all_features:
            # Normal case: segments found
            combined_features = pd.concat(all_features, ignore_index=True)
            range_columns = [col for col in combined_features.columns if 'Root Length Diameter Range' in col]
            core_summary = combined_features[range_columns].sum()
            core_summary_dict: Dict[str, Any] = {"soil_core": core_name, "total_images": len(all_features)}
            core_summary_dict.update(core_summary.to_dict())
            all_core_features.extend(all_features)
        else:
            # No segments found: create empty DataFrame with NA values
            print(f"WARNING: No segments found for {core_name}, generating CSV with NA values")
            combined_features = create_empty_features_dataframe(pixels_per_range, num_ranges)
            range_columns = [col for col in combined_features.columns if 'Root Length Diameter Range' in col]
            core_summary = combined_features[range_columns].sum()  # This will be all NaN
            core_summary_dict: Dict[str, Any] = {"soil_core": core_name, "total_images": 0}
            core_summary_dict.update(core_summary.to_dict())
            all_core_features.append(combined_features)

        # Save CSV file regardless of whether segments were found
        try:
            combined_features.to_csv(core_csv_path, index=False)
            saved_files["per_core_csvs"].append(core_csv_path)
        except Exception:
            backup_path = os.path.join(os.path.dirname(parent_dir), f"{core_name}_root_length_by_diameter_{view}.csv")
            combined_features.to_csv(backup_path, index=False)
            saved_files["per_core_csvs"].append(backup_path)

        core_summaries.append(core_summary_dict)

    if core_summaries:
        parent_dir = os.path.dirname(selected_folders[0])
        summary_csv_path = os.path.join(parent_dir, f"soil_cores_summary_{view}.csv")
        summary_df = pd.DataFrame(core_summaries)
        summary_df.to_csv(summary_csv_path, index=False)
        saved_files["aggregated_csv"] = summary_csv_path

        # Only create aggregated plot when analyzing multiple cores and we have actual features
        if len(selected_folders) > 1 and any(not df.empty and not df.isna().all().all() for df in all_core_features):
            # Filter out completely empty DataFrames for plotting
            valid_features = [df for df in all_core_features if not df.empty and not df.isna().all().all()]
            if valid_features:
                combined_features = pd.concat(valid_features, ignore_index=True)
                plot_path = os.path.join(parent_dir, f"soil_cores_aggregated_plot_{view}.png")
                fig, _ = create_root_topology_plot(combined_features, pixels_per_range, num_ranges, plot_path)
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass
                saved_files["aggregated_plot"] = plot_path

    return saved_files

def collect_nifti_files_from_dir(nifti_dir: str) -> List[str]:
    """Collect .nii and .nii.gz files from a directory (non-recursive)."""
    paths: List[str] = []
    for ext in ("*.nii", "*.nii.gz"):
        paths.extend(glob.glob(os.path.join(nifti_dir, ext)))
    return sorted(paths)

def find_core_folders_in_parent(parent_dir: str) -> List[str]:
    """Find subfolders containing PNG/JPG/TIF images (soil cores)."""
    subdirs: List[str] = []
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            image_files = glob.glob(os.path.join(item_path, "*.png")) + \
                          glob.glob(os.path.join(item_path, "*.jpg")) + \
                          glob.glob(os.path.join(item_path, "*.tif"))
            if image_files:
                subdirs.append(item_path)
    return subdirs

def run_pipeline_for_niftis(
    nifti_files: List[str],
    model_name: str,
    lower_percent: float,
    upper_percent: float,
    pixels_per_range: int,
    num_ranges: int,
    outputs_dir: str = "outputs",
    view: str = "horizontal",
) -> Dict[str, Any]:
    """High-level end-to-end pipeline: segment → save slices → analyze.

    Returns a dict summarizing output locations with keys:
      - nifti_outputs: list[str]
      - png_root: str
      - analysis_outputs: dict (from analyze_core_folders)
    """
    nifti_outputs = segment_multiple_files(
        nifti_files, model_name=model_name, output_dir=None, threshold_value=None
    )

    net_type, _ = MODEL_CONFIGS[model_name]
    png_root = os.path.join(outputs_dir, f"{net_type}_{model_name}_{view}_png")
    os.makedirs(png_root, exist_ok=True)

    generated_folders: List[str] = []
    for out_path in nifti_outputs:
        folder = save_thresholded_slices_from_nifti(
            nifti_path=out_path,
            output_root_dir=png_root,
            lower_percent=lower_percent,
            upper_percent=upper_percent,
            view=view,
        )
        generated_folders.append(folder)

    analysis_outputs = analyze_core_folders(
        selected_folders=generated_folders,
        pixels_per_range=pixels_per_range,
        num_ranges=num_ranges,
        view=view,
    )

    return {
        "nifti_outputs": nifti_outputs,
        "png_root": png_root,
        "analysis_outputs": analysis_outputs,
    }

# -----------------------------
# Correlation helpers (GT vs summary)
# -----------------------------

def _map_gt_to_prediction_ranges() -> Dict[str, str]:
    gt_columns = [
        'Sum of 0<.L.<=1.000000',
        'Sum of 1.0000000<.L.<=2.0000000',
        'Sum of 2.0000000<.L.<=3.0000000',
        'Sum of 3.0000000<.L.<=4.0000000',
        'Sum of .L.>4.0000000',
    ]
    pred_columns = [
        'Root Length Diameter Range 1 (px)',
        'Root Length Diameter Range 2 (px)',
        'Root Length Diameter Range 3 (px)',
        'Root Length Diameter Range 4 (px)',
        'Root Length Diameter Range 5 (px)',
    ]
    return dict(zip(gt_columns, pred_columns))

def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if np.var(x) == 0 or np.var(y) == 0:
        return 0.0
    try:
        if method == 'pearson':
            r, _ = pearsonr(x, y)
        else:
            r, _ = spearmanr(x, y)
        return float(r)
    except Exception:
        return 0.0

def compute_correlation_matrix(gt_df: pd.DataFrame, pred_df: pd.DataFrame, method: str) -> pd.DataFrame:
    mapping = _map_gt_to_prediction_ranges()
    gt_cols = list(mapping.keys()) + ['Sum of .L.>0.0000000']
    pred_cols = list(mapping.values()) + ['total_images']

    # Ensure only common cores are used
    if 'corename' in gt_df.columns and 'soil_core' not in gt_df.columns:
        gt_df = gt_df.rename(columns={'corename': 'soil_core'})
    common = sorted(set(gt_df['soil_core']).intersection(set(pred_df['soil_core'])))
    gt_sub = gt_df[gt_df['soil_core'].isin(common)].sort_values('soil_core').reset_index(drop=True)
    pred_sub = pred_df[pred_df['soil_core'].isin(common)].sort_values('soil_core').reset_index(drop=True)

    mat = np.zeros((len(gt_cols), len(pred_cols)))
    for i, gcol in enumerate(gt_cols):
        for j, pcol in enumerate(pred_cols):
            gx = gt_sub[gcol].values if gcol in gt_sub.columns else np.zeros(len(gt_sub))
            px = pred_sub[pcol].values if pcol in pred_sub.columns else np.zeros(len(pred_sub))
            mat[i, j] = _safe_corr(gx, px, method)

    return pd.DataFrame(mat, index=gt_cols, columns=pred_cols)

def load_gt_and_summary(gt_csv_path: str, summary_csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_df = pd.read_csv(gt_csv_path)
    pred_df = pd.read_csv(summary_csv_path)
    if 'corename' in gt_df.columns and 'soil_core' not in gt_df.columns:
        gt_df = gt_df.rename(columns={'corename': 'soil_core'})
    return gt_df, pred_df

def _sanitize_thresh(val: float) -> str:
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return f"{val:.2f}".rstrip('0').rstrip('.')

def compute_and_save_correlations(
    gt_csv_path: str,
    summary_csv_path: str,
    save_dir: str,
    lower_percent: float,
    upper_percent: float,
    view: str = "horizontal",
) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    gt_df, pred_df = load_gt_and_summary(gt_csv_path, summary_csv_path)

    pearson_df = compute_correlation_matrix(gt_df, pred_df, method='pearson')
    spearman_df = compute_correlation_matrix(gt_df, pred_df, method='spearman')

    l = _sanitize_thresh(lower_percent)
    u = _sanitize_thresh(upper_percent)
    pearson_path = os.path.join(save_dir, f"correlation_pearson_l{l}_u{u}_{view}.csv")
    spearman_path = os.path.join(save_dir, f"correlation_spearman_l{l}_u{u}_{view}.csv")

    pearson_df.to_csv(pearson_path)
    spearman_df.to_csv(spearman_path)

    return {"pearson": pearson_path, "spearman": spearman_path}