import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from project1 import run_analysis

# === CONFIGURATION ===
BASE_DATA_FOLDER = r"D:\AA BK ASUS\one drive\ML-rehab\portfolio\project1-stroke-subgroups\robot\data_new"
OUTPUT_FOLDER = BASE_DATA_FOLDER  # Save preprocessed files here
TARGET_NUM_FRAMES = 100  # Resample all sequences to this length
NUM_JOINTS = 25
NUM_COORDS = 3
SPINE_BASE_IDX = 0  # First joint index (usually spine base)
# Patients and healthy participants
PATIENTS = [f"P{str(i).zfill(2)}" for i in range(1, 10)]
HEALTHY = [f"H{str(i).zfill(2)}" for i in range(1, 11)]  # adjust number of healthy participants

MOVEMENT_VARIANTS_PATIENTS = { 
    "Sd2Sd_Bck_L": ["L", "L_1", "L_2"], 
    "Sd2Sd_Bck_R": ["R", "R_1", "R_2"], 
    "Fwr_Bck_L": ["L", "L_1", "L_2"], 
    "Fwr_Bck_R": ["R", "R_1", "R_2"] 
}

MOVEMENT_VARIANTS_HEALTHY = {
    "Sd2Sd_Bck_L": ["L"], 
    "Sd2Sd_Bck_R": ["R"], 
    "Fwr_Bck_L": ["L"], 
    "Fwr_Bck_R": ["R"]
}

def load_joint_positions(full_path):
    """
    Load joint data from a .csv file (N x 3), where N must be a multiple of 25.
    Each frame consists of 25 joints with (x, y, z) coordinates.
    Returns a (num_frames, 25, 3) array.
    """
    try:
        df = pd.read_csv(full_path, header=None)

        if df.shape[1] != 3 or df.shape[0] % 25 != 0:
            raise ValueError(f"Expected (N x 3) with N % 25 == 0. Got: {df.shape}")

        data = df.values.astype(np.float32)
        num_frames = df.shape[0] // 25
        reshaped = data.reshape(num_frames, 25, 3)
        return reshaped  # shape: (frames, 25, 3)

    except Exception as e:
        return None

def smooth_sequence(sequence, sigma=None):
    """
    Apply Gaussian smoothing along the frame axis.
    If sigma is None, automatically compute it relative to the sequence length.
    """
    try:
        # Default: scale sigma relative to sequence length
        if sigma is None:
            # Example: target smoothing over ~1% of sequence length
            sigma = max(1, sequence.shape[0] * 0.005)

        return gaussian_filter1d(sequence, sigma=sigma, axis=0)
    
    except Exception as e:
        print(f"❌ Error during smoothing: {e}")
        return sequence

def resample_sequence(sequence, target_length=TARGET_NUM_FRAMES):
    """
    Resample the sequence along the time axis using linear interpolation.
    Logs any out-of-range errors but continues processing.
    """
    try:
        original_length = sequence.shape[0]
        new_indices = np.linspace(0, original_length - 1, target_length)
        resampled = np.zeros((target_length, NUM_JOINTS, NUM_COORDS))

        for joint in range(NUM_JOINTS):
            for coord in range(NUM_COORDS):
                try:
                    interp_func = interp1d(
                        np.arange(original_length),
                        sequence[:, joint, coord],
                        kind='linear',
                        fill_value=None  # raise error if out-of-range
                    )
                    resampled[:, joint, coord] = interp_func(new_indices)

                except ValueError as e:
                    print(f"⚠️ Interpolation error for joint {joint}, coord {coord}: {e}")
                    # fallback: clip new indices to valid range and interpolate
                    safe_indices = np.clip(new_indices, 0, original_length - 1)
                    resampled[:, joint, coord] = interp1d(
                        np.arange(original_length),
                        sequence[:, joint, coord],
                        kind='linear',
                        fill_value="extrapolate"  # safe fallback for clipping
                    )(safe_indices)

        return resampled

    except Exception as e:
        print(f"❌ Error during resampling: {e}")
        return sequence

def preprocess_sequence(raw_sequence):
    """
    Apply full preprocessing: centering, smoothing, resampling, and optional mirroring.
    """
    try:
        # Center on spine base (usually joint 0)
        centered = raw_sequence - raw_sequence[:, SPINE_BASE_IDX:SPINE_BASE_IDX+1, :]

        # Mirror left-right
        left_indices  = [4, 5, 6, 7, 12, 13, 14, 15, 21, 22]
        right_indices = [8, 9, 10, 11, 16, 17, 18, 19, 23, 24]
        centered[:, :, 0] *= -1
        centered[:, left_indices + right_indices, :] = centered[:, right_indices + left_indices, :]
        
        # Swap Y (up) and Z (forward) to align with camera view
        centered = centered[:, :, [0, 2, 1]]  # X stays, Y <-> Z

        # Smooth and resample
        smoothed = smooth_sequence(centered, sigma=None)
        resampled = resample_sequence(smoothed, TARGET_NUM_FRAMES)
        return resampled
    
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return None

def generate_sample_labels(movement_key, group=None):
    """
    Build dict of participant_id -> list of folder names.

    Assumes movement_key and variants already follow the folder naming style with underscores.
    Example:
        movement_key = "Sd2Sd_Bck_L"
        variants = ["L", "L_1", "L_2"]

    Produces:
        ["Rch_Sd2Sd_Bck_L", "Rch_Sd2Sd_Bck_L_1", "Rch_Sd2Sd_Bck_L_2"]
    """
    sample_labels = {}

    if group == "patients":
        variants = MOVEMENT_VARIANTS_PATIENTS[movement_key]
        participants = PATIENTS
    else:
        variants = MOVEMENT_VARIANTS_HEALTHY[movement_key]
        participants = HEALTHY

    # Side is always the last part of the key (L or R)
    side = movement_key.split('_')[-1]

    for pid in participants:
        sample_labels[pid] = []
        for variant in variants:
            if variant == side:
                # Base folder (no suffix)
                folder_name = f"Rch_{movement_key}"
            elif variant.startswith(f"{side}_"):
                # Variant with suffix (e.g. "L_1" -> "_1")
                suffix = variant.split('_', 1)[1]
                folder_name = f"Rch_{movement_key}_{suffix}"
            else:
                # Catch-all for other cases (shouldn’t really happen)
                folder_name = f"Rch_{movement_key}_{variant}"

            sample_labels[pid].append(folder_name)

    return sample_labels

def load_and_preprocess_all_samples(movement_key, sample_labels, group=None):
    """
    Load and preprocess all CSV files for a movement group.

    Parameters:
        movement_key (str): movement variant key (e.g., 'sd2sd-L')
        sample_labels (dict): participant_id -> list of folder names
        group (str): 'patients' or 'healthy'

    Returns:
        np.ndarray or None: stacked preprocessed data array or None on failure
    """
    flat_labels = [f"{pid}_{folder}" for pid, folders in sample_labels.items() for folder in folders]
    
    processed_samples = []

    for label in flat_labels:
        parts = label.split("_", 1)
        participant = parts[0]          # e.g., "P06" or "H01"
        movement_folder = parts[1]      # e.g., "Rch_Fwr_Bck_R_2"

        full_path = os.path.join(
            BASE_DATA_FOLDER,
            participant,
            movement_folder,
            "Joint_Positions.csv"
        )
        raw = load_joint_positions(full_path)
        if raw is None:
            continue

        processed = preprocess_sequence(raw)
        if processed is not None:
            processed_samples.append(processed)
        else:
            print(f"⚠️ Skipping {label} due to preprocessing error.")

    if not processed_samples:
        return None

    data_array = np.stack(processed_samples, axis=0)
    # 👇 save with group in filename
    output_filename = f"preprocessed_kin-{movement_key}_{group}.npy"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    try:
        np.save(output_path, data_array)
        print(f"✅ Saved {group} preprocessed data to {output_path}")
    except Exception as e:
        print(f"❌ Failed to save .npy file: {e}")
        return None
    return data_array

def main():
    # === Patients ===
    try:
        for movement_key in MOVEMENT_VARIANTS_PATIENTS.keys():
            sample_labels = generate_sample_labels(movement_key, group="patients")
            data = load_and_preprocess_all_samples(movement_key, sample_labels, group="patients")

            if data is None:
                print(f"❌ Skipping analysis for {movement_key} (patients) due to missing data.\n")
                continue

            # Step 2: Run clustering and visualization
            cluster_labels, sample_indices = run_analysis(
                movement_key,
                base_folder=BASE_DATA_FOLDER,
                sample_labels=sample_labels,
                group="patients"  # or "healthy"
            )
            print(f"✅ Analysis done for patients - {movement_key}\n")

    except Exception as e:
        print(f"❌ Error processing patients: {e}")

    # === Healthy controls ===
    try:
        for movement_key in MOVEMENT_VARIANTS_HEALTHY.keys():
            sample_labels = generate_sample_labels(movement_key, group="healthy")
            data = load_and_preprocess_all_samples(movement_key, sample_labels, group="healthy")

            if data is None:
                print(f"❌ Skipping analysis for {movement_key} (healthy) due to missing data.\n")
                continue

            # Step 2: Run clustering and visualization
            cluster_labels, sample_indices = run_analysis(
                movement_key,
                base_folder=BASE_DATA_FOLDER,
                sample_labels=sample_labels,
                group="healthy"  # or "healthy"
            )
            print(f"✅ Analysis done for healthy - {movement_key}\n")

    except Exception as e:
        print(f"❌ Error processing healthy: {e}")


if __name__ == "__main__":
    main()