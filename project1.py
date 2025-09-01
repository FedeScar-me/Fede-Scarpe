import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from fastdtw import fastdtw
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
from tslearn.metrics import dtw_path
import pandas as pd

def load_movement_data(movement_key, base_folder, group=None):
    """
    Load data for a given movement key and group from a .npy file.

    Parameters:
        movement_key (str): movement name, e.g., 'fwr-bck-L'
        base_folder (str): folder path where files are stored
        group (str): 'patients' or 'healthy'

    Returns:
        data (np.ndarray): loaded data array
    """
    filename = f"preprocessed_kin-{movement_key}_{group}.npy"
    filepath = os.path.join(base_folder, filename)

    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return None

    try:
        data = np.load(filepath)
        print(f"Loaded {group} data shape for {movement_key}: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# Constants
DTW_DISTANCE_METRIC = euclidean

def compute_dtw_distance_matrix(data):
    """
    Compute the full pairwise DTW distance matrix using tslearn.dtw_path.
    This version uses independent multidimensional DTW (sums DTW over each feature).
    
    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, timepoints, features)
        
    Returns
    -------
    distance_matrix : np.ndarray
        Array of shape (n_samples, n_samples) containing pairwise DTW distances
    """
    n_samples, _, n_features = data.shape
    distance_matrix = np.zeros((n_samples, n_samples))
    print("Computing DTW distances using tslearn...")
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            series_i = data[i]
            series_j = data[j]
            
            # Compute DTW distance independently for each feature
            dist = 0
            for f in range(n_features):
                _, d = dtw_path(series_i[:, f], series_j[:, f])
                dist += d  # sum distances over all features
                
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # symmetric matrix
            
    return distance_matrix

import pandas as pd

def evaluate_clusters(distance_matrix, min_k, max_k):
    n_samples = distance_matrix.shape[0]
    max_k = min(max_k, n_samples - 1)  

    k_values = []
    silhouette_scores = []
    avg_distances = []
    cluster_labels = []

    for k in range(min_k, max_k + 1):
        print(f"Clustering with k={k}...")
        try:
            clustering = AgglomerativeClustering(
                n_clusters=k, 
                metric="precomputed", 
                linkage="average"
            )
            labels = clustering.fit_predict(distance_matrix)

            # Compute silhouette score only if > 1 cluster
            if len(set(labels)) > 1:
                sil_score = silhouette_score(distance_matrix, labels, metric="precomputed")
            else:
                sil_score = np.nan  # placeholder

            # Average intra-cluster distance
            avg_dist = np.mean([
                distance_matrix[i, j]
                for i in range(len(labels))
                for j in range(len(labels))
                if i < j and labels[i] == labels[j]
            ]) if len(set(labels)) > 1 else np.nan

            # Append results
            k_values.append(k)
            silhouette_scores.append(sil_score)
            avg_distances.append(avg_dist)

        except Exception as e:
            print(f"⚠️ Skipping k={k} due to error: {e}")
            k_values.append(k)
            silhouette_scores.append(np.nan)
            avg_distances.append(np.nan)

    chosen_k = 2 #manual choice

    # Final clustering using chosen_k
    clustering = AgglomerativeClustering(
        n_clusters=chosen_k,
        metric="precomputed",
        linkage="average"
    )
    cluster_labels = clustering.fit_predict(distance_matrix)

    return k_values, silhouette_scores, avg_distances, cluster_labels

def compute_dtw_and_cluster(data, min_k=2, max_k=None):
    """
    Full pipeline for DTW clustering evaluation.
    - min_k: smallest cluster count to try
    - max_k: largest cluster count (defaults to number of samples)
    """
    n_samples = data.shape[0]
    if max_k is None:
        max_k = n_samples  # default to total participants

    # Step 1: Compute distance matrix
    distance_matrix = compute_dtw_distance_matrix(data)

    # Step 2: Evaluate clustering
    k_values, silhouette_scores, avg_distances, cluster_labels = evaluate_clusters(distance_matrix, min_k, max_k)

    # Step 3: Put into DataFrame
    import pandas as pd
    results_df = pd.DataFrame({
        "k": list(k_values),
        "Silhouette Score": list(silhouette_scores),
        "Avg Intra-cluster Distance": list(avg_distances)
    })

    # Print table
    print("\nClustering results:")
    print(results_df)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left y-axis: Silhouette Score
    ax1.plot(results_df['k'], results_df['Silhouette Score'], 'o-', color='blue', label='Silhouette Score')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Silhouette Score', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(results_df['k'])

    # Right y-axis: Avg Intra-cluster Distance
    ax2 = ax1.twinx()
    ax2.plot(results_df['k'], results_df['Avg Intra-cluster Distance'], 's--', color='red', label='Avg Intra-cluster Distance')
    ax2.set_ylabel('Avg Intra-cluster Distance', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Optional: Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Clustering Evaluation')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    print(cluster_labels)
    return results_df, cluster_labels

def plot_3d_trajectories_with_pca_eigenvectors(
    data,
    flattened_data,
    cluster_labels,
    movement_name,
    joint_index=0,
    sample_labels=None
):
    """
    Plot 3D trajectories for a specific joint across all participants,
    and overlay one PCA eigenvector per cluster (if cluster size >= 2).

    Parameters:
    - data: numpy array of shape (samples, timepoints, joints, 3)
    - flattened_data: numpy array of shape (samples, features) used for PCA
    - cluster_labels: array-like, cluster label per sample
    - movement_name: str, used for plot title
    - joint_index: int, index of joint to plot (0 to 24)
    - sample_labels: list of str, optional labels for samples
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np
    from sklearn.decomposition import PCA
    from matplotlib.lines import Line2D

    assert 0 <= joint_index < 25, "Joint index must be between 0 and 24."
    print(f"📈 Plotting 3D trajectories for joint {joint_index + 1} in {movement_name}...")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    unique_clusters = np.unique(cluster_labels)
    cmap = plt.get_cmap('tab10', len(unique_clusters))

    if sample_labels is None:
        sample_labels = [f"Sample_{i}" for i in range(data.shape[0])]

    # Plot each participant trajectory colored by cluster
    for i in range(data.shape[0]):
        traj = data[i, :, joint_index, :]  # (timepoints, 3)
        color = cmap(cluster_labels[i])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.7)

    # Calculate data range (max span across all axes) for scaling eigenvectors
    all_points = np.concatenate([data[i, :, joint_index, :] for i in range(data.shape[0])], axis=0)
    data_range = np.max(all_points, axis=0) - np.min(all_points, axis=0)
    max_range = np.max(data_range)

    # Dictionary to store explained variance per cluster for legend
    cluster_explained_var = {}

    # Overlay PCA eigenvectors per cluster (only clusters with >= 2 participants)
    for idx, cluster in enumerate(unique_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) < 2:
            print(f"⚠️ Skipping PCA vector for Cluster {cluster} (only {len(cluster_indices)} sample).")
            cluster_explained_var[cluster] = None
            continue

        # Extract all 3D points for this joint across all time points and samples in the cluster
        cluster_points = data[cluster_indices, :, joint_index, :].reshape(-1, 3)  # (samples*timepoints, 3)

        # Compute mean position to use as vector origin
        cluster_mean_pos = np.mean(cluster_points, axis=0)

        # Perform PCA on these 3D points directly
        pca = PCA(n_components=1)
        pca.fit(cluster_points)

        eigenvector = pca.components_[0]  # unit vector in 3D
        explained_var = pca.explained_variance_[0]

        # Store explained variance for legend (as %)
        cluster_explained_var[cluster] = explained_var * 100

        # Increase scale factor here to make segments longer, while keeping variance dependence
        scale = max_range * 0.6 * np.sqrt(explained_var)  # doubled from 0.3 to 0.6

        vec = eigenvector * scale
        vector_length = np.linalg.norm(vec)

        print(f"Cluster {cluster}: Vector length: {vector_length:.3e}")
        print(f"Eigenvector Cluster {cluster}: {eigenvector}, scale: {scale}; explained variance: {explained_var}")

        start = cluster_mean_pos
        end = cluster_mean_pos + vec

        # Plot PCA vector as a thick black line
        ax.plot([start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color='black', linewidth=3, zorder=10)

    # Create legend for clusters including explained variance in label
    cluster_handles = [
        Line2D(
            [0], [0],
            color=cmap(i),
            lw=3,
            label=(
                f"Cluster {cluster}" + 
                (f" (λ% = {cluster_explained_var[cluster]:.2f})" if cluster_explained_var.get(cluster) is not None else "")
            )
        )
        for i, cluster in enumerate(unique_clusters)
    ]

    # Simplify participant labels (e.g., "P01_Rch_..." → "P01")
    simplified_labels = [
        label.split('_')[0] if '_' in label else label
        for label in sample_labels
    ]

    # Create legend for participants using simplified labels
    participant_handles = [
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor=cmap(cluster_labels[i]),
            markersize=8,
            label=simplified_labels[i] if i < len(simplified_labels) else f"Sample_{i}")
        for i in range(len(data))
    ]


    ax.legend(handles=cluster_handles + participant_handles,
              title='Clusters and Participants',
              loc='center left', bbox_to_anchor=(1.05, 0.5),
              borderaxespad=0., ncol=2)

    ax.set_title(f"3D Trajectories with PCA Eigenvectors — {movement_name}", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Adjust axis limits to include zero and all data points
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    z_min, z_max = np.min(all_points[:, 2]), np.max(all_points[:, 2])

    ax.set_xlim(min(x_min, 0), max(x_max, 0))
    ax.set_ylim(min(y_min, 0), max(y_max, 0))
    ax.set_zlim(min(z_min, 0), max(z_max, 0))

    plt.tight_layout()
    plt.show()
    plt.close(fig)  # Close the figure to prevent empty figures popping up


def plot_tsne(flattened_data, cluster_labels, sample_labels, movement_name):
    """
    Perform t-SNE dimensionality reduction and plot clusters.

    Parameters:
        flattened_data (np.ndarray): 2D array (samples, features)
        cluster_labels (np.ndarray): Cluster assignments per sample
        sample_labels (list): List of sample names
        movement_name (str): Label for plots
    """
    print(f"🌀 Performing t-SNE for {movement_name}...")

    # No reshaping here — data is already flattened
    tsne_model = TSNE(n_components=2, random_state=42)
    tsne_emb = tsne_model.fit_transform(flattened_data)

    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_clusters))

    plt.figure(figsize=(8, 6))
    for idx, cluster in enumerate(unique_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        plt.scatter(
            tsne_emb[cluster_indices, 0], tsne_emb[cluster_indices, 1],
            label=f'Cluster {cluster}', alpha=0.7, color=colors(idx)
        )

    plt.title(f't-SNE - {movement_name}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_umap(flattened_data, cluster_labels, sample_labels, movement_name):
    """
    Perform UMAP dimensionality reduction and plot the 2D embedding.

    Parameters:
        flattened_data (np.ndarray): 2D array (samples, features)
        cluster_labels (np.ndarray): Cluster assignments.
        sample_labels (list): List of sample names.
        movement_name (str): Movement type for title.
    """
    print(f"Performing UMAP for {movement_name}...")

    try:
        umap_model = umap.UMAP(n_components=2, random_state=42)
        umap_emb = umap_model.fit_transform(flattened_data)

        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.get_cmap('tab10', len(unique_clusters))

        plt.figure(figsize=(8, 6))
        for idx, cluster in enumerate(unique_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            plt.scatter(
                umap_emb[cluster_indices, 0], umap_emb[cluster_indices, 1],
                label=f'Cluster {cluster}', alpha=0.7, color=colors(idx)
            )

        plt.title(f'UMAP Visualization - {movement_name}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"❌ UMAP plotting failed for {movement_name}: {e}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np

def plot_cluster_trajectories(data, cluster_labels, movement_key, sample_labels):
    """
    Plot average 3D trajectories for shoulder, elbow, wrist across all clusters in a single figure.
    Each cluster gets a subplot showing the average trajectory for selected joints.

    Parameters:
        data: np.ndarray of shape (samples, timepoints, joints, 3)
        cluster_labels: array-like of cluster assignments per sample
        movement_key: str
        sample_labels: dict {participant_id: [folder names...]}
    """
    print(f"📊 Plotting cluster-average trajectories for {movement_key}...")

    # --- Config ---
    selected_joints = {
        "Shoulder": [4, 8],
        "Elbow": [5, 9],
        "Wrist": [6, 10]
    }
    line_styles = {
        "Shoulder": ':',
        "Elbow": '--',
        "Wrist": '-'
    }

    joint_name_map = {idx: name for name, indices in selected_joints.items() for idx in indices}

    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters)

    fig = plt.figure(figsize=(5 * num_clusters, 5))
    cmap = plt.get_cmap('tab10', num_clusters)

    for i, cluster in enumerate(unique_clusters):
        ax = fig.add_subplot(1, num_clusters, i + 1, projection='3d')
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_data = data[cluster_indices]  # (samples_in_cluster, timepoints, joints, 3)

        # Compute average trajectory per joint
        avg_traj = np.mean(cluster_data, axis=0)  # shape: (timepoints, joints, 3)
        cluster_color = cmap(cluster)

        for name, joint_idxs in selected_joints.items():
            for joint_idx in joint_idxs:
                traj = avg_traj[:, joint_idx, :]
                ax.plot(
                    traj[:, 0], traj[:, 1], traj[:, 2],
                    linestyle=line_styles[name],
                    color=cluster_color,
                    label=f"{name} (Joint {joint_idx})"
                )

        # Add cluster info
        ax.set_title(f"Cluster {cluster}", fontsize=12)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.grid(True)
        ax.set_box_aspect([1, 1, 1])

        # Add participant list in annotation
        participant_ids = []
        flat_index = 0
        for pid, folders in sample_labels.items():
            for _ in folders:
                if flat_index in cluster_indices:
                    participant_ids.append(pid)
                flat_index += 1

        participants_str = "\n".join(participant_ids)
        ax.text2D(0.05, 0.95, f"Participants:\n{participants_str}", transform=ax.transAxes,
                  fontsize=8, verticalalignment='top')

    # Create a single legend with line styles
    legend_lines = [
        Line2D([0], [0], linestyle=':', color='black', label='Shoulder'),
        Line2D([0], [0], linestyle='--', color='black', label='Elbow'),
        Line2D([0], [0], linestyle='-', color='black', label='Wrist')
    ]
    fig.legend(handles=legend_lines, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=10)

    fig.suptitle(f"Average 3D Trajectories — {movement_key}", fontsize=14)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Function definition:
# def plot_cluster_joint_differences(data, cluster_labels, shoulder_joints, elbow_joints, wrist_joints, movement_name, cmap=None)

# Correct positional argument order:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_cluster_joint_differences( 
    data,
    cluster_labels,
    shoulder_joints,
    elbow_joints,
    wrist_joints,
    lower_back_joints,
    movement_name,
    cmap=None
):
    """
    Plot time series of average joint differences per cluster.
    Also plots horizontal lines for time-averaged values per cluster,
    with small numeric labels, and shaded variance areas.
    Returns a DataFrame with cluster × joint group averages.
    """
    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters)
    timepoints = data.shape[1]

    summary_rows = []  # store cluster × joint group averages

    if cmap is None:
        cmap = plt.get_cmap('tab10', num_clusters)

    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    fig.suptitle(f"Cluster-Averaged Joint Differences — {movement_name}", fontsize=16)

    joint_groups = {
        "Upper arm movement [m]": (shoulder_joints, elbow_joints),
        "Lower arm movement [m]": (elbow_joints, wrist_joints),
        "Torso-shoulder movement [m]": (lower_back_joints, shoulder_joints),
    }

    spatial_labels = ["Flex-Extension (X)", "Abd-Adduction (Y)", "Intra-Extrarotation (Z)"]
    direction_texts = [
        ("← flexion", "← extension"),
        ("← abduction", "← adduction"),
        ("← external", "← internal")
    ]

    y_limits = [[] for _ in range(3)]

    for cluster_idx, cluster in enumerate(unique_clusters):
        cluster_mask = (cluster_labels == cluster)
        cluster_data = data[cluster_mask]  # (samples_in_cluster, timepoints, joints, 3)

        for row_idx, (joint_label, (joint_a, joint_b)) in enumerate(joint_groups.items()):
            joint_a = np.array(joint_a, dtype=int)
            joint_b = np.array(joint_b, dtype=int)

            diff = cluster_data[:, :, joint_a, :] - cluster_data[:, :, joint_b, :]
            avg_diff = np.mean(diff, axis=2)  # mean across joint pairs
            mean_diff = np.mean(avg_diff, axis=0)  # mean across samples
            std_diff = np.std(avg_diff, axis=0)    # std deviation across samples

            time_avg = np.mean(mean_diff, axis=0)  # average across time

            # --- Append cluster × joint group averages here ---
            summary_rows.append({
                "Cluster": cluster,
                "Joint group": joint_label,
                "X_avg": time_avg[0],
                "Y_avg": time_avg[1],
                "Z_avg": time_avg[2],
            })

            for col_idx in range(3):
                ax = axes[row_idx, col_idx]
                
                # Plot time series
                ax.plot(
                    np.arange(timepoints),
                    mean_diff[:, col_idx],
                    label=f"Cluster {cluster}",
                    color=cmap(cluster_idx),
                    linewidth=2
                )
                
                # Plot shaded variance
                ax.fill_between(
                    np.arange(timepoints),
                    mean_diff[:, col_idx] - std_diff[:, col_idx],
                    mean_diff[:, col_idx] + std_diff[:, col_idx],
                    color=cmap(cluster_idx),
                    alpha=0.2
                )

                # Plot horizontal mean line
                ax.hlines(
                    y=time_avg[col_idx],
                    xmin=0,
                    xmax=timepoints - 1,
                    colors=cmap(cluster_idx),
                    linestyles='dashed',
                    linewidth=1.5,
                    alpha=0.7
                )
                
                # Add numeric label
                ax.text(
                    timepoints - 1, time_avg[col_idx] + 0.01*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                    f"{time_avg[col_idx]:.2f}",
                    color='black',
                    fontsize=8,
                    verticalalignment='bottom',
                    horizontalalignment='right'
                )

                y_limits[row_idx].extend(mean_diff[:, col_idx])

                if row_idx == 0:
                    ax.set_title(spatial_labels[col_idx], fontsize=11)

    for row_idx in range(3):
        min_y = min(y_limits[row_idx])
        max_y = max(y_limits[row_idx])
        buffer = 0.05 * (max_y - min_y)
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.set_ylim(min_y - buffer, max_y + buffer)
            ax.text(0.01, 0.95, direction_texts[col_idx][0], transform=ax.transAxes, fontsize=8, verticalalignment='top', color='gray')
            ax.text(0.01, 0.05, direction_texts[col_idx][1], transform=ax.transAxes, fontsize=8, verticalalignment='bottom', color='gray')

    for row_idx, joint_label in enumerate(joint_groups.keys()):
        axes[row_idx, 0].set_ylabel(joint_label, fontsize=11)

    for col_idx in range(3):
        axes[2, col_idx].set_xlabel("Time frames")

    cluster_handles = [
        Line2D([0], [0], color=cmap(i), lw=3, label=f'Cluster {cluster}')
        for i, cluster in enumerate(unique_clusters)
    ]
    fig.legend(
        handles=cluster_handles,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        title="Clusters"
    )

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

    # Convert summary to DataFrame and return
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df)
    return summary_df

def run_analysis(movement_key, base_folder, sample_labels=None, group=None):
    """
    Run analysis for the whole movement data loaded from one file.

    Parameters:
        movement_key (str): movement name like 'fwr-bck-R'
        base_folder (str): folder where .npy files are stored
        sample_labels (dict or list, optional): participant labels for each sample
        group (str): 'patients' or 'healthy'

    Returns:
        cluster_labels (np.ndarray) or None: cluster assignments per sample
        sample_labels (list) or None: sample labels used in analysis
    """
    print(f"\n🔍 Running analysis for movement: {movement_key} ({group})")

    if isinstance(sample_labels, dict):
        flat_labels = []
        original_labels = sample_labels
        for pid, folders in sample_labels.items():
            for folder in folders:
                flat_labels.append(f"{pid}_{folder}")
        sample_labels = flat_labels  

    # 👇 pass group to loader
    data = load_movement_data(movement_key, base_folder, group=group)
    if data is None:
        print(f"No data loaded for {movement_key} ({group}). Skipping analysis.")
        return None, None

    # Create dummy labels if none provided
    if sample_labels is None:
        sample_labels = [f"Sample_{i}" for i in range(data.shape[0])]

    # Reshape data to 3D for DTW/clustering: (samples, timepoints, features)
    n_samples, n_timepoints, n_markers, n_coords = data.shape
    reshaped_data = data.reshape(n_samples, n_timepoints, n_markers * n_coords)

    # Flatten reshaped_data for PCA and similar methods: (samples, timepoints * features)
    flattened_data = reshaped_data.reshape(n_samples, n_timepoints * n_markers * n_coords)

    try:
        # Step 1: Compute DTW distance matrix and cluster
        results_df, cluster_labels = compute_dtw_and_cluster(reshaped_data)
        # Step 2: Plot 3D trajectories with PCA eigenvectors (combined)
        # Select joint index (0 = 1st joint, 24 = 25th joint)
        joint_to_plot = 6  # example: joint 8
        plot_3d_trajectories_with_pca_eigenvectors(data, flattened_data, cluster_labels, movement_key, joint_index=joint_to_plot, sample_labels=sample_labels)

#0	SpineBase
#1	SpineMid
#2	Neck Base
#3	Top of the head
#4	Left Shoulder (L)
#5	Left Elbow (L)
#6	Left Wrist (L)
#7	Left Hand (L)
#8	Right Shoulder (R)
#9	Right Elbow (R)
#10	Right Wrist (L)
#11	Left Hand (L)
#12	Left Hip (L)
#13	Left Knee (L)
#14	Left Ankle (L)
#15	Left Foot (L)
#16	Right Hip (R)
#17	Right Knee (R)
#18	Right Ankle (R)
#19	Right Foot (R)
#20	SpineSchoulder
#21	Left HandTip (L)
#22	Left Thumb (L)
#23	Right HandTip (R)
#24	Right Thumb (R)

        # cluster_labels already computed
        plot_cluster_trajectories(data, cluster_labels, movement_key, original_labels)
        shoulder_joints = [4]  # example shoulder joint indices
        elbow_joints = [5]
        wrist_joints = [10]
        lower_back_joints = [0]

        plot_cluster_joint_differences(data, cluster_labels,
                               shoulder_joints, elbow_joints, wrist_joints, lower_back_joints, movement_key)

        # Step 4 & 5: t-SNE and UMAP also expect flattened data
        plot_tsne(flattened_data, cluster_labels, sample_labels, movement_key)
        plot_umap(flattened_data, cluster_labels, sample_labels, movement_key)

        return cluster_labels, sample_labels

    except Exception as e:
        print(f"❌ Error during analysis of {movement_key}: {e}")
        print("Check input data shapes and try debugging individual steps.")
    return None, None

