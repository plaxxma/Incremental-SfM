"""
Compare custom SfM implementation with COLMAP results

This script:
1. Extracts frames from the same video (IMG_1578.MOV)
2. Runs COLMAP on those frames
3. Loads our custom implementation results (output_interval_10)
4. Visualizes both reconstructions side-by-side
5. Compares camera trajectories and 3D points
6. Generates comparison plots and statistics
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import subprocess
import os
import shutil

# COLMAP executable path
COLMAP_PATH = r"C:\CHAEEUN_DATA\3-2\기초컴퓨터비전\assignment2\colmap-x64-windows-nocuda\COLMAP.bat"


def extract_frames_for_colmap(video_path, output_dir, frame_interval=10):
    """Extract frames from video for COLMAP processing"""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExtracting frames from video...")
    print(f"  Video: {video_path}")
    print(f"  Output: {images_dir}")
    print(f"  Interval: every {frame_interval} frames")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Save frame with zero-padded numbering
            frame_name = f"frame_{saved_count:04d}.jpg"
            frame_path = images_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"  Extracted {saved_count} frames...")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"\nExtracted {saved_count} frames to {images_dir}")
    return saved_count, width, height


def run_colmap_feature_extraction(workspace_dir, camera_model="SIMPLE_PINHOLE"):
    """Run COLMAP feature extraction"""
    print("\n" + "="*80)
    print("STEP 1: COLMAP Feature Extraction")
    print("="*80)
    
    database_path = workspace_dir / "database.db"
    images_dir = workspace_dir / "images"
    
    # Remove existing database
    if database_path.exists():
        database_path.unlink()
    
    cmd = [
        COLMAP_PATH, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0"
    ]
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("\n✓ Feature extraction completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during feature extraction:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("\n✗ Error: COLMAP not found!")
        print("\nPlease install COLMAP:")
        print("  - Download from: https://colmap.github.io/")
        print("  - Or install via: conda install -c conda-forge colmap")
        print("  - Make sure 'colmap' is in your PATH")
        return False


def run_colmap_matching(workspace_dir):
    """Run COLMAP feature matching"""
    print("\n" + "="*80)
    print("STEP 2: COLMAP Feature Matching")
    print("="*80)
    
    database_path = workspace_dir / "database.db"
    
    cmd = [
        COLMAP_PATH, "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "0"
    ]
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("\n✓ Feature matching completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during matching:")
        print(e.stderr)
        return False


def run_colmap_mapper(workspace_dir):
    """Run COLMAP incremental mapping"""
    print("\n" + "="*80)
    print("STEP 3: COLMAP Incremental Mapping")
    print("="*80)
    
    database_path = workspace_dir / "database.db"
    images_dir = workspace_dir / "images"
    sparse_dir = workspace_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    
    cmd = [
        COLMAP_PATH, "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir)
    ]
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("\n✓ Incremental mapping completed successfully")
        
        # Check if reconstruction was created
        model_dirs = list(sparse_dir.glob("*"))
        if model_dirs:
            print(f"\nFound {len(model_dirs)} reconstruction model(s)")
            return True
        else:
            print("\n✗ No reconstruction model created")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during mapping:")
        print(e.stderr)
        return False


def run_colmap_pipeline(video_path, output_dir, frame_interval=10):
    """Run complete COLMAP pipeline"""
    output_dir = Path(output_dir)
    
    print("="*80)
    print("RUNNING COLMAP PIPELINE")
    print("="*80)
    print(f"\nVideo: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Frame interval: {frame_interval}")
    
    # Step 0: Extract frames
    print("\n" + "="*80)
    print("STEP 0: Frame Extraction")
    print("="*80)
    
    n_frames, width, height = extract_frames_for_colmap(video_path, output_dir, frame_interval)
    
    # Step 1: Feature extraction
    if not run_colmap_feature_extraction(output_dir):
        return False
    
    # Step 2: Feature matching
    if not run_colmap_matching(output_dir):
        return False
    
    # Step 3: Incremental mapping
    if not run_colmap_mapper(output_dir):
        return False
    
    print("\n" + "="*80)
    print("COLMAP PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return True


def load_colmap_cameras(cameras_file):
    """Load camera intrinsics from COLMAP cameras.txt"""
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            cameras[cam_id] = {
                'model': parts[1],
                'width': int(parts[2]),
                'height': int(parts[3]),
                'params': list(map(float, parts[4:]))
            }
    return cameras


def load_colmap_images(images_file):
    """Load camera poses from COLMAP images.txt"""
    images = {}
    with open(images_file, 'r') as f:
        lines = [l for l in f if not l.startswith('#')]
        for i in range(0, len(lines), 2):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            if len(parts) < 10:
                continue
            
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]
            
            # Convert quaternion to rotation matrix
            R = quat_to_rotation_matrix(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            
            # Camera center in world coordinates
            C = -R.T @ t
            
            images[image_id] = {
                'R': R,
                't': t,
                'C': C,
                'camera_id': camera_id,
                'name': name
            }
    
    return images


def load_colmap_points(points_file):
    """Load 3D points from COLMAP points3D.txt"""
    points = []
    colors = []
    errors = []
    
    with open(points_file, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            
            point_id = int(parts[0])
            xyz = np.array(list(map(float, parts[1:4])))
            rgb = np.array(list(map(int, parts[4:7]))) / 255.0
            error = float(parts[7])
            
            points.append(xyz)
            colors.append(rgb)
            errors.append(error)
    
    return np.array(points), np.array(colors), np.array(errors)


def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix"""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R


def align_reconstructions(points1, points2):
    """
    Align two point clouds using Procrustes analysis (similarity transform)
    Returns: scale, rotation, translation to align points1 to points2
    """
    # Remove outliers first (use 95th percentile)
    def filter_outliers(pts):
        centroid = np.median(pts, axis=0)
        distances = np.linalg.norm(pts - centroid, axis=1)
        threshold = np.percentile(distances, 95)
        return pts[distances < threshold]
    
    pts1_filtered = filter_outliers(points1)
    pts2_filtered = filter_outliers(points2)
    
    # Use fewer points for alignment if needed
    n_align = min(len(pts1_filtered), len(pts2_filtered), 1000)
    
    # Random sampling
    np.random.seed(42)
    idx1 = np.random.choice(len(pts1_filtered), n_align, replace=False)
    idx2 = np.random.choice(len(pts2_filtered), n_align, replace=False)
    
    p1 = pts1_filtered[idx1]
    p2 = pts2_filtered[idx2]
    
    # Compute centroids
    c1 = np.mean(p1, axis=0)
    c2 = np.mean(p2, axis=0)
    
    # Center the points
    p1_centered = p1 - c1
    p2_centered = p2 - c2
    
    # Compute scale
    scale = np.std(p2_centered) / np.std(p1_centered)
    
    # Compute rotation using SVD
    H = (p1_centered * scale).T @ p2_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = c2 - scale * R @ c1
    
    return scale, R, t


def align_by_frame_names(our_images, colmap_images):
    """
    Align reconstructions by matching camera positions using frame names
    
    This ensures 1:1 correspondence based on frame names (e.g., frame_0000.jpg)
    which is more robust than point cloud matching.
    """
    # Match cameras by frame name
    matched_pairs = []
    
    for our_id, our_img in our_images.items():
        our_name = our_img['name']
        
        # Find matching frame in COLMAP
        for colmap_id, colmap_img in colmap_images.items():
            if colmap_img['name'] == our_name:
                matched_pairs.append({
                    'name': our_name,
                    'our_C': our_img['C'],
                    'colmap_C': colmap_img['C']
                })
                break
    
    print(f"\nMatched {len(matched_pairs)} camera pairs by frame name")
    
    if len(matched_pairs) < 3:
        print("Warning: Not enough matched cameras for alignment!")
        return None, None, None
    
    # Extract camera centers
    our_centers = np.array([p['our_C'] for p in matched_pairs])
    colmap_centers = np.array([p['colmap_C'] for p in matched_pairs])
    
    # Compute centroids
    our_centroid = np.mean(our_centers, axis=0)
    colmap_centroid = np.mean(colmap_centers, axis=0)
    
    # Center the points
    our_centered = our_centers - our_centroid
    colmap_centered = colmap_centers - colmap_centroid
    
    # Compute scale
    our_scale = np.sqrt(np.sum(our_centered**2) / len(our_centered))
    colmap_scale = np.sqrt(np.sum(colmap_centered**2) / len(colmap_centered))
    scale = colmap_scale / our_scale
    
    # Compute rotation using SVD
    H = (our_centered / our_scale).T @ (colmap_centered / colmap_scale)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = colmap_centroid - scale * R @ our_centroid
    
    print(f"  Scale factor: {scale:.4f}")
    print(f"  Rotation angle: {np.arccos((np.trace(R)-1)/2)*180/np.pi:.2f}°")
    
    return scale, R, t


def apply_transform(points, scale, R, t):
    """Apply similarity transform to points"""
    return scale * (points @ R.T) + t


def compute_camera_trajectory_error(cameras1, cameras2):
    """Compute error between camera trajectories"""
    # Match cameras by name
    common_names = set()
    name_to_id1 = {}
    name_to_id2 = {}
    
    for id1, cam1 in cameras1.items():
        name_to_id1[cam1['name']] = id1
    
    for id2, cam2 in cameras2.items():
        name_to_id2[cam2['name']] = id2
    
    common_names = set(name_to_id1.keys()) & set(name_to_id2.keys())
    
    if len(common_names) == 0:
        print("Warning: No common camera names found. Matching by order instead.")
        ids1 = sorted(cameras1.keys())
        ids2 = sorted(cameras2.keys())
        n_common = min(len(ids1), len(ids2))
        
        positions1 = np.array([cameras1[ids1[i]]['C'] for i in range(n_common)])
        positions2 = np.array([cameras2[ids2[i]]['C'] for i in range(n_common)])
    else:
        positions1 = []
        positions2 = []
        for name in sorted(common_names):
            positions1.append(cameras1[name_to_id1[name]]['C'])
            positions2.append(cameras2[name_to_id2[name]]['C'])
        
        positions1 = np.array(positions1)
        positions2 = np.array(positions2)
    
    # Compute distances
    distances = np.linalg.norm(positions1 - positions2, axis=1)
    
    return distances, positions1, positions2


def filter_outliers_percentile(points, colors, threshold=5.0):
    """Filter outliers using percentile method (same as visualize_filtered.py)"""
    lower = np.percentile(points, threshold, axis=0)
    upper = np.percentile(points, 100 - threshold, axis=0)
    
    inlier_mask = np.all((points >= lower) & (points <= upper), axis=1)
    
    filtered_points = points[inlier_mask]
    filtered_colors = colors[inlier_mask]
    
    print(f"  Original: {len(points)} points")
    print(f"  Filtered: {len(filtered_points)} points ({100 * len(filtered_points)/len(points):.1f}%)")
    
    return filtered_points, filtered_colors


def visualize_comparison(our_points, our_colors, our_cameras,
                        colmap_points, colmap_colors, colmap_cameras,
                        output_dir):
    """Create comprehensive comparison visualization"""
    
    # Filter outliers from our implementation (for visualization only)
    print("\nFiltering outliers from our implementation...")
    our_points_filtered, our_colors_filtered = filter_outliers_percentile(our_points, our_colors, threshold=5.0)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Point clouds comparison (top row)
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.scatter(our_points_filtered[:, 0], our_points_filtered[:, 1], our_points_filtered[:, 2],
                c=our_colors_filtered, s=1, alpha=0.5)
    for cam_id, cam in our_cameras.items():
        C = cam['C']
        ax1.scatter(C[0], C[1], C[2], c='red', s=50, marker='^')
    ax1.set_title(f'Our Implementation (Filtered)\n({len(our_points_filtered)} points, {len(our_cameras)} cameras)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    ax2 = fig.add_subplot(232, projection='3d')
    ax2.scatter(colmap_points[:, 0], colmap_points[:, 1], colmap_points[:, 2],
                c=colmap_colors, s=1, alpha=0.5)
    for cam_id, cam in colmap_cameras.items():
        C = cam['C']
        ax2.scatter(C[0], C[1], C[2], c='blue', s=50, marker='^')
    ax2.set_title(f'COLMAP\n({len(colmap_points)} points, {len(colmap_cameras)} cameras)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 2. Aligned comparison - using frame name matching
    print("\nAligning reconstructions by frame names...")
    scale, R, t = align_by_frame_names(our_cameras, colmap_cameras)
    
    if scale is None:
        print("Alignment failed! Using identity transform.")
        scale, R, t = 1.0, np.eye(3), np.zeros(3)
    
    our_points_aligned = apply_transform(our_points_filtered, scale, R, t)
    our_camera_centers = np.array([cam['C'] for cam in our_cameras.values()])
    our_camera_centers_aligned = apply_transform(our_camera_centers, scale, R, t)
    
    ax3 = fig.add_subplot(233, projection='3d')
    ax3.scatter(our_points_aligned[:, 0], our_points_aligned[:, 1], our_points_aligned[:, 2],
                c=our_colors_filtered, s=1, alpha=0.3, label='Ours (aligned)')
    ax3.scatter(colmap_points[:, 0], colmap_points[:, 1], colmap_points[:, 2],
                c=colmap_colors, s=1, alpha=0.3, label='COLMAP')
    ax3.scatter(our_camera_centers_aligned[:, 0], our_camera_centers_aligned[:, 1], 
                our_camera_centers_aligned[:, 2], c='red', s=50, marker='^', label='Our cameras')
    colmap_camera_centers = np.array([cam['C'] for cam in colmap_cameras.values()])
    ax3.scatter(colmap_camera_centers[:, 0], colmap_camera_centers[:, 1],
                colmap_camera_centers[:, 2], c='blue', s=50, marker='^', label='COLMAP cameras')
    ax3.set_title('Aligned Overlay (Filtered)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    # 3. Camera trajectories (bottom row)
    ax4 = fig.add_subplot(234)
    our_traj = np.array([cam['C'] for cam in our_cameras.values()])
    colmap_traj = np.array([cam['C'] for cam in colmap_cameras.values()])
    ax4.plot(our_traj[:, 0], our_traj[:, 1], 'r.-', label='Our Implementation', linewidth=2, markersize=8)
    ax4.plot(colmap_traj[:, 0], colmap_traj[:, 1], 'b.-', label='COLMAP', linewidth=2, markersize=8)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('Camera Trajectories (Top View)')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')
    
    # 4. Statistics comparison
    ax5 = fig.add_subplot(235)
    stats_text = f"""
    Statistics Comparison
    
    Our Implementation:
      - Points: {len(our_points):,}
      - Cameras: {len(our_cameras)}
      - Point density: {len(our_points)/len(our_cameras):.1f} pts/cam
    
    COLMAP:
      - Points: {len(colmap_points):,}
      - Cameras: {len(colmap_cameras)}
      - Point density: {len(colmap_points)/len(colmap_cameras):.1f} pts/cam
    
    Alignment:
      - Scale factor: {scale:.4f}
      - Rotation angle: {np.arccos((np.trace(R)-1)/2)*180/np.pi:.2f}°
    """
    ax5.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.axis('off')
    
    # 5. Point distance histogram
    ax6 = fig.add_subplot(236)
    
    # Sample points for comparison
    n_sample = min(len(our_points_aligned), len(colmap_points), 5000)
    sample_idx_our = np.random.choice(len(our_points_aligned), n_sample, replace=False)
    sample_idx_colmap = np.random.choice(len(colmap_points), n_sample, replace=False)
    
    our_sample = our_points_aligned[sample_idx_our]
    colmap_sample = colmap_points[sample_idx_colmap]
    
    # Compute nearest neighbor distances
    from scipy.spatial import cKDTree
    tree = cKDTree(colmap_sample)
    distances, _ = tree.query(our_sample, k=1)
    
    ax6.hist(distances, bins=50, edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Distance to nearest COLMAP point')
    ax6.set_ylabel('Frequency')
    ax6.set_title(f'Point Correspondence\nMedian dist: {np.median(distances):.4f}')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(output_dir) / "comparison_colmap_frame_aligned.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot: {save_path}")
    plt.close()
    
    return scale, R, t, distances


def visualize_open3d_comparison(our_points, our_colors, our_cameras,
                                colmap_points, colmap_colors, colmap_cameras,
                                scale, R, t):
    """Interactive Open3D comparison visualization"""
    
    # Filter outliers from our implementation (for visualization only)
    print("\nFiltering outliers for Open3D visualization...")
    our_points_filtered, our_colors_filtered = filter_outliers_percentile(our_points, our_colors, threshold=5.0)
    
    # Align our reconstruction
    our_points_aligned = apply_transform(our_points_filtered, scale, R, t)
    our_camera_centers = np.array([cam['C'] for cam in our_cameras.values()])
    our_camera_centers_aligned = apply_transform(our_camera_centers, scale, R, t)
    
    # Create point clouds
    pcd_ours = o3d.geometry.PointCloud()
    pcd_ours.points = o3d.utility.Vector3dVector(our_points_aligned)
    pcd_ours.colors = o3d.utility.Vector3dVector(our_colors_filtered)
    
    pcd_colmap = o3d.geometry.PointCloud()
    pcd_colmap.points = o3d.utility.Vector3dVector(colmap_points)
    pcd_colmap.colors = o3d.utility.Vector3dVector(colmap_colors)
    
    # Create camera markers
    our_cam_pcd = o3d.geometry.PointCloud()
    our_cam_pcd.points = o3d.utility.Vector3dVector(our_camera_centers_aligned)
    our_cam_pcd.paint_uniform_color([1, 0, 0])  # Red
    
    colmap_cam_pcd = o3d.geometry.PointCloud()
    colmap_cam_centers = np.array([cam['C'] for cam in colmap_cameras.values()])
    colmap_cam_pcd.points = o3d.utility.Vector3dVector(colmap_cam_centers)
    colmap_cam_pcd.paint_uniform_color([0, 0, 1])  # Blue
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    print("\n" + "="*80)
    print("INTERACTIVE 3D COMPARISON")
    print("="*80)
    print("\nVisualization:")
    print("  - Point clouds: Our implementation and COLMAP overlaid")
    print("  - Red markers: Our camera positions")
    print("  - Blue markers: COLMAP camera positions")
    print("\nControls:")
    print("  - Mouse wheel: Zoom")
    print("  - Left-click + drag: Rotate")
    print("  - Shift + left-click + drag: Pan")
    print("  - R: Reset view")
    print("\nClose the window when done\n")
    
    o3d.visualization.draw_geometries(
        [pcd_ours, pcd_colmap, our_cam_pcd, colmap_cam_pcd, coord_frame],
        window_name="COLMAP vs Our Implementation (Aligned)",
        width=1920,
        height=1080
    )


def print_comparison_summary(our_points, our_cameras, colmap_points, colmap_cameras, distances):
    """Print detailed comparison summary"""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Our Implementation':<20} {'COLMAP':<20} {'Difference':<20}")
    print("-" * 90)
    
    print(f"{'Number of 3D points':<30} {len(our_points):<20,} {len(colmap_points):<20,} "
          f"{len(our_points) - len(colmap_points):<20,}")
    
    print(f"{'Number of cameras':<30} {len(our_cameras):<20} {len(colmap_cameras):<20} "
          f"{len(our_cameras) - len(colmap_cameras):<20}")
    
    our_density = len(our_points) / len(our_cameras)
    colmap_density = len(colmap_points) / len(colmap_cameras)
    print(f"{'Points per camera':<30} {our_density:<20.1f} {colmap_density:<20.1f} "
          f"{our_density - colmap_density:<20.1f}")
    
    print(f"\n{'Point correspondence distances (after alignment):':<60}")
    print(f"  Mean: {np.mean(distances):.4f}")
    print(f"  Median: {np.median(distances):.4f}")
    print(f"  Std: {np.std(distances):.4f}")
    print(f"  Min: {np.min(distances):.4f}")
    print(f"  Max: {np.max(distances):.4f}")
    print(f"  95th percentile: {np.percentile(distances, 95):.4f}")


def main():
    print("="*80)
    print("COMPARISON WITH COLMAP")
    print("="*80)
    
    # Paths
    video_path = Path("Video/IMG_1578.MOV")
    our_dir = Path("output_interval_10")
    colmap_dir = Path("colmap_output")
    
    print(f"\nVideo: {video_path}")
    print(f"Our implementation results: {our_dir}")
    print(f"COLMAP output directory: {colmap_dir}")
    
    # Check if video exists
    if not video_path.exists():
        print(f"\nError: Video not found: {video_path}")
        return
    
    # Check if our results exist
    if not our_dir.exists():
        print(f"\nError: Our results directory not found: {our_dir}")
        print("Please run incremental_sfm.py first!")
        return
    
    # Ask user if they want to run COLMAP
    colmap_sparse = colmap_dir / "sparse" / "0"
    
    if colmap_sparse.exists() and (colmap_sparse / "cameras.txt").exists():
        print(f"\n✓ Found existing COLMAP results at {colmap_sparse}")
        response = input("\nDo you want to re-run COLMAP? (y/n): ").strip().lower()
        run_colmap = (response == 'y')
    else:
        print(f"\n✗ COLMAP results not found at {colmap_sparse}")
        response = input("\nDo you want to run COLMAP now? (y/n): ").strip().lower()
        run_colmap = (response == 'y')
    
    if run_colmap:
        # Run COLMAP pipeline
        success = run_colmap_pipeline(video_path, colmap_dir, frame_interval=10)
        if not success:
            print("\n✗ COLMAP pipeline failed. Cannot proceed with comparison.")
            return
    
    # Find COLMAP reconstruction directory
    colmap_sparse = colmap_dir / "sparse" / "0"
    if not colmap_sparse.exists():
        print(f"\n✗ Error: COLMAP reconstruction not found at {colmap_sparse}")
        return
    
    # Load our implementation
    print("\n" + "="*80)
    print("LOADING OUR IMPLEMENTATION")
    print("="*80)
    
    our_cameras = load_colmap_cameras(our_dir / "cameras.txt")
    our_images = load_colmap_images(our_dir / "images.txt")
    our_points, our_colors, our_errors = load_colmap_points(our_dir / "points3D.txt")
    
    print(f"\nLoaded:")
    print(f"  - {len(our_cameras)} camera model(s)")
    print(f"  - {len(our_images)} image(s)")
    print(f"  - {len(our_points):,} 3D point(s)")
    
    # Load COLMAP
    print("\n" + "="*80)
    print("LOADING COLMAP RESULTS")
    print("="*80)
    
    colmap_cameras = load_colmap_cameras(colmap_sparse / "cameras.txt")
    colmap_images = load_colmap_images(colmap_sparse / "images.txt")
    colmap_points, colmap_colors, colmap_errors = load_colmap_points(colmap_sparse / "points3D.txt")
    
    print(f"\nLoaded:")
    print(f"  - {len(colmap_cameras)} camera model(s)")
    print(f"  - {len(colmap_images)} image(s)")
    print(f"  - {len(colmap_points):,} 3D point(s)")
    
    # Create comparison visualization
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    scale, R, t, distances = visualize_comparison(
        our_points, our_colors, our_images,
        colmap_points, colmap_colors, colmap_images,
        our_dir
    )
    
    # Print summary
    print_comparison_summary(our_points, our_images, colmap_points, colmap_images, distances)
    
    # Interactive visualization
    visualize_open3d_comparison(
        our_points, our_colors, our_images,
        colmap_points, colmap_colors, colmap_images,
        scale, R, t
    )
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETED")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {our_dir}/comparison_colmap_frame_aligned.png (NEW - frame-based alignment)")
    print(f"  - {our_dir}/comparison_colmap.png (OLD - point cloud-based alignment)")
    print(f"\nCOLMAP results saved in:")
    print(f"  - {colmap_sparse}/cameras.txt")
    print(f"  - {colmap_sparse}/images.txt")
    print(f"  - {colmap_sparse}/points3D.txt")


if __name__ == "__main__":
    main()
