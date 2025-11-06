"""
Visualize 3D reconstruction with outlier filtering
"""
import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_reconstruction(output_dir):
    """Load reconstruction from COLMAP format"""
    output_dir = Path(output_dir)
    
    # Load cameras
    cameras = {}
    with open(output_dir / "cameras.txt", "r") as f:
        for line in f:
            if line.startswith("#"):
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
    
    # Load images (camera poses)
    image_poses = {}
    with open(output_dir / "images.txt", "r") as f:
        lines = [l for l in f if not l.startswith("#")]
        for i in range(0, len(lines), 2):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            if len(parts) < 10:
                continue
            
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            
            # Convert quaternion to rotation matrix
            R = quat_to_rot(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            
            image_poses[image_id] = {'R': R, 't': t, 'name': parts[9]}
    
    # Load 3D points
    points_3d = []
    colors = []
    with open(output_dir / "points3D.txt", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            
            point_id = int(parts[0])
            xyz = np.array(list(map(float, parts[1:4])))
            rgb = np.array(list(map(int, parts[4:7])))
            
            points_3d.append(xyz)
            colors.append(rgb / 255.0)
    
    points_3d = np.array(points_3d)
    colors = np.array(colors)
    
    return cameras, image_poses, points_3d, colors


def quat_to_rot(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix"""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R


def filter_outliers(points_3d, colors, method='statistical', threshold=2.0):
    """
    Filter outliers from 3D points
    
    Args:
        points_3d: Nx3 array of 3D points
        colors: Nx3 array of RGB colors
        method: 'statistical', 'distance', or 'percentile'
        threshold: threshold parameter (depends on method)
    
    Returns:
        filtered_points, filtered_colors, outlier_mask
    """
    print(f"\nOriginal points: {len(points_3d)}")
    
    if method == 'statistical':
        # Remove points based on distance to neighbors
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # Statistical outlier removal
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=threshold)
        
        inlier_mask = np.zeros(len(points_3d), dtype=bool)
        inlier_mask[ind] = True
        
    elif method == 'distance':
        # Remove points beyond certain distance from centroid
        centroid = np.median(points_3d, axis=0)
        distances = np.linalg.norm(points_3d - centroid, axis=1)
        median_dist = np.median(distances)
        
        inlier_mask = distances < (median_dist * threshold)
        
    elif method == 'percentile':
        # Remove extreme percentile in each axis
        lower = np.percentile(points_3d, threshold, axis=0)
        upper = np.percentile(points_3d, 100 - threshold, axis=0)
        
        inlier_mask = np.all((points_3d >= lower) & (points_3d <= upper), axis=1)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    filtered_points = points_3d[inlier_mask]
    filtered_colors = colors[inlier_mask]
    
    print(f"Filtered points: {len(filtered_points)}")
    print(f"Removed: {len(points_3d) - len(filtered_points)} ({100 * (1 - len(filtered_points)/len(points_3d)):.1f}%)")
    
    # Print statistics
    print("\nBounding box:")
    print(f"  X: [{filtered_points[:, 0].min():.2f}, {filtered_points[:, 0].max():.2f}]")
    print(f"  Y: [{filtered_points[:, 1].min():.2f}, {filtered_points[:, 1].max():.2f}]")
    print(f"  Z: [{filtered_points[:, 2].min():.2f}, {filtered_points[:, 2].max():.2f}]")
    
    return filtered_points, filtered_colors, inlier_mask


def visualize_comparison(points_3d, colors, filtered_points, filtered_colors, 
                        image_poses, output_dir):
    """Create side-by-side comparison visualization"""
    fig = plt.figure(figsize=(20, 8))
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                c=colors, s=1, alpha=0.5)
    
    # Plot cameras
    for img_id, pose in image_poses.items():
        R, t = pose['R'], pose['t']
        C = -R.T @ t
        ax1.scatter(C[0], C[1], C[2], c='red', s=100, marker='^')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Original ({len(points_3d)} points)')
    
    # Filtered
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2],
                c=filtered_colors, s=2, alpha=0.6)
    
    # Plot cameras
    for img_id, pose in image_poses.items():
        R, t = pose['R'], pose['t']
        C = -R.T @ t
        ax2.scatter(C[0], C[1], C[2], c='red', s=100, marker='^')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Filtered ({len(filtered_points)} points)')
    
    plt.tight_layout()
    save_path = Path(output_dir) / "comparison_filtered.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison: {save_path}")
    plt.close()


def visualize_open3d_filtered(points_3d, colors, image_poses, cameras, scale=1.0):
    """Visualize filtered reconstruction with Open3D"""
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries = [pcd]
    
    # Get camera parameters
    cam = list(cameras.values())[0]
    w, h = cam['width'], cam['height']
    fx, fy, cx, cy = cam['params'][:4]
    
    # Estimate appropriate frustum size based on point cloud
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_size = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    frustum_scale = bbox_size / 20 * scale  # Adjust this ratio
    
    print(f"\nBounding box size: {bbox_size:.2f}")
    print(f"Frustum scale: {frustum_scale:.2f}")
    
    # Add camera frustums
    for img_id, pose in image_poses.items():
        R, t = pose['R'], pose['t']
        
        # Create camera frustum
        frame = o3d.geometry.LineSet.create_camera_visualization(
            w, h, 
            np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            np.eye(4),
            scale=frustum_scale
        )
        
        # Transform to world coordinates
        C = -R.T @ t
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = C
        frame.transform(T)
        frame.paint_uniform_color([1, 0, 0])  # Red cameras
        
        geometries.append(frame)
    
    # Coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frustum_scale * 2, origin=[0, 0, 0]
    )
    geometries.append(coord_frame)
    
    # Visualize
    print("\nDisplaying filtered Open3D visualization...")
    print("Mouse controls:")
    print("  - Scroll wheel: Zoom in/out")
    print("  - Left-click + drag: Rotate view")
    print("  - Shift + Left-click + drag: Pan")
    print("  - Ctrl + Left-click + drag: Change field of view")
    print("  - R: Reset view")
    print("\nClose the window when done")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Filtered 3D Reconstruction",
        width=1920,
        height=1080,
        point_show_normal=False
    )


def save_filtered_ply(points_3d, colors, output_path):
    """Save filtered point cloud to PLY"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Saved filtered PLY: {output_path}")


def main():
    import sys
    
    # Allow specifying output directory as command line argument
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("output_interval_10")  # Changed to output_interval_10
    
    print("="*80)
    print("LOADING RECONSTRUCTION")
    print("="*80)
    print(f"Reading from: {output_dir}")
    
    # Load data
    cameras, image_poses, points_3d, colors = load_reconstruction(output_dir)
    
    print(f"\nLoaded:")
    print(f"  - {len(cameras)} camera(s)")
    print(f"  - {len(image_poses)} image(s)")
    print(f"  - {len(points_3d)} 3D point(s)")
    
    print("\n" + "="*80)
    print("FILTERING OUTLIERS")
    print("="*80)
    
    # Try multiple filtering methods
    methods = [
        ('percentile', 5.0, "Remove extreme 5% in each axis"),
        ('distance', 3.0, "Remove points >3x median distance from centroid"),
        ('statistical', 2.0, "Statistical outlier removal (std_ratio=2.0)"),
    ]
    
    best_filtered = None
    best_colors = None
    best_method = None
    
    for method, threshold, description in methods:
        print(f"\n--- Method: {method} (threshold={threshold}) ---")
        print(f"Description: {description}")
        
        filtered_points, filtered_colors, mask = filter_outliers(
            points_3d, colors, method=method, threshold=threshold
        )
        
        # Keep the one that removes a reasonable amount (10-30%)
        removal_ratio = 1 - len(filtered_points) / len(points_3d)
        if 0.1 <= removal_ratio <= 0.5:
            if best_filtered is None or len(filtered_points) > len(best_filtered):
                best_filtered = filtered_points
                best_colors = filtered_colors
                best_method = (method, threshold)
    
    # If no good filter found, use percentile
    if best_filtered is None:
        print("\n>>> Using percentile method as default")
        best_filtered, best_colors, _ = filter_outliers(
            points_3d, colors, method='percentile', threshold=5.0
        )
        best_method = ('percentile', 5.0)
    else:
        print(f"\n>>> Selected method: {best_method[0]} (threshold={best_method[1]})")
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create comparison plot
    visualize_comparison(points_3d, colors, best_filtered, best_colors,
                        image_poses, output_dir)
    
    # Save filtered PLY
    save_filtered_ply(best_filtered, best_colors, 
                     output_dir / "reconstruction_filtered.ply")
    
    # Open3D visualization
    print("\n" + "="*80)
    print("INTERACTIVE 3D VISUALIZATION")
    print("="*80)
    visualize_open3d_filtered(best_filtered, best_colors, image_poses, cameras, scale=1.5)
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {output_dir}/comparison_filtered.png")
    print(f"  - {output_dir}/reconstruction_filtered.ply")


if __name__ == "__main__":
    main()
