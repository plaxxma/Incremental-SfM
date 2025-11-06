"""
Incremental Structure-from-Motion (SfM) Pipeline
Complete implementation from scratch using OpenCV, NumPy, SciPy, and Open3D

Author: Computer Vision Assignment
Date: 2025-10-30
"""

import cv2
import numpy as np
import os
import shutil
from pathlib import Path
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
import open3d as o3d
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import copy


class IncrementalSfM:
    """
    Incremental Structure-from-Motion Pipeline
    
    This class implements a complete incremental SfM system including:
    - Frame extraction from video
    - Feature detection and matching
    - Two-view initialization
    - Incremental camera registration
    - Triangulation
    - Bundle adjustment with intrinsics optimization
    """
    
    def __init__(self, video_path: str, output_dir: str = "./output"):
        """
        Initialize the SfM pipeline
        
        Args:
            video_path: Path to input video file
            output_dir: Directory for output files
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        
        # Create output directories
        self._create_directories()
        
        # Initialize data structures
        self.frames = []  # List of frame images
        self.frame_names = []  # List of frame filenames
        self.K = None  # Intrinsic matrix (will be optimized)
        
        # Camera parameters: {frame_idx: {'R': R, 't': t, 'registered': bool}}
        self.cameras = {}
        
        # 3D points: {point_id: {'xyz': np.array, 'color': np.array, 'track': set()}}
        self.points3D = {}
        self.next_point3d_id = 0
        
        # 2D-3D correspondence: {(frame_idx, keypoint_idx): point3d_id}
        self.point2d_to_3d = {}
        
        # Features: {frame_idx: {'keypoints': [], 'descriptors': []}}
        self.features = {}
        
        # Matches: {(frame_i, frame_j): [(kp_idx_i, kp_idx_j), ...]}
        self.matches = {}
        
        # SIFT detector
        self.sift = cv2.SIFT_create()
        
        print(f"Initialized SfM pipeline")
        print(f"Video: {video_path}")
        print(f"Output: {output_dir}")
    
    def _create_directories(self):
        """Create necessary output directories"""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        self.images_dir.mkdir()
        print(f"Created output directories at {self.output_dir}")
    
    # ============================================================================
    # STEP 1: Frame Extraction
    # ============================================================================
    
    def extract_frames(self, frame_interval: int = 30):
        """
        Extract frames from video at specified intervals
        
        Args:
            frame_interval: Extract every Nth frame
        """
        print(f"\n{'='*80}")
        print("STEP 1: Extracting frames from video")
        print(f"{'='*80}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height}, {fps:.2f} fps, {total_frames} total frames")
        print(f"Extracting every {frame_interval} frames")
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Save frame
                frame_name = f"frame_{extracted_count:04d}.jpg"
                frame_path = self.images_dir / frame_name
                cv2.imwrite(str(frame_path), frame)
                
                self.frames.append(frame)
                self.frame_names.append(frame_name)
                extracted_count += 1
                
                if extracted_count % 10 == 0:
                    print(f"  Extracted {extracted_count} frames...")
            
            frame_count += 1
        
        cap.release()
        
        print(f"Extracted {extracted_count} frames to {self.images_dir}")
        
        # Initialize intrinsic matrix based on image size
        self._initialize_intrinsics(width, height)
    
    def _initialize_intrinsics(self, width: int, height: int):
        """
        Initialize intrinsic matrix K using image dimensions
        (No EXIF data used)
        
        Args:
            width: Image width
            height: Image height
        """
        f = max(width, height)
        cx, cy = width / 2.0, height / 2.0
        
        self.K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        print(f"\nInitialized intrinsic matrix K:")
        print(f"  f = {f:.2f}")
        print(f"  cx, cy = {cx:.2f}, {cy:.2f}")
        print(f"  K =\n{self.K}")
    
    # ============================================================================
    # STEP 2: Feature Detection and Matching
    # ============================================================================
    
    def detect_features(self):
        """Detect SIFT features for all frames"""
        print(f"\n{'='*80}")
        print("STEP 2: Detecting SIFT features")
        print(f"{'='*80}")
        
        for idx, frame in enumerate(self.frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            self.features[idx] = {
                'keypoints': keypoints,
                'descriptors': descriptors
            }
            
            if (idx + 1) % 10 == 0:
                print(f"  Detected features for {idx + 1}/{len(self.frames)} frames...")
        
        print(f"Detected features for all {len(self.frames)} frames")
    
    def match_features(self, frame_i: int, frame_j: int, 
                      ratio_thresh: float = 0.75) -> List[Tuple[int, int]]:
        """
        Match features between two frames using ratio test
        
        Args:
            frame_i: First frame index
            frame_j: Second frame index
            ratio_thresh: Lowe's ratio test threshold
            
        Returns:
            List of matched keypoint indices [(idx_i, idx_j), ...]
        """
        desc_i = self.features[frame_i]['descriptors']
        desc_j = self.features[frame_j]['descriptors']
        
        # BF matcher with cross-check
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(desc_i, desc_j, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append((m.queryIdx, m.trainIdx))
        
        return good_matches
    
    def match_with_ransac(self, frame_i: int, frame_j: int) -> Tuple[List, np.ndarray]:
        """
        Match features and refine with RANSAC using Essential matrix
        
        Args:
            frame_i: First frame index
            frame_j: Second frame index
            
        Returns:
            inlier_matches: List of inlier matches
            mask: RANSAC inlier mask
        """
        matches = self.match_features(frame_i, frame_j)
        
        if len(matches) < 8:
            return [], np.array([])
        
        # Get keypoint coordinates
        kps_i = self.features[frame_i]['keypoints']
        kps_j = self.features[frame_j]['keypoints']
        
        pts_i = np.float32([kps_i[m[0]].pt for m in matches])
        pts_j = np.float32([kps_j[m[1]].pt for m in matches])
        
        # RANSAC with Essential matrix
        E, mask = cv2.findEssentialMat(
            pts_i, pts_j, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if mask is None:
            return [], np.array([])
        
        # Filter inlier matches
        mask = mask.ravel()
        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        
        return inlier_matches, mask
    
    # ============================================================================
    # STEP 3: Two-View Initialization
    # ============================================================================
    
    def initialize_two_views(self):
        """
        Initialize reconstruction with first two frames (0 and 1)
        """
        print(f"\n{'='*80}")
        print("STEP 3: Two-view initialization")
        print(f"{'='*80}")
        
        frame_i, frame_j = 0, 1
        
        # Match features with RANSAC
        print(f"Matching frames {frame_i} and {frame_j}...")
        matches, mask = self.match_with_ransac(frame_i, frame_j)
        
        if len(matches) < 50:
            raise ValueError(f"Not enough matches for initialization: {len(matches)}")
        
        print(f"Found {len(matches)} inlier matches")
        self.matches[(frame_i, frame_j)] = matches
        
        # Get matched keypoints
        kps_i = self.features[frame_i]['keypoints']
        kps_j = self.features[frame_j]['keypoints']
        
        pts_i = np.float32([kps_i[m[0]].pt for m in matches]).reshape(-1, 1, 2)
        pts_j = np.float32([kps_j[m[1]].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate Essential matrix and recover pose
        E, mask = cv2.findEssentialMat(pts_i, pts_j, self.K, method=cv2.RANSAC)
        _, R, t, mask = cv2.recoverPose(E, pts_i, pts_j, self.K, mask=mask)
        
        print(f"Recovered pose:")
        print(f"  R =\n{R}")
        print(f"  t =\n{t.ravel()}")
        
        # Set first camera at origin
        self.cameras[frame_i] = {
            'R': np.eye(3, dtype=np.float64),
            't': np.zeros((3, 1), dtype=np.float64),
            'registered': True
        }
        
        # Set second camera with estimated pose
        self.cameras[frame_j] = {
            'R': R.copy(),
            't': t.copy(),
            'registered': True
        }
        
        # Triangulate initial 3D points
        print(f"Triangulating initial 3D points...")
        self._triangulate_two_views(frame_i, frame_j, matches)
        
        print(f"Created {len(self.points3D)} initial 3D points")
        
        # Local bundle adjustment on first two frames
        print(f"\nPerforming local bundle adjustment...")
        self._bundle_adjustment_local([frame_i, frame_j])
        
        # Save initial reconstruction
        self._save_reconstruction()
        
        # Visualize
        self._visualize_2d_matches(frame_i, frame_j, "initial_matches.png")
        self._visualize_3d_points("initial_3d.png")
    
    def _triangulate_two_views(self, frame_i: int, frame_j: int, 
                               matches: List[Tuple[int, int]]):
        """
        Triangulate 3D points from two views
        
        Args:
            frame_i: First frame index
            frame_j: Second frame index
            matches: List of matched keypoint indices
        """
        R_i = self.cameras[frame_i]['R']
        t_i = self.cameras[frame_i]['t']
        R_j = self.cameras[frame_j]['R']
        t_j = self.cameras[frame_j]['t']
        
        # Projection matrices
        P_i = self.K @ np.hstack([R_i, t_i])
        P_j = self.K @ np.hstack([R_j, t_j])
        
        kps_i = self.features[frame_i]['keypoints']
        kps_j = self.features[frame_j]['keypoints']
        
        for match_idx, (idx_i, idx_j) in enumerate(matches):
            # Get 2D points
            pt_i = np.array(kps_i[idx_i].pt, dtype=np.float32).reshape(2, 1)
            pt_j = np.array(kps_j[idx_j].pt, dtype=np.float32).reshape(2, 1)
            
            # Triangulate
            pt_4d = cv2.triangulatePoints(P_i, P_j, pt_i, pt_j)
            pt_3d = pt_4d[:3] / pt_4d[3]
            pt_3d = pt_3d.ravel()
            
            # Check if point is in front of both cameras
            if not self._is_point_in_front(pt_3d, R_i, t_i, R_j, t_j):
                continue
            
            # Check reprojection error
            reproj_error_i = self._compute_reprojection_error(pt_3d, pt_i.ravel(), R_i, t_i)
            reproj_error_j = self._compute_reprojection_error(pt_3d, pt_j.ravel(), R_j, t_j)
            
            if reproj_error_i > 10.0 or reproj_error_j > 10.0:
                continue
            
            # Get color from first frame
            x, y = int(pt_i[0]), int(pt_i[1])
            h, w = self.frames[frame_i].shape[:2]
            if 0 <= x < w and 0 <= y < h:
                color = self.frames[frame_i][y, x][::-1]  # BGR to RGB
            else:
                color = np.array([128, 128, 128], dtype=np.uint8)
            
            # Create 3D point
            point_id = self.next_point3d_id
            self.points3D[point_id] = {
                'xyz': pt_3d.astype(np.float64),
                'color': color,
                'track': {(frame_i, idx_i), (frame_j, idx_j)}
            }
            self.next_point3d_id += 1
            
            # Update 2D-3D correspondence
            self.point2d_to_3d[(frame_i, idx_i)] = point_id
            self.point2d_to_3d[(frame_j, idx_j)] = point_id
    
    def _is_point_in_front(self, pt_3d: np.ndarray, R_i: np.ndarray, t_i: np.ndarray,
                          R_j: np.ndarray, t_j: np.ndarray) -> bool:
        """Check if 3D point is in front of both cameras"""
        # Transform to camera coordinates
        pt_cam_i = R_i @ pt_3d.reshape(3, 1) + t_i
        pt_cam_j = R_j @ pt_3d.reshape(3, 1) + t_j
        
        return pt_cam_i[2, 0] > 0 and pt_cam_j[2, 0] > 0
    
    def _compute_reprojection_error(self, pt_3d: np.ndarray, pt_2d: np.ndarray,
                                   R: np.ndarray, t: np.ndarray) -> float:
        """Compute reprojection error for a 3D point"""
        # Project 3D point to 2D
        pt_cam = R @ pt_3d.reshape(3, 1) + t
        pt_proj = self.K @ pt_cam
        pt_proj = pt_proj[:2] / pt_proj[2]
        pt_proj = pt_proj.ravel()
        
        # Compute error
        error = np.linalg.norm(pt_proj - pt_2d)
        return error
    
    # ============================================================================
    # STEP 4: Bundle Adjustment
    # ============================================================================
    
    def _bundle_adjustment_local(self, frame_indices: List[int], 
                                max_nfev: int = 100):
        """
        Perform local bundle adjustment on specified frames
        Optimizes: K (intrinsics), R, t (extrinsics), and 3D points
        
        Args:
            frame_indices: List of frame indices to optimize
            max_nfev: Maximum number of function evaluations
        """
        print(f"  Running local BA on frames {frame_indices}...")
        
        # Collect observations
        observations = []  # [(frame_idx, point_id, pt_2d), ...]
        point_ids_set = set()
        
        for frame_idx in frame_indices:
            for (f_idx, kp_idx), point_id in self.point2d_to_3d.items():
                if f_idx == frame_idx and point_id in self.points3D:
                    kp = self.features[f_idx]['keypoints'][kp_idx]
                    pt_2d = np.array(kp.pt, dtype=np.float64)
                    observations.append((f_idx, point_id, pt_2d))
                    point_ids_set.add(point_id)
        
        if len(observations) < 10:
            print(f"  Too few observations ({len(observations)}), skipping BA")
            return
        
        print(f"  Observations: {len(observations)}, Points: {len(point_ids_set)}, Cameras: {len(frame_indices)}")
        
        # Prepare optimization parameters
        frame_indices_sorted = sorted(frame_indices)
        point_ids_sorted = sorted(point_ids_set)
        
        frame_idx_to_opt = {f_idx: i for i, f_idx in enumerate(frame_indices_sorted)}
        point_id_to_opt = {p_id: i for i, p_id in enumerate(point_ids_sorted)}
        
        # Pack parameters: [fx, fy, cx, cy, cam0_params, ..., camN_params, pt0, ..., ptM]
        # Camera params: angle-axis (3) + translation (3) = 6 params per camera
        n_cams = len(frame_indices_sorted)
        n_pts = len(point_ids_sorted)
        
        x0 = []
        
        # Pack intrinsics: [fx, fy, cx, cy]
        x0.extend([self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]])
        
        # Pack camera parameters
        for f_idx in frame_indices_sorted:
            R = self.cameras[f_idx]['R']
            t = self.cameras[f_idx]['t'].ravel()
            rvec, _ = cv2.Rodrigues(R)
            x0.extend(rvec.ravel())
            x0.extend(t)
        
        # Pack 3D points
        for p_id in point_ids_sorted:
            x0.extend(self.points3D[p_id]['xyz'])
        
        x0 = np.array(x0, dtype=np.float64)
        
        # Setup bounds
        bounds_low = [-np.inf] * len(x0)
        bounds_high = [np.inf] * len(x0)
        
        # Intrinsics bounds (positive focal lengths)
        bounds_low[0] = 100.0  # fx
        bounds_low[1] = 100.0  # fy
        bounds_high[0] = 10000.0
        bounds_high[1] = 10000.0
        
        bounds = (bounds_low, bounds_high)
        
        # Build sparsity structure
        rows = []
        cols = []
        
        for obs_idx, (f_idx, p_id, pt_2d) in enumerate(observations):
            cam_idx = frame_idx_to_opt[f_idx]
            pt_idx = point_id_to_opt[p_id]
            
            # Each observation contributes 2 residuals (x, y)
            residual_idx_base = obs_idx * 2
            
            # Intrinsics affect all observations
            rows.extend([residual_idx_base, residual_idx_base + 1])
            cols.extend([0, 0])  # fx
            rows.extend([residual_idx_base, residual_idx_base + 1])
            cols.extend([1, 1])  # fy
            rows.extend([residual_idx_base, residual_idx_base + 1])
            cols.extend([2, 2])  # cx
            rows.extend([residual_idx_base, residual_idx_base + 1])
            cols.extend([3, 3])  # cy
            
            # Camera parameters (6 params per camera)
            cam_param_start = 4 + cam_idx * 6
            for i in range(6):
                rows.extend([residual_idx_base, residual_idx_base + 1])
                cols.extend([cam_param_start + i, cam_param_start + i])
            
            # 3D point parameters (3 params per point)
            pt_param_start = 4 + n_cams * 6 + pt_idx * 3
            for i in range(3):
                rows.extend([residual_idx_base, residual_idx_base + 1])
                cols.extend([pt_param_start + i, pt_param_start + i])
        
        n_residuals = len(observations) * 2
        n_vars = len(x0)
        
        jac_sparsity = coo_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)),
            shape=(n_residuals, n_vars)
        )
        
        # Define residual function
        def residual_func(params):
            residuals = []
            
            # Unpack intrinsics
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            K_opt = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            
            # Unpack cameras
            cameras_opt = {}
            for i, f_idx in enumerate(frame_indices_sorted):
                start = 4 + i * 6
                rvec = params[start:start+3]
                tvec = params[start+3:start+6]
                R_opt, _ = cv2.Rodrigues(rvec)
                cameras_opt[f_idx] = (R_opt, tvec)
            
            # Unpack points
            points_opt = {}
            for i, p_id in enumerate(point_ids_sorted):
                start = 4 + n_cams * 6 + i * 3
                points_opt[p_id] = params[start:start+3]
            
            # Compute residuals
            for f_idx, p_id, pt_2d in observations:
                R_opt, t_opt = cameras_opt[f_idx]
                pt_3d = points_opt[p_id]
                
                # Project
                pt_cam = R_opt @ pt_3d + t_opt
                if pt_cam[2] < 0.01:
                    pt_cam[2] = 0.01
                
                pt_proj = K_opt @ pt_cam
                pt_proj = pt_proj[:2] / pt_proj[2]
                
                # Residual
                residuals.extend(pt_proj - pt_2d)
            
            return np.array(residuals, dtype=np.float64)
        
        # Run optimization
        huber_delta = 1.0
        result = least_squares(
            residual_func, x0,
            method='trf',
            loss='huber', f_scale=huber_delta,
            max_nfev=max_nfev,
            jac_sparsity=jac_sparsity,
            bounds=bounds,
            verbose=0
        )
        
        print(f"  BA completed: cost={result.cost:.4f}, iterations={result.nfev}")
        
        # Update parameters
        params_opt = result.x
        
        # Update intrinsics
        self.K[0, 0] = params_opt[0]  # fx
        self.K[1, 1] = params_opt[1]  # fy
        self.K[0, 2] = params_opt[2]  # cx
        self.K[1, 2] = params_opt[3]  # cy
        
        # Update cameras
        for i, f_idx in enumerate(frame_indices_sorted):
            start = 4 + i * 6
            rvec = params_opt[start:start+3]
            tvec = params_opt[start+3:start+6]
            R_opt, _ = cv2.Rodrigues(rvec)
            self.cameras[f_idx]['R'] = R_opt
            self.cameras[f_idx]['t'] = tvec.reshape(3, 1)
        
        # Update 3D points
        for i, p_id in enumerate(point_ids_sorted):
            start = 4 + n_cams * 6 + i * 3
            self.points3D[p_id]['xyz'] = params_opt[start:start+3]
    
    def _bundle_adjustment_global(self, max_nfev: int = 50):
        """
        Perform global bundle adjustment on all registered cameras
        Optimizes: K, all R, all t, and all 3D points
        
        Args:
            max_nfev: Maximum number of function evaluations
        """
        registered_frames = [f_idx for f_idx, cam in self.cameras.items() 
                           if cam['registered']]
        
        if len(registered_frames) < 2:
            return
        
        print(f"  Running global BA on {len(registered_frames)} cameras and {len(self.points3D)} points...")
        
        self._bundle_adjustment_local(registered_frames, max_nfev=max_nfev)
    
    # ============================================================================
    # STEP 5: Incremental Reconstruction
    # ============================================================================
    
    def incremental_reconstruction(self):
        """
        Incrementally add new frames to the reconstruction
        """
        print(f"\n{'='*80}")
        print("STEP 5: Incremental reconstruction")
        print(f"{'='*80}")
        
        n_frames = len(self.frames)
        
        for frame_idx in range(2, n_frames):
            print(f"\n--- Adding frame {frame_idx}/{n_frames-1} ---")
            
            # Try to register this frame
            success = self._register_new_frame(frame_idx)
            
            if not success:
                print(f"  Failed to register frame {frame_idx}, skipping...")
                continue
            
            # Triangulate new points
            self._triangulate_new_points(frame_idx)
            
            # Global bundle adjustment
            print(f"  Performing global BA...")
            self._bundle_adjustment_global(max_nfev=50)
            
            # Save reconstruction
            self._save_reconstruction()
            
            # Visualize progress
            if frame_idx % 5 == 0 or frame_idx == n_frames - 1:
                self._visualize_3d_points(f"reconstruction_frame_{frame_idx:04d}.png")
            
            print(f"  Current reconstruction: {len([c for c in self.cameras.values() if c['registered']])} cameras, "
                  f"{len(self.points3D)} points")
        
        print(f"\n{'='*80}")
        print("Incremental reconstruction completed")
        print(f"{'='*80}")
        print(f"Final: {len([c for c in self.cameras.values() if c['registered']])} cameras, "
              f"{len(self.points3D)} 3D points")
    
    def _register_new_frame(self, frame_idx: int) -> bool:
        """
        Register a new frame using PnP
        
        Args:
            frame_idx: Frame index to register
            
        Returns:
            True if registration successful
        """
        # Find 2D-3D correspondences with previous frame
        prev_frame_idx = frame_idx - 1
        
        # Match with previous frame
        matches, _ = self.match_with_ransac(prev_frame_idx, frame_idx)
        
        if len(matches) < 20:
            print(f"  Not enough matches with previous frame: {len(matches)}")
            return False
        
        self.matches[(prev_frame_idx, frame_idx)] = matches
        
        # Find 2D points in new frame that correspond to known 3D points
        # Strategy: Check which keypoints in previous frame have 3D points,
        # then use their matched keypoints in current frame for PnP
        pts_2d = []
        pts_3d = []
        correspondence_list = []  # Store (kp_idx_prev, kp_idx_curr, point_id)
        
        print(f"  Searching for 2D-3D correspondences...")
        print(f"  Total matches: {len(matches)}")
        print(f"  Existing 3D points: {len(self.points3D)}")
        print(f"  Frame {prev_frame_idx} has {len([k for k in self.point2d_to_3d.keys() if k[0] == prev_frame_idx])} keypoints with 3D points")
        
        for kp_idx_prev, kp_idx_curr in matches:
            key_prev = (prev_frame_idx, kp_idx_prev)
            if key_prev in self.point2d_to_3d:
                point_id = self.point2d_to_3d[key_prev]
                if point_id in self.points3D:
                    # Get 2D point in current frame
                    kp = self.features[frame_idx]['keypoints'][kp_idx_curr]
                    pt_2d = np.array(kp.pt, dtype=np.float64)
                    
                    # Get 3D point
                    pt_3d = self.points3D[point_id]['xyz']
                    
                    pts_2d.append(pt_2d)
                    pts_3d.append(pt_3d)
                    correspondence_list.append((kp_idx_prev, kp_idx_curr, point_id))
        
        if len(pts_2d) < 20:
            print(f"  Not enough 2D-3D correspondences: {len(pts_2d)}")
            return False
        
        pts_2d = np.array(pts_2d, dtype=np.float32)
        pts_3d = np.array(pts_3d, dtype=np.float32)
        
        print(f"  Found {len(pts_2d)} 2D-3D correspondences")
        
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.K, None,
            iterationsCount=1000,
            reprojectionError=8.0,
            confidence=0.99
        )
        
        if not success or inliers is None or len(inliers) < 15:
            print(f"  PnP failed or too few inliers")
            return False
        
        print(f"  PnP successful: {len(inliers)} inliers")
        
        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Register camera
        self.cameras[frame_idx] = {
            'R': R.copy(),
            't': tvec.copy(),
            'registered': True
        }
        
        # Update 2D-3D correspondences for inliers
        print(f"  Updating 2D-3D correspondences for {len(inliers)} inliers...")
        for inlier_idx in inliers.ravel():
            kp_idx_prev, kp_idx_curr, point_id = correspondence_list[inlier_idx]
            key_curr = (frame_idx, kp_idx_curr)
            self.point2d_to_3d[key_curr] = point_id
            self.points3D[point_id]['track'].add(key_curr)
        
        print(f"  Frame {frame_idx} registered successfully")
        print(f"  Frame {frame_idx} now has {len([k for k in self.point2d_to_3d.keys() if k[0] == frame_idx])} keypoints with 3D points")
        
        return True
    
    def _triangulate_new_points(self, frame_idx: int):
        """
        Triangulate new 3D points between current and previous frame
        
        This function finds 2D-2D matches that don't have 3D points yet,
        and triangulates them using the known camera poses.
        
        Args:
            frame_idx: Current frame index
        """
        prev_frame_idx = frame_idx - 1
        
        if (prev_frame_idx, frame_idx) not in self.matches:
            return
        
        matches = self.matches[(prev_frame_idx, frame_idx)]
        
        R_prev = self.cameras[prev_frame_idx]['R']
        t_prev = self.cameras[prev_frame_idx]['t']
        R_curr = self.cameras[frame_idx]['R']
        t_curr = self.cameras[frame_idx]['t']
        
        P_prev = self.K @ np.hstack([R_prev, t_prev])
        P_curr = self.K @ np.hstack([R_curr, t_curr])
        
        kps_prev = self.features[prev_frame_idx]['keypoints']
        kps_curr = self.features[frame_idx]['keypoints']
        
        print(f"  Triangulating new points from {len(matches)} matches...")
        
        new_points_count = 0
        skipped_already_has_3d = 0
        skipped_behind_camera = 0
        skipped_large_error = 0
        
        for kp_idx_prev, kp_idx_curr in matches:
            # Skip if already has 3D point
            # This is crucial: we only triangulate NEW 2D-2D correspondences
            key_prev = (prev_frame_idx, kp_idx_prev)
            key_curr = (frame_idx, kp_idx_curr)
            
            if key_prev in self.point2d_to_3d or key_curr in self.point2d_to_3d:
                skipped_already_has_3d += 1
                continue
            
            # Get 2D points
            pt_prev = np.array(kps_prev[kp_idx_prev].pt, dtype=np.float32).reshape(2, 1)
            pt_curr = np.array(kps_curr[kp_idx_curr].pt, dtype=np.float32).reshape(2, 1)
            
            # Triangulate
            pt_4d = cv2.triangulatePoints(P_prev, P_curr, pt_prev, pt_curr)
            pt_3d = pt_4d[:3] / pt_4d[3]
            pt_3d = pt_3d.ravel()
            
            # Check validity
            if not self._is_point_in_front(pt_3d, R_prev, t_prev, R_curr, t_curr):
                skipped_behind_camera += 1
                continue
            
            reproj_error_prev = self._compute_reprojection_error(pt_3d, pt_prev.ravel(), R_prev, t_prev)
            reproj_error_curr = self._compute_reprojection_error(pt_3d, pt_curr.ravel(), R_curr, t_curr)
            
            if reproj_error_prev > 10.0 or reproj_error_curr > 10.0:
                skipped_large_error += 1
                continue
            
            # Get color
            x, y = int(pt_curr[0]), int(pt_curr[1])
            h, w = self.frames[frame_idx].shape[:2]
            if 0 <= x < w and 0 <= y < h:
                color = self.frames[frame_idx][y, x][::-1]
            else:
                color = np.array([128, 128, 128], dtype=np.uint8)
            
            # Create 3D point
            point_id = self.next_point3d_id
            self.points3D[point_id] = {
                'xyz': pt_3d.astype(np.float64),
                'color': color,
                'track': {key_prev, key_curr}
            }
            self.next_point3d_id += 1
            
            # Update correspondences
            # IMPORTANT: Both frame (i-1) and frame i keypoints now have 3D points
            # Frame (i-1) keypoint might not have had 3D point before - that's OK!
            self.point2d_to_3d[key_prev] = point_id
            self.point2d_to_3d[key_curr] = point_id
            
            new_points_count += 1
        
        print(f"  Triangulated {new_points_count} new points")
        print(f"    - Skipped {skipped_already_has_3d} (already have 3D points)")
        print(f"    - Skipped {skipped_behind_camera} (behind camera)")
        print(f"    - Skipped {skipped_large_error} (large reprojection error)")
        print(f"  Frame {prev_frame_idx} now has {len([k for k in self.point2d_to_3d.keys() if k[0] == prev_frame_idx])} keypoints with 3D points")
        print(f"  Frame {frame_idx} now has {len([k for k in self.point2d_to_3d.keys() if k[0] == frame_idx])} keypoints with 3D points")
    
    # ============================================================================
    # File I/O: COLMAP Format
    # ============================================================================
    
    def _save_reconstruction(self):
        """Save reconstruction in COLMAP format"""
        self._save_cameras()
        self._save_images()
        self._save_points3d()
        self._save_visibility()
    
    def _save_cameras(self):
        """Save cameras.txt in COLMAP format"""
        filepath = self.output_dir / "cameras.txt"
        
        with open(filepath, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("# Number of cameras: 1\n")
            
            h, w = self.frames[0].shape[:2]
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            
            f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
    
    def _save_images(self):
        """Save images.txt in COLMAP format"""
        filepath = self.output_dir / "images.txt"
        
        with open(filepath, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len([c for c in self.cameras.values() if c['registered']])}\n")
            
            for frame_idx in sorted(self.cameras.keys()):
                if not self.cameras[frame_idx]['registered']:
                    continue
                
                R = self.cameras[frame_idx]['R']
                t = self.cameras[frame_idx]['t'].ravel()
                
                # Convert to quaternion
                quat = self._rotation_matrix_to_quaternion(R)
                qw, qx, qy, qz = quat
                
                image_id = frame_idx + 1
                name = self.frame_names[frame_idx]
                
                f.write(f"{image_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {name}\n")
                
                # Write 2D points
                points2d_line = []
                kps = self.features[frame_idx]['keypoints']
                for kp_idx, kp in enumerate(kps):
                    key = (frame_idx, kp_idx)
                    if key in self.point2d_to_3d:
                        point3d_id = self.point2d_to_3d[key]
                        points2d_line.append(f"{kp.pt[0]} {kp.pt[1]} {point3d_id}")
                    else:
                        points2d_line.append(f"{kp.pt[0]} {kp.pt[1]} -1")
                
                f.write(" ".join(points2d_line) + "\n")
    
    def _save_points3d(self):
        """Save points3D.txt in COLMAP format"""
        filepath = self.output_dir / "points3D.txt"
        
        with open(filepath, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {len(self.points3D)}\n")
            
            for point_id, point_data in self.points3D.items():
                xyz = point_data['xyz']
                color = point_data['color']
                
                # Compute average reprojection error
                error = 0.0
                track = point_data['track']
                if len(track) > 0:
                    errors = []
                    for frame_idx, kp_idx in track:
                        if frame_idx in self.cameras and self.cameras[frame_idx]['registered']:
                            kp = self.features[frame_idx]['keypoints'][kp_idx]
                            pt_2d = np.array(kp.pt)
                            R = self.cameras[frame_idx]['R']
                            t = self.cameras[frame_idx]['t']
                            err = self._compute_reprojection_error(xyz, pt_2d, R, t)
                            errors.append(err)
                    if len(errors) > 0:
                        error = np.mean(errors)
                
                f.write(f"{point_id} {xyz[0]} {xyz[1]} {xyz[2]} "
                       f"{color[0]} {color[1]} {color[2]} {error}")
                
                # Write track
                for frame_idx, kp_idx in track:
                    image_id = frame_idx + 1
                    f.write(f" {image_id} {kp_idx}")
                
                f.write("\n")
    
    def _save_visibility(self):
        """Save vis.txt (visibility information)"""
        filepath = self.output_dir / "vis.txt"
        
        with open(filepath, 'w') as f:
            f.write("# Visibility information\n")
            f.write(f"# Number of 3D points: {len(self.points3D)}\n")
            
            for point_id, point_data in self.points3D.items():
                track = point_data['track']
                visible_images = sorted(set([frame_idx + 1 for frame_idx, _ in track]))
                
                f.write(f"{point_id} {len(visible_images)}")
                for image_id in visible_images:
                    f.write(f" {image_id}")
                f.write("\n")
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z]"""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    # ============================================================================
    # Visualization
    # ============================================================================
    
    def _visualize_2d_matches(self, frame_i: int, frame_j: int, filename: str):
        """Visualize 2D feature matches between two frames"""
        if (frame_i, frame_j) not in self.matches:
            return
        
        matches = self.matches[(frame_i, frame_j)]
        
        img_i = self.frames[frame_i]
        img_j = self.frames[frame_j]
        
        kps_i = self.features[frame_i]['keypoints']
        kps_j = self.features[frame_j]['keypoints']
        
        # Create match objects for cv2.drawMatches
        match_objs = [cv2.DMatch(m[0], m[1], 0) for m in matches[:100]]  # Limit to 100 for clarity
        
        img_matches = cv2.drawMatches(
            img_i, kps_i, img_j, kps_j, match_objs, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), img_matches)
        print(f"  Saved 2D matches visualization: {filepath}")
    
    def _visualize_3d_points(self, filename: str):
        """Visualize 3D points and camera poses"""
        if len(self.points3D) == 0:
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D points
        points = np.array([p['xyz'] for p in self.points3D.values()])
        colors = np.array([p['color'] for p in self.points3D.values()]) / 255.0
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=colors, marker='.', s=1, alpha=0.5)
        
        # Plot camera poses
        for frame_idx, cam in self.cameras.items():
            if not cam['registered']:
                continue
            
            R = cam['R']
            t = cam['t'].ravel()
            
            # Camera center in world coordinates
            C = -R.T @ t
            
            # Camera axes
            axis_length = 0.5
            axes = np.eye(3) * axis_length
            axes_world = (R.T @ axes.T).T + C
            
            # Plot camera center
            ax.scatter(C[0], C[1], C[2], c='red', marker='o', s=50)
            
            # Plot camera axes
            ax.plot([C[0], axes_world[0, 0]], [C[1], axes_world[0, 1]], [C[2], axes_world[0, 2]], 'r-', linewidth=2)
            ax.plot([C[0], axes_world[1, 0]], [C[1], axes_world[1, 1]], [C[2], axes_world[1, 2]], 'g-', linewidth=2)
            ax.plot([C[0], axes_world[2, 0]], [C[1], axes_world[2, 1]], [C[2], axes_world[2, 2]], 'b-', linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Reconstruction ({len(self.points3D)} points, {len([c for c in self.cameras.values() if c["registered"]])} cameras)')
        
        # Equal aspect ratio
        all_coords = points
        max_range = np.array([all_coords[:, 0].max() - all_coords[:, 0].min(),
                             all_coords[:, 1].max() - all_coords[:, 1].min(),
                             all_coords[:, 2].max() - all_coords[:, 2].min()]).max() / 2.0
        
        mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
        mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
        mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved 3D visualization: {filepath}")
    
    def visualize_final_open3d(self):
        """Create final visualization using Open3D"""
        print(f"\n{'='*80}")
        print("Creating final Open3D visualization")
        print(f"{'='*80}")
        
        # Create point cloud
        points = np.array([p['xyz'] for p in self.points3D.values()])
        colors = np.array([p['color'] for p in self.points3D.values()]) / 255.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create camera frustums
        geometries = [pcd]
        
        for frame_idx, cam in self.cameras.items():
            if not cam['registered']:
                continue
            
            R = cam['R']
            t = cam['t'].ravel()
            
            # Camera center
            C = -R.T @ t
            
            # Create coordinate frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            
            # Transform to camera pose
            T = np.eye(4)
            T[:3, :3] = R.T
            T[:3, 3] = C
            frame.transform(T)
            
            geometries.append(frame)
        
        # Visualize
        print("Displaying Open3D visualization...")
        print("Close the window to continue")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Incremental SfM Result",
            width=1280,
            height=720
        )
        
        # Save point cloud
        pcd_path = self.output_dir / "reconstruction.ply"
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        print(f"Saved point cloud: {pcd_path}")
    
    # ============================================================================
    # Main Pipeline
    # ============================================================================
    
    def run(self, frame_interval: int = 30):
        """
        Run the complete incremental SfM pipeline
        
        Args:
            frame_interval: Extract every Nth frame from video
        """
        print(f"\n{'#'*80}")
        print("INCREMENTAL STRUCTURE-FROM-MOTION PIPELINE")
        print(f"{'#'*80}\n")
        
        # Step 1: Extract frames
        self.extract_frames(frame_interval=frame_interval)
        
        # Step 2: Detect features
        self.detect_features()
        
        # Step 3: Initialize with two views
        self.initialize_two_views()
        
        # Step 4: Incremental reconstruction
        self.incremental_reconstruction()
        
        # Step 5: Final visualization
        self._visualize_3d_points("final_reconstruction.png")
        self.visualize_final_open3d()
        
        print(f"\n{'#'*80}")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'#'*80}")
        print(f"\nResults saved in: {self.output_dir}")
        print(f"  - cameras.txt")
        print(f"  - images.txt")
        print(f"  - points3D.txt")
        print(f"  - vis.txt")
        print(f"  - reconstruction.ply")
        print(f"  - Various visualization images")


def main():
    """Main entry point"""
    # Video path
    video_path = r"C:\CHAEEUN_DATA\3-2\기초컴퓨터비전\assignment2\assignment2_new\Video\IMG_1578.MOV"
    
    # Frame interval to use
    frame_interval = 10  # Change this to adjust sampling rate
    
    # Output directory based on frame interval
    output_dir = rf"C:\CHAEEUN_DATA\3-2\기초컴퓨터비전\assignment2\assignment2_new\output_interval_{frame_interval}"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Create SfM pipeline
    sfm = IncrementalSfM(video_path, output_dir)
    
    # Run pipeline
    # Adjust frame_interval based on video length
    # Smaller interval = more frames = more detailed reconstruction (but slower)
    print(f"\n{'='*80}")
    print(f"Running with frame_interval = {frame_interval}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    sfm.run(frame_interval=frame_interval)


if __name__ == "__main__":
    main()
