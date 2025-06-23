import cv2, numpy as np, torch, torch.nn.functional as F, pickle, os
from copy import copy
from tqdm import tqdm
from scipy.spatial import distance
from ultralytics import YOLO
# from bytetrack import BYTETracker

def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

def postprocess_ball(feature_map, scale=2):
    feature_map *= 255
    feature_map = feature_map.reshape((360, 640))
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                               maxRadius=7)
    x,y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0]*scale
            y = circles[0][0][1]*scale
    return x, y

def postprocess_court(heatmap, scale=2, low_thresh=155, min_radius=10, max_radius=30):
    x_pred, y_pred = None, None
    ret, heatmap = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=min_radius,
                               maxRadius=max_radius)
    if circles is not None:
        x_pred = circles[0][0][0] * scale
        y_pred = circles[0][0][1] * scale
    return x_pred, y_pred

def interpolate_points(p1, p2, alpha):
    if p1[0] is None or p2[0] is None:
        return (None, None) 
    x = p1[0] * (1 - alpha) + p2[0] * alpha
    y = p1[1] * (1 - alpha) + p2[1] * alpha
    return (x, y)

def infer_ball(frames, model, device, stub_path = 'stubs/ball_stub.pkl'):
    if os.path.isfile(stub_path):
        with open(stub_path, 'rb') as stub:
            loaded_stub = pickle.load(stub)
            return loaded_stub[0], loaded_stub[1]

    height = 360
    width = 640
    dists = [-1]*2
    ball_track = [(None,None)]*2
    for num in tqdm(range(2, len(frames))):
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num-1], (width, height))
        img_preprev = cv2.resize(frames[num-2], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess_ball(output)
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)

    with open(stub_path, 'wb') as stub:
        pickle.dump((ball_track, dists), stub)

    return ball_track, dists 

def infer_court(frames, width, height, model, device, step = 5, stub_path = 'stubs/court_stub.pkl'):
    if os.path.isfile(stub_path):
        with open(stub_path, 'rb') as stub:
            return pickle.load(stub)
    
    inferred_points = {}
    
    for idx in tqdm(range(0, len(frames), step)):
        image = frames[idx]
        img = cv2.resize(image, (width, height))
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        out = model(inp.float().to(device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess_court(heatmap, low_thresh=170, max_radius=25)
            
            points.append((x_pred, y_pred))

        inferred_points[idx] = points

    with open(stub_path, 'wb') as stub:
        pickle.dump(inferred_points, stub)
    
    return inferred_points

def interpolate_court_points_per_frame(frames, inferred_points):
    interpolated_points_per_frame = []
    key_idxs = sorted(inferred_points.keys())

    for idx in range(len(frames)):
        prev_idx = max([k for k in key_idxs if k <= idx], default=key_idxs[0])
        next_idx = min([k for k in key_idxs if k >= idx], default=key_idxs[-1])

        if prev_idx == next_idx:
            interp_points = inferred_points[prev_idx]
        else:
            alpha = (idx - prev_idx) / (next_idx - prev_idx)
            interp_points = [
                interpolate_points(inferred_points[prev_idx][j], inferred_points[next_idx][j], alpha)
                for j in range(len(inferred_points[prev_idx]))
            ]

        interpolated_points_per_frame.append(interp_points)

    return interpolated_points_per_frame

def draw_court(frames, interpolated_points_per_frame):
    frames_upd = []

    for idx, image in enumerate(frames):
        interp_points = interpolated_points_per_frame[idx]

        corner_points = []
        for pt_idx, p in enumerate(interp_points):
            if not None in p:
                if len(corner_points) < 4:
                    corner_points.append(p)
                # image = cv2.circle(image, (int(p[0]), int(p[1])),
                #                   radius=0, color=(0, 0, 255), thickness=10)
                # image = cv2.putText(image, f"{pt_idx + 1}", (int(p[0]), int(p[1])),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for idx, p1 in enumerate(corner_points[:-1]):
            for p2 in corner_points[idx + 1:]:
                image = cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (222, 74, 69), 5)

        frames_upd.append(image)

    return frames_upd

def remove_outliers(ball_track, dists, max_dist=100, min_dist=1.0, min_static_frames=5):
    """Basic outlier removal (legacy function)"""
    outliers = set(np.where(np.array(dists) > max_dist)[0])
    static_counter = 0
    
    for i in range(1, len(ball_track)):
        pt = ball_track[i]
        prev_pt = ball_track[i-1]
        
        # If current point is None, reset
        if pt[0] is None or prev_pt[0] is None:
            static_counter = 0
            continue
        # Distance between this point and previous
        dx = pt[0] - prev_pt[0]
        dy = pt[1] - prev_pt[1]
        dist = np.sqrt(dx**2 + dy**2)
        # If stuck
        if dist < min_dist:
            static_counter += 1
        else:
            static_counter = 0
        # If stuck too long, remove this point
        if static_counter >= min_static_frames:
            ball_track[i] = (None, None)
            static_counter = 0

    for i in outliers:
        if i < len(ball_track):
            ball_track[i] = (None, None)
    return ball_track

def remove_outliers_advanced(ball_track, dists, max_velocity=150, max_acceleration=50, 
                           smoothing_window=5, confidence_threshold=0.7, debug=False):
    """
    Advanced outlier removal with velocity and acceleration constraints
    
    Args:
        ball_track: List of (x, y) ball positions or (None, None)
        dists: List of distances between consecutive points
        max_velocity: Maximum allowed velocity (pixels per frame)
        max_acceleration: Maximum allowed acceleration change
        smoothing_window: Window size for smoothing
        confidence_threshold: Minimum confidence to keep a point
        debug: Print debug information
    """
    if len(ball_track) < 3:
        return ball_track
    
    cleaned_track = ball_track.copy()
    
    # Step 1: Remove obvious distance outliers
    if debug:
        print("🔍 Step 1: Removing distance outliers...")
    
    distance_threshold = np.percentile([d for d in dists if d > 0], 95)  # 95th percentile
    distance_outliers = 0
    
    for i, dist in enumerate(dists):
        if dist > distance_threshold:
            if i < len(cleaned_track):
                if debug:
                    print(f"   Removed distance outlier at frame {i}: dist={dist:.1f}")
                cleaned_track[i] = (None, None)
                distance_outliers += 1
    
    # Step 2: Velocity-based filtering
    if debug:
        print("🔍 Step 2: Velocity-based filtering...")
    
    velocities = []
    velocity_outliers = 0
    
    for i in range(1, len(cleaned_track)):
        prev_pt = cleaned_track[i-1]
        curr_pt = cleaned_track[i]
        
        if prev_pt[0] is not None and curr_pt[0] is not None:
            vx = curr_pt[0] - prev_pt[0]
            vy = curr_pt[1] - prev_pt[1]
            v_mag = np.sqrt(vx**2 + vy**2)
            velocities.append(v_mag)
            
            if v_mag > max_velocity:
                if debug:
                    print(f"   Removed velocity outlier at frame {i}: v={v_mag:.1f}")
                cleaned_track[i] = (None, None)
                velocity_outliers += 1
        else:
            velocities.append(0)
    
    # Step 3: Acceleration-based filtering
    if debug:
        print("🔍 Step 3: Acceleration-based filtering...")
    
    acceleration_outliers = 0
    
    for i in range(1, len(velocities)):
        if velocities[i] > 0 and velocities[i-1] > 0:
            acceleration = abs(velocities[i] - velocities[i-1])
            
            if acceleration > max_acceleration:
                frame_idx = i + 1  # +1 because velocities array is offset
                if frame_idx < len(cleaned_track):
                    if debug:
                        print(f"   Removed acceleration outlier at frame {frame_idx}: a={acceleration:.1f}")
                    cleaned_track[frame_idx] = (None, None)
                    acceleration_outliers += 1
    
    # Step 4: Temporal consistency check
    if debug:
        print("🔍 Step 4: Temporal consistency check...")
    
    consistency_outliers = 0
    
    for i in range(2, len(cleaned_track) - 2):
        curr_pt = cleaned_track[i]
        if curr_pt[0] is None:
            continue
        
        # Check if current point is consistent with surrounding points
        surrounding_points = []
        for j in range(max(0, i-2), min(len(cleaned_track), i+3)):
            if j != i and cleaned_track[j][0] is not None:
                surrounding_points.append(cleaned_track[j])
        
        if len(surrounding_points) >= 2:
            # Calculate expected position based on surrounding points
            avg_x = np.mean([pt[0] for pt in surrounding_points])
            avg_y = np.mean([pt[1] for pt in surrounding_points])
            
            # Check distance from expected position
            dist_from_expected = np.sqrt((curr_pt[0] - avg_x)**2 + (curr_pt[1] - avg_y)**2)
            
            # If too far from expected position, mark as outlier
            if dist_from_expected > max_velocity:
                if debug:
                    print(f"   Removed consistency outlier at frame {i}: dist_from_expected={dist_from_expected:.1f}")
                cleaned_track[i] = (None, None)
                consistency_outliers += 1
    
    # Step 5: Apply smoothing to remaining points
    if debug:
        print("🔍 Step 5: Applying smoothing...")
    
    smoothed_track = apply_smoothing(cleaned_track, window_size=smoothing_window)
    
    if debug:
        total_removed = distance_outliers + velocity_outliers + acceleration_outliers + consistency_outliers
        total_valid = sum(1 for pt in ball_track if pt[0] is not None)
        remaining_valid = sum(1 for pt in smoothed_track if pt[0] is not None)
        print(f"📊 Outlier removal summary:")
        print(f"   Distance outliers: {distance_outliers}")
        print(f"   Velocity outliers: {velocity_outliers}")
        print(f"   Acceleration outliers: {acceleration_outliers}")
        print(f"   Consistency outliers: {consistency_outliers}")
        print(f"   Total removed: {total_removed}")
        print(f"   Valid points: {total_valid} → {remaining_valid}")
    
    return smoothed_track

def apply_smoothing(ball_track, window_size=5):
    """
    Apply temporal smoothing to ball trajectory to reduce jitter
    """
    if window_size < 3:
        return ball_track
    
    smoothed_track = ball_track.copy()
    half_window = window_size // 2
    
    for i in range(len(ball_track)):
        curr_pt = ball_track[i]
        if curr_pt[0] is None:
            continue
        
        # Collect valid points in window
        window_points = []
        for j in range(max(0, i - half_window), min(len(ball_track), i + half_window + 1)):
            if ball_track[j][0] is not None:
                window_points.append(ball_track[j])
        
        # If we have enough points for smoothing
        if len(window_points) >= 3:
            # Use median smoothing (more robust than mean)
            smoothed_x = np.median([pt[0] for pt in window_points])
            smoothed_y = np.median([pt[1] for pt in window_points])
            smoothed_track[i] = (smoothed_x, smoothed_y)
    
    return smoothed_track

def interpolate_missing_points(ball_track, max_gap=10):
    """
    Interpolate missing points in ball trajectory for small gaps
    
    Args:
        ball_track: List of ball positions
        max_gap: Maximum gap size to interpolate (frames)
    """
    interpolated_track = ball_track.copy()
    
    i = 0
    while i < len(interpolated_track):
        if interpolated_track[i][0] is None:
            # Find the start and end of the gap
            gap_start = i
            gap_end = i
            
            # Find end of gap
            while gap_end < len(interpolated_track) and interpolated_track[gap_end][0] is None:
                gap_end += 1
            
            gap_size = gap_end - gap_start
            
            # Only interpolate small gaps
            if gap_size <= max_gap:
                # Find valid points before and after gap
                before_pt = None
                after_pt = None
                
                if gap_start > 0:
                    before_pt = interpolated_track[gap_start - 1]
                if gap_end < len(interpolated_track):
                    after_pt = interpolated_track[gap_end]
                
                # Interpolate if we have both endpoints and they are valid
                if (before_pt is not None and before_pt[0] is not None and 
                    after_pt is not None and after_pt[0] is not None):
                    
                    for j in range(gap_start, gap_end):
                        alpha = (j - gap_start + 1) / (gap_size + 1)
                        interp_x = before_pt[0] * (1 - alpha) + after_pt[0] * alpha
                        interp_y = before_pt[1] * (1 - alpha) + after_pt[1] * alpha
                        interpolated_track[j] = (interp_x, interp_y)
            
            i = gap_end
        else:
            i += 1
    
    return interpolated_track

def draw_ball(frames, ball_track, trace):
    output_frames = []
    
    for num in range(len(frames)):
        frame = frames[num]
        for i in range(trace):
            if (num-i > 0):
                if ball_track[num-i][0]:
                    x = int(ball_track[num-i][0])
                    y = int(ball_track[num-i][1])
                    frame = cv2.circle(frame, (x,y), radius=0, color=(127, 0, 255), thickness=10-i)
                else:
                    break
        output_frames.append(frame)
    
    return output_frames

def save_video(output_path, fps, frames):
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def frame_homographies(frames, corner_points, homography_obj):
    homographies = []
    for frame_idx, frame in enumerate(frames):
        input_points = np.array(corner_points[frame_idx][:4], dtype=np.float32)
        H = homography_obj.compute_homography(input_points)
        homographies.append(H)
    return homographies

def map_ball_points(ball_track, homography_obj, homographies):
    mapped_ball_points = []
    for frame_idx, H in enumerate(homographies):
        if ball_track[frame_idx][0]:
            ball_pt = (ball_track[frame_idx][0], ball_track[frame_idx][1])
            ball_mapped = homography_obj.map_point(ball_pt, H)
            mapped_ball_points.append(ball_mapped)
        else:
            mapped_ball_points.append(None)
    return mapped_ball_points

def draw_virtual_court(court_img, mapped_ball_points):
    court_frames = []
    for frame_idx, ball_mapped in enumerate(mapped_ball_points):
        curr_frame = copy(court_img)
        if ball_mapped is not None:
            x = int(ball_mapped[0])
            y = int(ball_mapped[1])
            cv2.circle(curr_frame, (x, y), radius=10, color=(127, 0, 255), thickness=-1)
        court_frames.append(curr_frame)
    return court_frames

def combine_frames(live_court_frames, frames, virtual_frames):
    combined_frames = []

    for frame_idx in range(len(frames)):
        frame_disp = frames[frame_idx].copy()
        court_disp = virtual_frames[frame_idx]
        live_warp_disp = live_court_frames[frame_idx]

        # First: match heights
        h_target = max(frame_disp.shape[0], court_disp.shape[0], live_warp_disp.shape[0])

        def pad_to_height(img, target_h):
            h_img = img.shape[0]
            if h_img < target_h:
                top_pad = (target_h - h_img) // 2
                bottom_pad = target_h - h_img - top_pad
                img = cv2.copyMakeBorder(img, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            return img

        frame_disp = pad_to_height(frame_disp, h_target)
        court_disp = pad_to_height(court_disp, h_target)
        live_warp_disp = pad_to_height(live_warp_disp, h_target)

        # Combine horizontally: left to right
        combined = np.hstack((live_warp_disp, frame_disp, court_disp))
        combined_frames.append(combined)

    return combined_frames

def compute_warp_points(corner_points_video, homography_obj):
    court_no_border_pts = np.array([
        [0, 0],
        [homography_obj.court_width, 0],
        [homography_obj.court_width, homography_obj.court_height],
        [0, homography_obj.court_height]
    ], dtype=np.float32)
    
    H = cv2.findHomography(corner_points_video, court_no_border_pts)[0]
    H_inv = np.linalg.inv(H)
    
    border_left = homography_obj.offset_x
    border_top = homography_obj.offset_y
    
    full_width = homography_obj.court_width + 2 * border_left
    full_height = homography_obj.court_height + 2 * border_top
    
    target_pts_with_border = np.array([
        [0, 0],
        [full_width, 0],
        [full_width, full_height],
        [0, full_height]
    ], dtype=np.float32)
    
    desired_pts_in_court = np.array([
        [-border_left, -border_top],
        [homography_obj.court_width + border_left, -border_top],
        [homography_obj.court_width + border_left, homography_obj.court_height + border_top],
        [-border_left, homography_obj.court_height + border_top]
    ], dtype=np.float32)
    
    desired_pts_in_video = cv2.perspectiveTransform(np.array([desired_pts_in_court]), H_inv)[0]
    
    return desired_pts_in_video, target_pts_with_border

def compute_live_warp_points(corner_points_video, homography_obj):
    court_w = homography_obj.court_width
    court_h = homography_obj.court_height
    offset_x = homography_obj.offset_x
    offset_y = homography_obj.offset_y

    output_width = court_w + 2 * offset_x
    output_height = court_h + 2 * offset_y

    dst_pts = np.array([
        [offset_x, offset_y],                              # top-left
        [offset_x + court_w, offset_y],                    # top-right
        [offset_x, offset_y + court_h],                    # bottom-left
        [offset_x + court_w, offset_y + court_h]           # bottom-right
    ], dtype=np.float32)

    src_pts_video = np.array([
        corner_points_video[0],   # top-left in video
        corner_points_video[1],   # top-right in video
        corner_points_video[2],   # bottom-left in video
        corner_points_video[3]    # bottom-right in video
    ], dtype=np.float32)

    return src_pts_video, dst_pts, output_width, output_height

def build_live_court_view(frames, interpolated_points_per_frame, homography_obj):
    live_court_frames = []
    for frame_idx, frame in enumerate(frames):
        corner_points_video = interpolated_points_per_frame[frame_idx]

        src_pts_video, dst_pts, output_width, output_height = compute_live_warp_points(corner_points_video, homography_obj)

        warp_matrix = cv2.getPerspectiveTransform(src_pts_video, dst_pts)

        # USE the correct output_width / output_height here!
        warped_frame = cv2.warpPerspective(frame, warp_matrix, (output_width, output_height))

        live_court_frames.append(warped_frame)
    return live_court_frames

def bbox_feet(bbox):
    x1, _, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return(int((x1 + x2) / 2), int((y1 + y2) / 2))

def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def tether_players_to_points(player_tracker, frame_points, player_detections, homography, homographies, frames, interpolated_points_per_frame):
    _, distances = player_tracker.choose_players(frame_points, player_detections[0], homography, homographies[0])
    for frame_num, frame in enumerate(frames):
        for track_id, min_distance, min_point in distances:
            point = interpolated_points_per_frame[frame_num][min_point]
            if track_id in player_detections[frame_num].keys():
                frame = cv2.line(frame, bbox_feet(player_detections[frame_num][track_id]), (int(point[0]), int(point[1])), (255, 0, 0))

def detect_trajectory_changes(mapped_ball_points, velocity_threshold=15.0):
    """
    Detect potential bounce points by analyzing velocity changes in the mapped ball trajectory
    """
    vx_list = []
    vy_list = []
    v_norm_list = []
    
    for i in range(1, len(mapped_ball_points)):
        pt_prev = mapped_ball_points[i-1]
        pt_curr = mapped_ball_points[i]
        
        if pt_prev is None or pt_curr is None:
            vx_list.append(0)
            vy_list.append(0)
            v_norm_list.append(0)
            continue
        
        vx = pt_curr[0] - pt_prev[0]
        vy = pt_curr[1] - pt_prev[1]
        v_norm = np.sqrt(vx**2 + vy**2)
        
        vx_list.append(vx)
        vy_list.append(vy)
        v_norm_list.append(v_norm)
    
    # Detect significant velocity changes that indicate bounces
    change_frames = []
    
    for i in range(2, len(v_norm_list)-2):
        if v_norm_list[i] == 0:
            continue
            
        # Look for sudden direction changes or velocity spikes
        dv_before = abs(v_norm_list[i] - v_norm_list[i-1])
        dv_after = abs(v_norm_list[i+1] - v_norm_list[i])
        
        if dv_before > velocity_threshold or dv_after > velocity_threshold:
            change_frames.append(i+1)  # +1 because of lag
    
    return set(change_frames)

def detect_bounces_advanced(mapped_ball_points, min_segment_length=5, angle_threshold=np.pi/4, 
                          use_multiple_methods=True, debug=False):
    """
    Advanced bounce detection using multiple methods with adjustable sensitivity
    
    Args:
        mapped_ball_points: List of (x,y) ball positions or None
        min_segment_length: Minimum frames between bounces (prevents noise)
        angle_threshold: Minimum angle change to consider a bounce (radians)
        use_multiple_methods: Whether to combine multiple detection methods
        debug: Print debug information
    """
    bounce_frames = []
    
    # First, find all non-None points
    valid_points = []
    valid_indices = []
    
    for i, point in enumerate(mapped_ball_points):
        if point is not None:
            valid_points.append(point)
            valid_indices.append(i)
    
    if len(valid_points) < 6:  # Need at least 6 points for analysis
        return bounce_frames
    
    # Method 1: Direction change analysis
    direction_bounces = []
    for i in range(2, len(valid_points) - 2):
        curr_idx = valid_indices[i]
        
        # Get points for analysis
        prev2_point = valid_points[i-2]
        prev_point = valid_points[i-1]
        curr_point = valid_points[i]
        next_point = valid_points[i+1]
        next2_point = valid_points[i+2]
        
        # Calculate direction vectors (using wider window for stability)
        v1 = np.array([curr_point[0] - prev2_point[0], curr_point[1] - prev2_point[1]])
        v2 = np.array([next2_point[0] - curr_point[0], next2_point[1] - curr_point[1]])
        
        # Skip if vectors are too small
        if np.linalg.norm(v1) < 2 or np.linalg.norm(v2) < 2:
            continue
        
        # Calculate angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        # Check for significant direction change
        if angle > angle_threshold:
            direction_bounces.append(curr_idx)
            if debug:
                print(f"Direction bounce at frame {curr_idx}, angle: {np.degrees(angle):.1f}°")
    
    # Method 2: Velocity change analysis
    velocity_bounces = []
    velocities = []
    
    for i in range(1, len(valid_points)):
        prev_point = valid_points[i-1]
        curr_point = valid_points[i]
        
        vx = curr_point[0] - prev_point[0]
        vy = curr_point[1] - prev_point[1]
        v_mag = np.sqrt(vx**2 + vy**2)
        velocities.append(v_mag)
    
    # Detect velocity spikes/drops
    if len(velocities) > 4:
        velocity_threshold = np.std(velocities) * 1.5  # Adaptive threshold
        
        for i in range(2, len(velocities) - 2):
            curr_v = velocities[i]
            prev_v = velocities[i-1]
            next_v = velocities[i+1]
            
            # Look for sudden velocity changes
            dv_before = abs(curr_v - prev_v)
            dv_after = abs(next_v - curr_v)
            
            if dv_before > velocity_threshold or dv_after > velocity_threshold:
                frame_idx = valid_indices[i+1]  # +1 because velocities array is offset
                velocity_bounces.append(frame_idx)
                if debug:
                    print(f"Velocity bounce at frame {frame_idx}, dv: {max(dv_before, dv_after):.1f}")
    
    # Method 3: Y-direction analysis (vertical bounces)
    y_bounces = []
    if len(valid_points) > 4:
        y_coords = [pt[1] for pt in valid_points]
        
        # Find local minima in Y direction (assuming Y increases downward)
        for i in range(2, len(y_coords) - 2):
            y_prev2 = y_coords[i-2]
            y_prev = y_coords[i-1] 
            y_curr = y_coords[i]
            y_next = y_coords[i+1]
            y_next2 = y_coords[i+2]
            
            # Check if current point is a local minimum
            if (y_curr < y_prev and y_curr < y_next and 
                y_curr < y_prev2 and y_curr < y_next2):
                
                # Additional check: ensure it's a significant minimum
                min_depth = min(abs(y_curr - y_prev), abs(y_curr - y_next))
                if min_depth > 5:  # Minimum depth threshold
                    frame_idx = valid_indices[i]
                    y_bounces.append(frame_idx)
                    if debug:
                        print(f"Y-direction bounce at frame {frame_idx}, depth: {min_depth:.1f}")
    
    # Combine methods
    if use_multiple_methods:
        # Combine all detected bounces
        all_bounces = set(direction_bounces + velocity_bounces + y_bounces)
        bounce_frames = sorted(list(all_bounces))
    else:
        # Use only direction change method (most reliable)
        bounce_frames = direction_bounces
    
    # Filter out bounces that are too close together
    filtered_bounces = []
    for bounce in bounce_frames:
        if not filtered_bounces or bounce - filtered_bounces[-1] > min_segment_length:
            filtered_bounces.append(bounce)
    
    if debug:
        print(f"Total bounces detected: {len(filtered_bounces)} at frames: {filtered_bounces}")
    
    return filtered_bounces

def detect_bounces_with_sensitivity(mapped_ball_points, sensitivity="medium"):
    """
    Convenience function to detect bounces with different sensitivity levels
    
    Args:
        mapped_ball_points: List of ball positions
        sensitivity: "low", "medium", "high", or "max"
    """
    if sensitivity == "low":
        return detect_bounces_advanced(mapped_ball_points, 
                                     min_segment_length=8, 
                                     angle_threshold=np.pi/3, 
                                     use_multiple_methods=False)
    elif sensitivity == "medium":
        return detect_bounces_advanced(mapped_ball_points, 
                                     min_segment_length=5, 
                                     angle_threshold=np.pi/4, 
                                     use_multiple_methods=True)
    elif sensitivity == "high":
        return detect_bounces_advanced(mapped_ball_points, 
                                     min_segment_length=3, 
                                     angle_threshold=np.pi/6, 
                                     use_multiple_methods=True)
    elif sensitivity == "max":
        return detect_bounces_advanced(mapped_ball_points, 
                                     min_segment_length=2, 
                                     angle_threshold=np.pi/8, 
                                     use_multiple_methods=True)
    else:
        raise ValueError(f"Unknown sensitivity: {sensitivity}")

def create_straight_trajectory(mapped_ball_points, bounce_frames=None):
    """
    Create a straight-line trajectory between bounce points to correct for 3D perspective distortion
    """
    if bounce_frames is None:
        bounce_frames = detect_bounces_advanced(mapped_ball_points)
    
    # Add start and end frames
    bounce_frames = [0] + sorted(bounce_frames) + [len(mapped_ball_points) - 1]
    
    straightened_points = [None] * len(mapped_ball_points)
    
    # Process each segment between bounces
    for i in range(len(bounce_frames) - 1):
        start_frame = bounce_frames[i]
        end_frame = bounce_frames[i + 1]
        
        # Find the actual ball positions at segment endpoints
        start_point = None
        end_point = None
        
        # Find start point
        for f in range(start_frame, min(start_frame + 10, end_frame)):
            if mapped_ball_points[f] is not None:
                start_point = mapped_ball_points[f]
                start_frame = f
                break
        
        # Find end point
        for f in range(end_frame, max(end_frame - 10, start_frame), -1):
            if mapped_ball_points[f] is not None:
                end_point = mapped_ball_points[f]
                end_frame = f
                break
        
        # If we have both points, create straight line interpolation
        if start_point is not None and end_point is not None and start_frame < end_frame:
            for f in range(start_frame, end_frame + 1):
                # Linear interpolation between start and end points
                alpha = (f - start_frame) / (end_frame - start_frame)
                x = start_point[0] * (1 - alpha) + end_point[0] * alpha
                y = start_point[1] * (1 - alpha) + end_point[1] * alpha
                straightened_points[f] = (x, y)
        else:
            # If we can't find both endpoints, use original points
            for f in range(start_frame, end_frame + 1):
                if mapped_ball_points[f] is not None:
                    straightened_points[f] = mapped_ball_points[f]
    
    return straightened_points

def build_straight_trajectory(mapped_ball_points, change_frames=None):
    """
    Legacy function name for compatibility
    """
    return create_straight_trajectory(mapped_ball_points, change_frames)

def draw_virtual_court_comparison(court_img, original_points, corrected_points, show_bounces=True, bounce_frames=None):
    """
    Draw both original (curved) and corrected (straight) trajectories for comparison
    """
    court_frames = []
    
    if bounce_frames is None:
        bounce_frames = detect_bounces_advanced(original_points)
    
    for frame_idx in range(len(original_points)):
        curr_frame = court_img.copy()
        
        # Draw original trajectory in red
        if original_points[frame_idx] is not None:
            x_orig = int(original_points[frame_idx][0])
            y_orig = int(original_points[frame_idx][1])
            cv2.circle(curr_frame, (x_orig, y_orig), radius=6, color=(0, 0, 255), thickness=-1)  # Red
        
        # Draw corrected trajectory in green
        if corrected_points[frame_idx] is not None:
            x_corr = int(corrected_points[frame_idx][0])
            y_corr = int(corrected_points[frame_idx][1])
            cv2.circle(curr_frame, (x_corr, y_corr), radius=8, color=(0, 255, 0), thickness=2)   # Green outline
        
        # Mark bounce points in yellow
        if show_bounces and frame_idx in bounce_frames:
            if corrected_points[frame_idx] is not None:
                x_bounce = int(corrected_points[frame_idx][0])
                y_bounce = int(corrected_points[frame_idx][1])
                cv2.circle(curr_frame, (x_bounce, y_bounce), radius=12, color=(0, 255, 255), thickness=3)  # Yellow
        
        # Add legend
        cv2.putText(curr_frame, "Red: Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(curr_frame, "Green: Corrected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if show_bounces:
            cv2.putText(curr_frame, "Yellow: Bounces", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        court_frames.append(curr_frame)
    
    return court_frames

def analyze_bounce_detection(mapped_ball_points, output_path="bounce_analysis.png"):
    """
    Create a visualization showing bounce detection results with different sensitivity levels
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not available - skipping bounce analysis visualization")
        return False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Bounce Detection Analysis - Different Sensitivity Levels', fontsize=16, fontweight='bold')
    
    sensitivities = ['low', 'medium', 'high', 'max']
    colors = ['blue', 'green', 'orange', 'red']
    
    # Extract valid points for plotting
    valid_points = [(i, pt) for i, pt in enumerate(mapped_ball_points) if pt is not None]
    if len(valid_points) < 10:
        print("⚠️ Not enough valid ball points for bounce analysis")
        return False
    
    frames, points = zip(*valid_points)
    x_coords = [pt[0] for pt in points]
    y_coords = [pt[1] for pt in points]
    
    for idx, (sensitivity, color) in enumerate(zip(sensitivities, colors)):
        ax = axes[idx // 2, idx % 2]
        
        # Plot trajectory
        ax.plot(x_coords, y_coords, 'gray', alpha=0.5, linewidth=1, label='Ball trajectory')
        ax.scatter(x_coords, y_coords, c='lightblue', s=10, alpha=0.6)
        
        # Detect bounces with current sensitivity
        bounces = detect_bounces_with_sensitivity(mapped_ball_points, sensitivity)
        
        # Mark detected bounces
        bounce_x = []
        bounce_y = []
        for bounce_frame in bounces:
            if mapped_ball_points[bounce_frame] is not None:
                pt = mapped_ball_points[bounce_frame]
                bounce_x.append(pt[0])
                bounce_y.append(pt[1])
        
        if bounce_x:
            ax.scatter(bounce_x, bounce_y, c=color, s=100, edgecolor='black', 
                      linewidth=2, label=f'Bounces ({len(bounces)})', zorder=5)
        
        ax.set_title(f'{sensitivity.title()} Sensitivity\n{len(bounces)} bounces detected', 
                    fontweight='bold')
        ax.set_xlabel('Court X')
        ax.set_ylabel('Court Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Invert Y axis if needed (assuming Y increases downward)
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Bounce analysis saved to: {output_path}")
    
    # Also print summary
    print("\n🎾 Bounce Detection Summary:")
    print("=" * 40)
    for sensitivity in sensitivities:
        bounces = detect_bounces_with_sensitivity(mapped_ball_points, sensitivity)
        print(f"{sensitivity.ljust(8)}: {len(bounces)} bounces detected")
    
    return True