import os, cv2, pickle, torch, numpy as np
from copy import copy
from tqdm import tqdm
from utils.ball_model import BallModel
from utils.geometry_utils import euclidean_distance as distance
from utils.homography_utils import Homography

def infer_ball(frames, device, ball_model_path, stub_path = 'stubs/ball_stub.pkl'):
    if os.path.isfile(stub_path):
        with open(stub_path, 'rb') as stub:
            loaded_stub = pickle.load(stub)
            return loaded_stub[0], loaded_stub[1]

    print("Inferring ball position...")

    # Set up ball-tracking model
    ball_model = BallModel()
    ball_model.load_state_dict(torch.load(ball_model_path, map_location=device))
    ball_model = ball_model.to(device)
    ball_model.eval()

    height = 360
    width = 640
    scale = frames[0].shape[1] / width
    dists = [-1]*2
    ball_track = [(None,None)]*2
    for num in tqdm(range(2, len(frames)), desc="Inferring ball position", unit="frame"):
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num-1], (width, height))
        img_preprev = cv2.resize(frames[num-2], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out = ball_model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess_ball(output, scale = scale)
        
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)

    with open(stub_path, 'wb') as stub:
        pickle.dump((ball_track, dists), stub)

    return ball_track, dists 

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

def remove_outliers(ball_track, dists, max_velocity=150, max_acceleration=50, smoothing_window=5, confidence_threshold=0.7):
    if len(ball_track) < 3: return ball_track
    cleaned_track = ball_track.copy()
    
    # Distance thresholding
    distance_threshold = np.percentile([d for d in dists if d > 0], 95)    
    for i, dist in enumerate(dists):
        if dist > distance_threshold:
            if i < len(cleaned_track):
                cleaned_track[i] = (None, None)

    # Velocity thresholding
    velocities = []    
    for i in range(1, len(cleaned_track)):
        prev_pt = cleaned_track[i-1]
        curr_pt = cleaned_track[i]
        
        if prev_pt[0] is not None and curr_pt[0] is not None:
            vx = curr_pt[0] - prev_pt[0]
            vy = curr_pt[1] - prev_pt[1]
            v_mag = np.sqrt(vx**2 + vy**2)
            velocities.append(v_mag)
            
            if v_mag > max_velocity:
                cleaned_track[i] = (None, None)
        else:
            velocities.append(0)

    # Acceleration thresholding    
    for i in range(1, len(velocities)):
        if velocities[i] > 0 and velocities[i-1] > 0:
            acceleration = abs(velocities[i] - velocities[i-1])
            
            if acceleration > max_acceleration:
                frame_idx = i + 1
                if frame_idx < len(cleaned_track):
                    cleaned_track[frame_idx] = (None, None)

    # Check for consistency with neighbor points    
    for i in range(2, len(cleaned_track) - 2):
        curr_pt = cleaned_track[i]
        if curr_pt[0] is None:
            continue
        
        surrounding_points = []
        for j in range(max(0, i-2), min(len(cleaned_track), i+3)):
            if j != i and cleaned_track[j][0] is not None:
                surrounding_points.append(cleaned_track[j])
        
        if len(surrounding_points) >= 2:
            avg_x = np.mean([pt[0] for pt in surrounding_points])
            avg_y = np.mean([pt[1] for pt in surrounding_points])
            
            dist_from_expected = np.sqrt((curr_pt[0] - avg_x)**2 + (curr_pt[1] - avg_y)**2)
            if dist_from_expected > max_velocity:
                cleaned_track[i] = (None, None)

    smoothed_track = apply_smoothing(cleaned_track, window_size=smoothing_window)
    return smoothed_track

def apply_smoothing(ball_track, window_size=5):
    if window_size < 3:
        return ball_track
    
    smoothed_track = ball_track.copy()
    half_window = window_size // 2
    
    for i in range(len(ball_track)):
        curr_pt = ball_track[i]
        if curr_pt[0] is None:
            continue
        
        window_points = []
        for j in range(max(0, i - half_window), min(len(ball_track), i + half_window + 1)):
            if ball_track[j][0] is not None:
                window_points.append(ball_track[j])
        
        # maybe median
        if len(window_points) >= 3:
            smoothed_x = np.mean([pt[0] for pt in window_points])
            smoothed_y = np.mean([pt[1] for pt in window_points])
            smoothed_track[i] = (smoothed_x, smoothed_y)
    
    return smoothed_track

def interpolate_missing_points(ball_track, max_gap=10):
    interpolated_track = ball_track.copy()
    
    i = 0
    while i < len(interpolated_track):
        if interpolated_track[i][0] is None:
            gap_start = i
            gap_end = i
            
            while gap_end < len(interpolated_track) and interpolated_track[gap_end][0] is None:
                gap_end += 1
            
            gap_size = gap_end - gap_start
            
            if gap_size <= max_gap:
                before_pt = None
                after_pt = None
                
                if gap_start > 0:
                    before_pt = interpolated_track[gap_start - 1]
                if gap_end < len(interpolated_track):
                    after_pt = interpolated_track[gap_end]
                
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

def map_ball_points(ball_track, homography_obj, homographies):
    mapped_ball_points = []
    for frame_idx, H in enumerate(homographies):
        if ball_track[frame_idx]:
            ball_pt = (ball_track[frame_idx][0], ball_track[frame_idx][1])
            ball_mapped = homography_obj.map_point(ball_pt, H)
            mapped_ball_points.append(ball_mapped)
        else:
            mapped_ball_points.append(None)
    return mapped_ball_points

def detect_bounces(mapped_ball_points, min_segment_length=5, angle_threshold=np.pi/4):
    bounce_frames = []
    
    valid_points = []
    valid_indices = []
    
    for i, point in enumerate(mapped_ball_points):
        if point is not None:
            valid_points.append(point)
            valid_indices.append(i)
    
    if len(valid_points) < 6:  
        return bounce_frames
    
    # Direction change
    direction_bounces = []
    for i in range(2, len(valid_points) - 2):
        curr_idx = valid_indices[i]
        
        prev2_point = valid_points[i-2]
        prev_point = valid_points[i-1]
        curr_point = valid_points[i]
        next_point = valid_points[i+1]
        next2_point = valid_points[i+2]
        
        v1 = np.array([curr_point[0] - prev2_point[0], curr_point[1] - prev2_point[1]])
        v2 = np.array([next2_point[0] - curr_point[0], next2_point[1] - curr_point[1]])
        
        if np.linalg.norm(v1) < 2 or np.linalg.norm(v2) < 2:
            continue
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        if angle > angle_threshold:
            direction_bounces.append(curr_idx)
            
    # Velocity change
    velocity_bounces = []
    velocities = []
    
    for i in range(1, len(valid_points)):
        prev_point = valid_points[i-1]
        curr_point = valid_points[i]
        
        vx = curr_point[0] - prev_point[0]
        vy = curr_point[1] - prev_point[1]
        v_mag = np.sqrt(vx**2 + vy**2)
        velocities.append(v_mag)
    
    # Velocity spikes/drops
    if len(velocities) > 4:
        velocity_threshold = np.std(velocities) * 1.5
        
        for i in range(2, len(velocities) - 2):
            curr_v = velocities[i]
            prev_v = velocities[i-1]
            next_v = velocities[i+1]
            
            dv_before = abs(curr_v - prev_v)
            dv_after = abs(next_v - curr_v)
            
            if dv_before > velocity_threshold or dv_after > velocity_threshold:
                frame_idx = valid_indices[i+1]  
                velocity_bounces.append(frame_idx)
    
    # Y-direction analysis
    y_bounces = []
    if len(valid_points) > 4:
        y_coords = [pt[1] for pt in valid_points]
        
        for i in range(2, len(y_coords) - 2):
            y_prev2 = y_coords[i-2]
            y_prev = y_coords[i-1] 
            y_curr = y_coords[i]
            y_next = y_coords[i+1]
            y_next2 = y_coords[i+2]
            
            if (y_curr < y_prev and y_curr < y_next and 
                y_curr < y_prev2 and y_curr < y_next2):
                
                min_depth = min(abs(y_curr - y_prev), abs(y_curr - y_next))
                if min_depth > 5:
                    frame_idx = valid_indices[i]
                    y_bounces.append(frame_idx)
                    
    all_bounces = set(direction_bounces + velocity_bounces + y_bounces)
    bounce_frames = sorted(list(all_bounces))
    
    # Filter nearby bounces
    filtered_bounces = []
    bounce_frames.reverse()
    for bounce in bounce_frames:
        if not filtered_bounces or filtered_bounces[-1] - bounce > min_segment_length:
            filtered_bounces.append(bounce)
    
    return filtered_bounces

def create_straight_trajectory(mapped_ball_points, bounce_frames=None):
    if bounce_frames is None:
        bounce_frames = detect_bounces(mapped_ball_points)
    
    bounce_frames = [0] + sorted(bounce_frames) + [len(mapped_ball_points) - 1]
    
    straightened_points = [None] * len(mapped_ball_points)
    
    for i in range(len(bounce_frames) - 1):
        start_frame = bounce_frames[i]
        end_frame = bounce_frames[i + 1]
        
        start_point = None
        end_point = None
        
        for f in range(start_frame, min(start_frame + 10, end_frame)):
            if mapped_ball_points[f] is not None:
                start_point = mapped_ball_points[f]
                start_frame = f
                break
        
        for f in range(end_frame, max(end_frame - 10, start_frame), -1):
            if mapped_ball_points[f] is not None:
                end_point = mapped_ball_points[f]
                end_frame = f
                break
        
        if start_point is not None and end_point is not None and start_frame < end_frame:
            for f in range(start_frame, end_frame + 1):
                alpha = (f - start_frame) / (end_frame - start_frame)
                x = start_point[0] * (1 - alpha) + end_point[0] * alpha
                y = start_point[1] * (1 - alpha) + end_point[1] * alpha
                straightened_points[f] = (x, y)
        else:
            for f in range(start_frame, end_frame + 1):
                if mapped_ball_points[f] is not None:
                    straightened_points[f] = mapped_ball_points[f]
    
    return straightened_points

def draw_ball(frames, ball_track, trace, scale):
    output_frames = []
    
    for num in range(len(frames)):
        frame = frames[num]
        for i in range(trace):
            if (num-i > 0):
                if ball_track[num-i][0]:
                    x = int(ball_track[num-i][0])
                    y = int(ball_track[num-i][1])
                    frame = cv2.circle(frame, (x,y), radius=0, color=(127, 0, 255), thickness=int(7 * scale))
                else:
                    break
        output_frames.append(frame)
    
    return output_frames

def adjust_ball_height(
    mapped_ball_points, mapped_player_detections, ref_court_points, 
    fade_dist=20, cap_offset=10, player_x_thresh=80
):
    adjusted_track = []
    for frame_idx, ball in enumerate(mapped_ball_points):
        if ball is None or ball[0] is None or ball[1] is None:
            adjusted_track.append(ball)
            continue

        x0, y0 = ref_court_points[0]
        x1, y1 = ref_court_points[1]
        bx, by = ball

        baseline_dx = x1 - x0
        baseline_dy = y1 - y0
        denominator = np.sqrt(baseline_dx**2 + baseline_dy**2)
        if denominator == 0:
            adjusted_track.append(ball)
            continue
        numerator = baseline_dx * (by - y0) - baseline_dy * (bx - x0)
        signed_dist = numerator / denominator  

        if signed_dist < 0:
            by_new = by + cap_offset
        elif 0 <= signed_dist < fade_dist:
            weight = 1 - (signed_dist / fade_dist)
            by_new = by + cap_offset * weight
        else:
            by_new = by

        top_player_id = None
        min_feet_y = float('inf')
        min_feet_x = None
        min_feet_signed_dist = None
        for track_id, (feet_x, feet_y) in mapped_player_detections[frame_idx].items():
            # Compute signed distance for player's feet
            num = baseline_dx * (feet_y - y0) - baseline_dy * (feet_x - x0)
            player_signed_dist = num / denominator
            if feet_y < min_feet_y:
                min_feet_y = feet_y
                min_feet_x = feet_x
                min_feet_signed_dist = player_signed_dist
                top_player_id = track_id

        # Cap ball to player's feet
        if (
            top_player_id is not None and
            abs(bx - min_feet_x) < player_x_thresh and
            min_feet_signed_dist is not None and
            min_feet_signed_dist < fade_dist
        ):
            by_new = max(by_new, min_feet_y)

        adjusted_track.append((bx, by_new))
    return adjusted_track

def player_gravity_adjustment(
    ball_track, player_detections, mapped_ball_points, mapped_player_detections,
    bbox_thresh=200, attract_strength=0.75
):
    adjusted_track = []
    for frame_idx, (ball, mapped_ball) in enumerate(zip(ball_track, mapped_ball_points)):
        if (
            ball is None or ball[0] is None or ball[1] is None or
            mapped_ball is None or mapped_ball[0] is None or mapped_ball[1] is None
        ):
            adjusted_track.append(mapped_ball)
            continue

        bx, by = ball
        mapped_bx, mapped_by = mapped_ball

        closest_dist = float('inf')
        closest_player_id = None
        closest_bbox = None

        # Find the closest player bbox (unmapped)
        for track_id, bbox in player_detections[frame_idx].items():
            x1, y1, x2, y2 = bbox
            # Clamp ball to bbox edges to get closest point
            closest_x = min(max(bx, x1), x2)
            closest_y = min(max(by, y1), y2)
            dist = np.sqrt((bx - closest_x) ** 2 + (by - closest_y) ** 2)
            if dist < closest_dist:
                closest_dist = dist
                closest_player_id = track_id
                closest_bbox = bbox

        if closest_dist < bbox_thresh and closest_player_id in mapped_player_detections[frame_idx]:
            # Proximity weight: 1 when on bbox, 0 when at threshold
            weight = 1 - (closest_dist / bbox_thresh)
            # Target: mapped player feet (or center)
            mapped_px, mapped_py = mapped_player_detections[frame_idx][closest_player_id]
            # Move mapped ball toward mapped player
            new_bx = mapped_bx * (1 - attract_strength * weight) + mapped_px * (attract_strength * weight)
            new_by = mapped_by * (1 - attract_strength * weight) + mapped_py * (attract_strength * weight)
            adjusted_track.append((new_bx, new_by))
        else:
            adjusted_track.append(mapped_ball)

    return adjusted_track

'''
Bounce sensitivity constants (for reference, default medium):

Level   ||  min_segment_length      angle_threshold
===================================================
low     ||  8                       np.pi/3
medium  ||  5                       np.pi/4
high    ||  3                       np.pi/6
max     ||  2                       np.pi/8
'''