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
    outliers = set(np.where(np.array(dists) > max_dist)[0])
    static_counter = 0
    last_valid_pt = None
    
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













# def build_straight_trajectory(mapped_ball_points, bounce_frames):
#     N = len(mapped_ball_points)
    
#     # Sort bounce frames
#     bounces_sorted = sorted(list(bounce_frames))
    
#     # Edge cases: always start at first and last frame
#     if 0 not in bounces_sorted:
#         bounces_sorted = [0] + bounces_sorted
#     if (N-1) not in bounces_sorted:
#         bounces_sorted = bounces_sorted + [N-1]
    
#     straightened_points = [None] * N
    
#     # For each segment between bounces:
#     for i in range(len(bounces_sorted)-1):
#         start_idx = bounces_sorted[i]
#         end_idx = bounces_sorted[i+1]
        
#         # Get (x, y) at start and end bounce
#         start_pt = mapped_ball_points[start_idx]
#         end_pt = mapped_ball_points[end_idx]
        
#         if start_pt is None or end_pt is None:
#             # Skip segment if missing data
#             continue
        
#         x0, y0 = start_pt
#         x1, y1 = end_pt
        
#         for f in range(start_idx, end_idx+1):
#             alpha = (f - start_idx) / (end_idx - start_idx + 1e-8)
#             x_interp = (1 - alpha) * x0 + alpha * x1
#             y_interp = (1 - alpha) * y0 + alpha * y1
#             straightened_points[f] = (x_interp, y_interp)
    
#     return straightened_points

# def detect_trajectory_changes(mapped_ball_points, velocity_threshold=10.0):
#     vx_list = []
#     vy_list = []
#     v_norm_list = []
    
#     for i in range(1, len(mapped_ball_points)):
#         pt_prev = mapped_ball_points[i-1]
#         pt_curr = mapped_ball_points[i]
        
#         if pt_prev is None or pt_curr is None:
#             vx_list.append(0)
#             vy_list.append(0)
#             v_norm_list.append(0)
#             continue
        
#         vx = pt_curr[0] - pt_prev[0]
#         vy = pt_curr[1] - pt_prev[1]
#         v_norm = np.sqrt(vx**2 + vy**2)
        
#         vx_list.append(vx)
#         vy_list.append(vy)
#         v_norm_list.append(v_norm)
    
#     # Now compute delta_v
#     delta_v = []
#     change_frames = []
    
#     for i in range(1, len(v_norm_list)):
#         dv = abs(v_norm_list[i] - v_norm_list[i-1])
#         delta_v.append(dv)
        
#         if dv > velocity_threshold:
#             change_frames.append(i+1)  # because of lag
    
#     return set(change_frames)