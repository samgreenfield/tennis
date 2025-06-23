import cv2, torch, pickle, os, numpy as np, torch.nn.functional as F
from tqdm import tqdm
from utils import interpolate_points

def postprocess_court(heatmap, scale=2, low_thresh=155, min_radius=10, max_radius=30):
    x_pred, y_pred = None, None
    ret, heatmap = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=min_radius,
                               maxRadius=max_radius)
    if circles is not None:
        x_pred = circles[0][0][0] * scale
        y_pred = circles[0][0][1] * scale
    return x_pred, y_pred

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
                image = cv2.circle(image, (int(p[0]), int(p[1])),
                                  radius=0, color=(0, 0, 255), thickness=10)
                image = cv2.putText(image, str(pt_idx), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for idx, p1 in enumerate(corner_points[:-1]):
            for p2 in corner_points[idx + 1:]:
                image = cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (222, 74, 69), 5)

        frames_upd.append(image)

    return frames_upd

def frame_homographies(frames, corner_points, homography_obj):
    homographies = []
    for frame_idx, frame in enumerate(frames):
        input_points = np.array(corner_points[frame_idx][:4], dtype=np.float32)
        H = homography_obj.compute_homography(input_points)
        homographies.append(H)
    return homographies

def fill_missing_points(frame_pts):
    ref_court_pts = np.array([
        [87, 35],
        [406, 35],
        [87, 705],
        [406, 705],
        [121, 35],
        [121, 705],
        [372, 35],
        [372, 705],
        [121, 201],
        [372, 201],
        [121, 539],
        [372, 539],
        [247, 202],
        [247, 539]], dtype=np.float32)

    src_pts = []
    dst_pts = []
    for i, pt in enumerate(frame_pts):
        if not None in pt:
            src_pt = ref_court_pts[i]
            src_pts.append((src_pt[0], src_pt[1]))
            dst_pts.append(pt)

    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)

    if len(src_pts) < 4:
        print("Not enough points to infer homography")
        return frame_pts
        
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    ref_pts_homo = cv2.perspectiveTransform(ref_court_pts.reshape(-1,1,2), H).reshape(-1,2)

    new_frame_pts = []
    for i in range(14):
        if not None in frame_pts[i]:
            new_frame_pts.append(frame_pts[i])
        else:
            x, y = ref_pts_homo[i]
            new_frame_pts.append( (float(x), float(y)) )

    return new_frame_pts

def fill_missing_points_per_frame(frame_points):
    filled_points_per_frame = {}
    for frame_idx, pts in frame_points.items():
        filled_pts = fill_missing_points(pts)
        filled_points_per_frame[frame_idx] = filled_pts
    return filled_points_per_frame