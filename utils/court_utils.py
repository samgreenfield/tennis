import cv2, torch, pickle, os, numpy as np, torch.nn.functional as F
from tqdm import tqdm
from utils import interpolate_points

def infer_court(frames, width, height, model, device, step = 5, stub_path = 'stubs/court_stub.pkl'):
    if os.path.isfile(stub_path):
        with open(stub_path, 'rb') as stub:
            return pickle.load(stub)
    
    inferred_points = {}
    
    for idx in tqdm(range(0, len(frames), step), desc="Inferring court points", unit="frame"):
        image = frames[idx]
        scale = image.shape[1] / width
        img = cv2.resize(image, (width, height))
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        out = model(inp.float().to(device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess_court(heatmap, low_thresh=170, max_radius=25, scale=scale)
            if x_pred is not None and y_pred is not None:
                max_val = float(np.max(pred[kps_num]))
                mean_val = float(np.mean(pred[kps_num]))
                peakiness = max_val - mean_val
                xh = int(x_pred / scale)
                yh = int(y_pred / scale)
                window = pred[kps_num][max(0, yh-7):yh+8, max(0, xh-7):xh+8]
                local_mean = float(np.mean(window)) if window.size > 0 else 0.0
                confidence = 0.5 * peakiness + 0.5 * local_mean
            else:
                confidence = 0.0
            points.append((x_pred, y_pred, confidence))

        inferred_points[idx] = points

    with open(stub_path, 'wb') as stub:
        pickle.dump(inferred_points, stub)
    
    return inferred_points

def postprocess_court(heatmap, scale=2, low_thresh=155, min_radius=10, max_radius=30):
    x_pred, y_pred = None, None
    ret, heatmap = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=min_radius,
                               maxRadius=max_radius)
    if circles is not None:
        x_pred = circles[0][0][0] * scale
        y_pred = circles[0][0][1] * scale
    return x_pred, y_pred

def fill_missing_points_per_frame(frame_points, homography):
    filled_points_per_frame = {}
    for frame_idx, pts in frame_points.items():
        filled_pts = fill_missing_points(pts, homography=homography)
        filled_points_per_frame[frame_idx] = filled_pts
    return filled_points_per_frame

def fill_missing_points(frame_pts, confidence_thresh=0.52, homography=None):
    ref_court_pts = homography.ref_court_pts
    src_pts = []
    dst_pts = []
    # confidence_tot = 0

    for i, pt in enumerate(frame_pts):
        if pt is not None and len(pt) == 3 and pt[2] >= confidence_thresh:
            src_pt = ref_court_pts[i]
            src_pts.append((src_pt[0], src_pt[1]))
            dst_pts.append(pt[:2])
            # confidence_tot += pt[2]

    # print(confidence_tot/len(frame_pts))

    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)

    if len(src_pts) < 4:
        # print("Not enough points to infer homography")
        return frame_pts
        
    H, _ = cv2.findHomography(src_pts, dst_pts)

    ref_pts_homo = cv2.perspectiveTransform(ref_court_pts.reshape(-1,1,2), H).reshape(-1,2)

    new_frame_pts = []
    for i in range(14):
        if not None in frame_pts[i]:
            new_frame_pts.append(frame_pts[i])
        else:
            x, y = ref_pts_homo[i]
            new_frame_pts.append( (float(x), float(y)) )

    return new_frame_pts

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

def frame_homographies(frames, corner_points, homography_obj):
    homographies = []
    for frame_idx in range(len(frames)):
        input_points = np.array([corner_point[:2] for corner_point in corner_points[frame_idx]], dtype=np.float32)
        H = homography_obj.compute_homography(input_points)
        homographies.append(H)
    return homographies

def draw_court(frames, interpolated_points_per_frame, scale):
    frames_upd = []

    for idx, image in enumerate(frames):
        interp_points = interpolated_points_per_frame[idx]

        # Draw translucent blue quadrilateral with correct point order: 0,1,3,2 (clockwise)
        quad_indices = [0, 1, 3, 2]
        quad_pts = [interp_points[i] for i in quad_indices]
        if all(p is not None and not None in p for p in quad_pts):
            pts = np.array([[int(p[0]), int(p[1])] for p in quad_pts], dtype=np.int32)
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], color=(255, 0, 0))  # Blue in BGR
            alpha = 0.25
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        for pt_idx, p in enumerate(interp_points):
            if not None in p:
                x = int(p[0])
                y = int(p[1])

                # if len(corner_points) < 4:
                #     corner_points.append((x, y))

                image = cv2.circle(image, (x, y), radius=0, color=(0, 0, 255), thickness=int(4 * scale))
                image = cv2.putText(image, str(pt_idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25 * scale, (255, 0, 0), int(1.5 * scale))

        # for i, p1 in enumerate(corner_points[:-1]):
        #     for p2 in corner_points[i + 1:]:
        # for (p1, p2) in [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]:
        #     p1 = (int(interp_points[p1][0]), int(interp_points[p1][1]))
        #     p2 = (int(interp_points[p2][0]), int(interp_points[p2][1]))        
        #     image = cv2.line(image, p1, p2, (222, 74, 69), int(3 * scale))

        

        frames_upd.append(image)

    return frames_upd