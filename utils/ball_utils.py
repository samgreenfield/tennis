import os, cv2, pickle, torch, numpy as np
from tqdm import tqdm
from utils import distance

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