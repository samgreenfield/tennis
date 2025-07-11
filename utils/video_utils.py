import cv2, numpy as np

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

def resize_to_height(img, target_h):
    h_img, w_img = img.shape[:2]
    if h_img != target_h:
        scale = target_h / h_img
        new_w = int(w_img * scale)
        img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    return img

def combine_frames(live_court_frames, frames, virtual_frames):
    combined_frames = []
    n = len(frames)
    h_target = max(frames[0].shape[0], virtual_frames[0].shape[0], live_court_frames[0].shape[0])
    
    for frame_idx in range(n):
        frame_disp = frames[frame_idx]
        court_disp = virtual_frames[frame_idx]
        live_warp_disp = live_court_frames[frame_idx]

        frame_disp = resize_to_height(frame_disp, h_target)
        court_disp = resize_to_height(court_disp, h_target)
        live_warp_disp = resize_to_height(live_warp_disp, h_target)

        combined = np.hstack((live_warp_disp, frame_disp, court_disp))
        combined_frames.append(combined)

    return combined_frames

def save_video(output_path, fps, frames):
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()