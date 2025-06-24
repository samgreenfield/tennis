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

def combine_frames(live_court_frames, frames, virtual_frames):
    combined_frames = []

    for frame_idx in range(len(frames)):
        frame_disp = frames[frame_idx].copy()
        court_disp = virtual_frames[frame_idx]
        live_warp_disp = live_court_frames[frame_idx]

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

def save_video(output_path, fps, frames):
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()