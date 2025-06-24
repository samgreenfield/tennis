import cv2, numpy as np
from copy import copy

def draw_virtual_court(court_img, mapped_ball_points, player_detections):
    court_frames = []
    for frame_idx, ball_mapped in enumerate(mapped_ball_points):
        curr_frame = copy(court_img)
        if ball_mapped is not None:
            x = int(ball_mapped[0])
            y = int(ball_mapped[1])
            curr_frame = cv2.circle(curr_frame, (x, y), radius=10, color=(127, 0, 255), thickness=-1)
        if player_detections[frame_idx] is not None:
            for detection_point in player_detections[frame_idx].values():
                curr_frame = cv2.circle(curr_frame, (detection_point[0], detection_point[1]), radius=15, color=(73, 247, 245), thickness=-1)
        court_frames.append(curr_frame)
    return court_frames

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

