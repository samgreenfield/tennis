import argparse, torch, os, subprocess
from utils import *

def main():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ball_model_path', type=str, help='path to ball tracking model')
    parser.add_argument('--court_model_path', type=str, help='path to court tracking model')
    parser.add_argument('--bounce_model_path', type=str, help='path to bounce detecting model')
    parser.add_argument('--player_tracking_model_path', type=str, help='path to player tracking model')
    parser.add_argument('--input_path', type=str, help='path to input video')
    parser.add_argument('--output_path', type=str, help='path to output video')
    parser.add_argument('--device', type=str, help="device to use with models (cpu/cuda/mps)")
    args = parser.parse_args()

    if not args.input_path:
        args.input_path = "media/tennis.mp4"
    if not args.ball_model_path:
        args.ball_model_path = 'models/ball_model.pt'
    if not args.court_model_path:
        args.court_model_path = 'models/court_model.pt'
    if not args.player_tracking_model_path:
        args.player_tracking_model_path = "models/yolo12m.pt"
    if not args.output_path:
        args.output_path = os.path.join(os.path.dirname(args.input_path), "processed.mp4")

    if not (os.path.isfile(args.ball_model_path) and os.path.isfile(args.court_model_path) and os.path.isfile(args.input_path)):
        print("Bad path")
        return

    # Define device to load models into
    device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Load the video
    frames, fps = read_video(args.input_path)
    scale = frames[0].shape[1] / 640.0

    # Define homography objext
    homography = Homography()

    # Compute court points
    frame_points = infer_court(frames, 640, 360, args.court_model_path, device)
    filled_frame_points = fill_missing_points_per_frame(frame_points, homography)
    interpolated_points_per_frame = interpolate_court_points_per_frame(frames, filled_frame_points)

    # Compute transform to 2D
    homographies = frame_homographies(frames, interpolated_points_per_frame, homography)

    # Track players
    player_tracker = PlayerTracker(args.player_tracking_model_path)
    player_detections = player_tracker.detect_frames(frames)
    player_detections = player_tracker.choose_and_filter_players(frame_points, player_detections, homography, homographies[0])

    # Compute ball location
    ball_track, dists = infer_ball(frames, device, args.ball_model_path)
    ball_track = remove_outliers(ball_track, dists)
    ball_track = apply_smoothing(ball_track)
    ball_track = interpolate_missing_points(ball_track)

    # Map ball and player locations to 2D with transform
    mapped_ball_points = map_ball_points(ball_track, homography, homographies)
    mapped_player_detections = map_player_detections(player_detections, homography, homographies)

    # Correct for curvature
    bounce_frames = detect_bounces(mapped_ball_points, 2, np.pi/8)
    mapped_ball_points = player_gravity_adjustment(ball_track, player_detections, mapped_ball_points, mapped_player_detections)
    mapped_ball_points = adjust_ball_height(mapped_ball_points, mapped_player_detections, homography.ref_court_pts, fade_dist=10, player_x_thresh=100)
    mapped_ball_points = create_straight_trajectory(mapped_ball_points, bounce_frames)
    
    print("Drawing...")
    # Draw ball on virtual court
    virtual_court_frames = draw_virtual_court(homography.court_image, mapped_ball_points, mapped_player_detections)
    live_court_frames = build_live_court_view(frames, interpolated_points_per_frame, homography)

    # Draw ball and court on original frames
    frames = draw_ball(frames, ball_track, 1, scale)
    frames = draw_court(frames, interpolated_points_per_frame, scale)
    frames = player_tracker.draw_bboxes(frames, player_detections, scale)

    # # Draw a line connecting tracked players to the points they are tracked from
    # tether_players_to_points(player_tracker, frame_points, player_detections, homography, homographies, frames, interpolated_points_per_frame)

    # Combine frames of live court, original frames, and virtual court
    print("Combining frames...")
    combined_video = combine_frames(live_court_frames, frames, virtual_court_frames)

    # Save video to file
    print("Saving video...")
    save_video(args.output_path, fps, combined_video)

    subprocess.call(['open', "-R", args.output_path])
    subprocess.call(['open', args.output_path])

if __name__ == '__main__':
    main()