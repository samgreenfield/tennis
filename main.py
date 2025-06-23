import argparse, torch, os, subprocess
from utils import *

def main():
    # Parse input
    parser = argparse.ArgumentParser()
    parser.add_argument('--ball_model_path', type=str, help='path to ball tracking model')
    parser.add_argument('--court_model_path', type=str, help='path to court tracking model')
    parser.add_argument('--bounce_model_path', type=str, help='path to bounce detecting model')
    parser.add_argument('--player_tracking_model_path', type=str, help='path to player tracking model')
    parser.add_argument('--input_path', type=str, help='path to input video')
    parser.add_argument('--output_path', type=str, help='path to output video')
    parser.add_argument('--device', type=str, help="device to use with models (cpu/cuda/mps)")
    parser.add_argument('--correct_perspective', action='store_true', help='use perspective-corrected trajectory for top-down view')
    parser.add_argument('--show_comparison', action='store_true', help='show both original and corrected trajectories')
    parser.add_argument('--bounce_sensitivity', type=str, choices=['low', 'medium', 'high', 'max'], 
                       default='medium', help='bounce detection sensitivity (low/medium/high/max)')
    parser.add_argument('--debug_bounces', action='store_true', help='print debug information about bounce detection')
    parser.add_argument('--outlier_removal', type=str, choices=['basic', 'advanced'], 
                       default='advanced', help='outlier removal method (basic/advanced)')
    parser.add_argument('--max_velocity', type=float, default=150, 
                       help='maximum allowed ball velocity (pixels per frame)')
    parser.add_argument('--smoothing', action='store_true', 
                       help='apply temporal smoothing to reduce tracking jitter')
    parser.add_argument('--interpolate_gaps', action='store_true', 
                       help='interpolate small gaps in ball tracking')
    parser.add_argument('--debug_tracking', action='store_true', 
                       help='show debug information about ball tracking improvements')
    args = parser.parse_args()

    if not args.input_path:
        args.input_path = "media/tennis.mp4"
    if not args.ball_model_path:
        args.ball_model_path = 'models/ball_model.pt'
    if not args.court_model_path:
        args.court_model_path = 'models/court_model.pt'
    if not args.bounce_model_path:
        args.bounce_model_path = 'models/ctb_regr_bounce.cbm'
    if not args.player_tracking_model_path:
        args.player_tracking_model_path = "models/yolo12n.pt"
    if not args.output_path:
        args.output_path = os.path.join(os.path.dirname(args.input_path), "processed.mp4")

    if not (os.path.isfile(args.ball_model_path) and os.path.isfile(args.court_model_path) and os.path.isfile(args.input_path) and os.path.isfile(args.bounce_model_path)):
        print("Bad path")
        return

    # Define device to load models into
    device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Set up ball-tracking model
    ball_model = BallModel()
    ball_model.load_state_dict(torch.load(args.ball_model_path, map_location=device))
    ball_model = ball_model.to(device)
    ball_model.eval()

    # Set up court-tracking model
    court_model = CourtModel(out_channels=15)
    court_model = court_model.to(device)
    court_model.load_state_dict(torch.load(args.court_model_path, map_location=device))
    court_model.eval()

    # Load the video
    frames, fps = read_video(args.input_path)

    # Compute ball location
    ball_track, dists = infer_ball(frames, ball_model, device)
    
    # Apply improved outlier removal and smoothing
    if args.debug_tracking:
        print("🎾 Original ball tracking:")
        valid_orig = sum(1 for pt in ball_track if pt[0] is not None)
        print(f"   Valid points: {valid_orig}/{len(ball_track)} ({valid_orig/len(ball_track)*100:.1f}%)")
    
    if args.outlier_removal == 'advanced':
        ball_track = remove_outliers_advanced(
            ball_track, dists, 
            max_velocity=args.max_velocity,
            debug=args.debug_tracking
        )
    else:
        ball_track = remove_outliers(ball_track, dists)
    
    # Apply optional smoothing
    if args.smoothing:
        if args.debug_tracking:
            print("🔧 Applying temporal smoothing...")
        ball_track = apply_smoothing(ball_track, window_size=5)
    
    # Apply optional gap interpolation
    if args.interpolate_gaps:
        if args.debug_tracking:
            print("🔧 Interpolating small gaps...")
        ball_track = interpolate_missing_points(ball_track, max_gap=8)
    
    if args.debug_tracking:
        valid_final = sum(1 for pt in ball_track if pt[0] is not None)
        print(f"🎾 Final ball tracking:")
        print(f"   Valid points: {valid_final}/{len(ball_track)} ({valid_final/len(ball_track)*100:.1f}%)")
        print(f"   Improvement: +{valid_final - valid_orig} points")

    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360

    # Compute court points
    frame_points = infer_court(frames, OUTPUT_WIDTH, OUTPUT_HEIGHT, court_model, device)
    interpolated_points_per_frame = interpolate_court_points_per_frame(frames, frame_points)

    # Compute transform to 2D
    homography = Homography()
    homographies = frame_homographies(frames, interpolated_points_per_frame, homography)

    # Track players
    player_tracker = PlayerTracker(args.player_tracking_model_path)
    player_detections = player_tracker.detect_frames(frames)
    player_detections = player_tracker.choose_and_filter_players(frame_points, player_detections, homography, homographies[0])

    # Map ball location to 2D with transform
    mapped_ball_points = map_ball_points(ball_track, homography, homographies)

    # Decide which trajectory to use for visualization
    if args.correct_perspective:
        # Create straight-line trajectory to correct for 3D perspective distortion
        # This shows how the ball would look from a true top-down view
        if args.debug_bounces:
            print(f"🔍 Using bounce detection sensitivity: {args.bounce_sensitivity}")
        
        bounce_frames = detect_bounces_with_sensitivity(mapped_ball_points, args.bounce_sensitivity)
        
        if args.debug_bounces:
            print(f"🎾 Detected {len(bounce_frames)} bounces at frames: {bounce_frames}")
        
        corrected_ball_points = create_straight_trajectory(mapped_ball_points, bounce_frames)
        
        if args.show_comparison:
            # Show both trajectories for comparison
            virtual_court_frames = draw_virtual_court_comparison(
                homography.court_image, mapped_ball_points, corrected_ball_points, 
                show_bounces=True, bounce_frames=bounce_frames
            )
        else:
            # Show only corrected trajectory
            virtual_court_frames = draw_virtual_court(homography.court_image, corrected_ball_points)
    else:
        # Use original trajectory (with perspective distortion)
        virtual_court_frames = draw_virtual_court(homography.court_image, mapped_ball_points)

    live_court_frames = build_live_court_view(frames, interpolated_points_per_frame, homography)

    # Draw ball and court
    frames = draw_ball(frames, ball_track, trace = 1)
    frames = draw_court(frames, interpolated_points_per_frame)
    frames = player_tracker.draw_bboxes(frames, player_detections)

    # # Draw a line connecting tracked players to the points they are tracked from
    # tether_players_to_points(player_tracker, frame_points, player_detections, homography, homographies, frames, interpolated_points_per_frame)

    combined_video = combine_frames(live_court_frames, frames, virtual_court_frames)

    # Draw ball location on frame and save to output path
    save_video(args.output_path, fps, combined_video)

    subprocess.call(['open', args.output_path])

if __name__ == '__main__':
    main()




# # # Bounce detection and trajectory analysis:

    # # Prepare for bounce detection
    # x_ball = [ pt[0] if pt is not None else None for pt in mapped_ball_points]
    # y_ball = [ pt[1] if pt is not None else None for pt in mapped_ball_points]
    
    # # Detect bounces using multiple methods
    # bounce_detector = BounceDetector(path_model=args.bounce_model_path, threshold = .1)
    # bounce_frames = bounce_detector.predict(x_ball, y_ball, smooth=True)
    # change_frames = detect_trajectory_changes(mapped_ball_points, velocity_threshold=15)
    # advanced_bounces = detect_bounces_advanced(mapped_ball_points)

    # # Create corrected trajectory using detected bounces
    # straightened_points = create_straight_trajectory(mapped_ball_points, advanced_bounces)