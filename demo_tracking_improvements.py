#!/usr/bin/env python3
"""
Ball Tracking Improvement Demo

This script demonstrates the improved ball tracking with outlier removal and smoothing.
"""

import sys
import os
import pickle
import subprocess
import numpy as np

def analyze_tracking_quality(ball_track, title="Ball Tracking"):
    """Analyze and report tracking quality metrics"""
    print(f"\n📊 {title} Analysis:")
    print("-" * 40)
    
    total_frames = len(ball_track)
    valid_frames = sum(1 for pt in ball_track if pt[0] is not None)
    coverage = valid_frames / total_frames * 100
    
    print(f"Total frames: {total_frames}")
    print(f"Valid detections: {valid_frames}")
    print(f"Coverage: {coverage:.1f}%")
    
    # Calculate tracking smoothness (distance between consecutive points)
    distances = []
    velocities = []
    
    for i in range(1, len(ball_track)):
        prev_pt = ball_track[i-1]
        curr_pt = ball_track[i]
        
        if prev_pt[0] is not None and curr_pt[0] is not None:
            dist = np.sqrt((curr_pt[0] - prev_pt[0])**2 + (curr_pt[1] - prev_pt[1])**2)
            distances.append(dist)
            velocities.append(dist)  # velocity in pixels per frame
    
    if distances:
        print(f"Average velocity: {np.mean(velocities):.1f} pixels/frame")
        print(f"Max velocity: {np.max(velocities):.1f} pixels/frame")
        print(f"Velocity std dev: {np.std(velocities):.1f} (lower = smoother)")
        
        # Count sudden jumps (outliers)
        velocity_threshold = np.mean(velocities) + 2 * np.std(velocities)
        sudden_jumps = sum(1 for v in velocities if v > velocity_threshold)
        print(f"Sudden jumps: {sudden_jumps} ({sudden_jumps/len(velocities)*100:.1f}%)")

def compare_tracking_methods():
    """Compare basic vs advanced tracking methods"""
    print("🎾 Ball Tracking Improvement Comparison")
    print("=" * 50)
    
    input_video = "media/tennis.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return
    
    print("This demo will run the same video with different tracking settings:")
    print("1. Basic tracking (original method)")
    print("2. Advanced tracking (with outlier removal and smoothing)")
    print()
    
    # Run basic tracking
    print("🔄 Running basic tracking...")
    basic_output = "media/tracking_basic.mp4"
    cmd_basic = [
        sys.executable, "main.py",
        "--input_path", input_video,
        "--output_path", basic_output,
        "--outlier_removal", "basic"
    ]
    
    try:
        subprocess.run(cmd_basic, check=True, capture_output=True)
        print(f"✅ Basic tracking complete: {basic_output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Basic tracking failed: {e}")
        return
    
    # Run advanced tracking
    print("🔄 Running advanced tracking...")
    advanced_output = "media/tracking_advanced.mp4"
    cmd_advanced = [
        sys.executable, "main.py",
        "--input_path", input_video,
        "--output_path", advanced_output,
        "--outlier_removal", "advanced",
        "--smoothing",
        "--interpolate_gaps",
        "--debug_tracking"
    ]
    
    try:
        result = subprocess.run(cmd_advanced, check=True, capture_output=True, text=True)
        print(f"✅ Advanced tracking complete: {advanced_output}")
        
        # Show debug output
        if result.stdout:
            print("\n📊 Advanced Tracking Details:")
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['Valid points:', 'Improvement:', 'Step', 'summary']):
                    print(f"   {line}")
                    
    except subprocess.CalledProcessError as e:
        print(f"❌ Advanced tracking failed: {e}")
        return
    
    print("\n🎯 Comparison Summary:")
    print("=" * 30)
    print("Basic tracking:")
    print("- Uses simple distance-based outlier removal")
    print("- May show sudden ball position jumps")
    print("- Faster processing")
    print()
    print("Advanced tracking:")
    print("- Uses velocity and acceleration constraints")
    print("- Applies temporal smoothing")
    print("- Interpolates small gaps")
    print("- Smoother, more realistic ball movement")
    
    # Try to open videos if on macOS
    if sys.platform == "darwin":
        print(f"\n🎬 Opening comparison videos...")
        subprocess.run(["open", basic_output], check=False)
        subprocess.run(["open", advanced_output], check=False)

def test_tracking_parameters():
    """Interactive parameter testing for tracking improvements"""
    print("\n🔧 Interactive Tracking Parameter Testing")
    print("=" * 45)
    
    # Load existing ball tracking data if available
    stub_path = 'stubs/ball_stub.pkl'
    if not os.path.exists(stub_path):
        print("❌ No ball tracking data found. Run main analysis first.")
        return
    
    with open(stub_path, 'rb') as stub:
        ball_track, dists = pickle.load(stub)
    
    print(f"📊 Loaded tracking data: {len(ball_track)} frames")
    
    # Import the functions we need
    sys.path.append('.')
    from utils import remove_outliers_advanced, apply_smoothing, interpolate_missing_points
    
    original_track = ball_track.copy()
    analyze_tracking_quality(original_track, "Original Tracking")
    
    while True:
        print("\n🎛️ Parameter Testing Options:")
        print("1. Test max velocity threshold")
        print("2. Test smoothing window size")
        print("3. Test interpolation gap size")
        print("4. Compare all methods")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            try:
                max_vel = float(input("Enter max velocity (50-300): "))
                test_track = remove_outliers_advanced(original_track, dists, max_velocity=max_vel, debug=True)
                analyze_tracking_quality(test_track, f"Max Velocity = {max_vel}")
            except ValueError:
                print("❌ Invalid input")
        
        elif choice == "2":
            try:
                window = int(input("Enter smoothing window size (3-15): "))
                test_track = apply_smoothing(original_track, window_size=window)
                analyze_tracking_quality(test_track, f"Smoothing Window = {window}")
            except ValueError:
                print("❌ Invalid input")
        
        elif choice == "3":
            try:
                gap_size = int(input("Enter max interpolation gap (5-20): "))
                test_track = interpolate_missing_points(original_track, max_gap=gap_size)
                analyze_tracking_quality(test_track, f"Max Gap = {gap_size}")
            except ValueError:
                print("❌ Invalid input")
        
        elif choice == "4":
            print("\n🔄 Testing all methods...")
            
            # Test advanced outlier removal
            clean_track = remove_outliers_advanced(original_track, dists, debug=True)
            analyze_tracking_quality(clean_track, "After Outlier Removal")
            
            # Test smoothing
            smooth_track = apply_smoothing(clean_track, window_size=5)
            analyze_tracking_quality(smooth_track, "After Smoothing")
            
            # Test interpolation
            final_track = interpolate_missing_points(smooth_track, max_gap=8)
            analyze_tracking_quality(final_track, "Final (All Methods)")
            
        elif choice == "5":
            break
            
        else:
            print("❌ Invalid choice")

def main():
    print("🎾 Ball Tracking Improvement Demo")
    print("=" * 40)
    print()
    print("This tool helps you understand and test the improved ball tracking system.")
    print("The new system removes outliers that cause sudden position jumps and")
    print("applies smoothing to create more realistic ball movement.")
    print()
    
    while True:
        print("Available options:")
        print("1. Compare basic vs advanced tracking")
        print("2. Test tracking parameters interactively")
        print("3. Show usage examples")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            compare_tracking_methods()
        
        elif choice == "2":
            test_tracking_parameters()
        
        elif choice == "3":
            print("\n📖 Usage Examples:")
            print("=" * 20)
            print("# Basic improved tracking:")
            print("python main.py --input_path media/tennis.mp4 --outlier_removal advanced")
            print()
            print("# Full improvements:")
            print("python main.py --input_path media/tennis.mp4 \\")
            print("    --outlier_removal advanced \\")
            print("    --smoothing \\")
            print("    --interpolate_gaps \\")
            print("    --debug_tracking")
            print()
            print("# Custom velocity limit:")
            print("python main.py --input_path media/tennis.mp4 \\")
            print("    --outlier_removal advanced \\")
            print("    --max_velocity 100")
            print()
            print("# With perspective correction:")
            print("python main.py --input_path media/tennis.mp4 \\")
            print("    --correct_perspective \\")
            print("    --outlier_removal advanced \\")
            print("    --smoothing \\")
            print("    --show_comparison")
        
        elif choice == "4":
            break
            
        else:
            print("❌ Invalid choice")
    
    print("\n🎾 Tips for Better Tracking:")
    print("=" * 30)
    print("• Use 'advanced' outlier removal for smoother tracking")
    print("• Enable smoothing to reduce jitter")
    print("• Enable gap interpolation to fill small missing segments")
    print("• Lower max_velocity for very controlled/slow footage")
    print("• Use debug_tracking to see what improvements are applied")

if __name__ == "__main__":
    main()
