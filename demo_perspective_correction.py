#!/usr/bin/env python3
"""
Example usage of the perspective-corrected ball tracking system.

This script demonstrates how to use the new --correct_perspective and --show_comparison
flags to fix the 3D to 2D perspective distortion in tennis ball tracking.
"""

import subprocess
import sys
import os

def run_tennis_analysis(input_video="media/tennis.mp4", mode="comparison", sensitivity="medium", 
                       smooth_tracking=True):
    """
    Run tennis analysis with different perspective correction modes.
    
    Args:
        input_video: Path to input tennis video
        mode: One of "original", "corrected", "comparison"
        sensitivity: Bounce detection sensitivity ("low", "medium", "high", "max")
        smooth_tracking: Whether to apply improved tracking with outlier removal
    """
    
    base_cmd = [
        sys.executable, "main.py",
        "--input_path", input_video,
        "--ball_model_path", "models/ball_model.pt",
        "--court_model_path", "models/court_model.pt", 
        "--bounce_model_path", "models/ctb_regr_bounce.cbm",
        "--player_tracking_model_path", "models/yolo12n.pt"
    ]
    
    # Add improved tracking options if enabled
    if smooth_tracking:
        base_cmd.extend([
            "--outlier_removal", "advanced",
            "--smoothing",
            "--interpolate_gaps",
            "--debug_tracking"
        ])
    
    if mode == "original":
        # Run with original perspective (default behavior)
        output_path = "media/processed_original.mp4"
        cmd = base_cmd + ["--output_path", output_path]
        print("🎾 Running analysis with ORIGINAL perspective (3D projection)...")
        print("   This shows the ball trajectory as projected onto the 2D court plane.")
        
    elif mode == "corrected":
        # Run with perspective correction
        output_path = "media/processed_corrected.mp4"
        cmd = base_cmd + [
            "--output_path", output_path,
            "--correct_perspective",
            "--bounce_sensitivity", sensitivity
        ]
        print("🎾 Running analysis with CORRECTED perspective...")
        print("   This shows straight-line trajectories between bounces (true top-down view).")
        print(f"   Using bounce detection sensitivity: {sensitivity}")
        
    elif mode == "comparison":
        # Run with both trajectories shown
        output_path = "media/processed_comparison.mp4"
        cmd = base_cmd + [
            "--output_path", output_path,
            "--correct_perspective",
            "--show_comparison",
            "--bounce_sensitivity", sensitivity,
            "--debug_bounces"
        ]
        print("🎾 Running analysis with COMPARISON view...")
        print("   Red: Original trajectory (with 3D distortion)")
        print("   Green: Corrected trajectory (straight lines between bounces)")
        print("   Yellow: Detected bounce points")
        print(f"   Using bounce detection sensitivity: {sensitivity}")
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    print(f"   Output will be saved to: {output_path}")
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Analysis complete! Check {output_path}")
        
        # Try to open the video if on macOS
        if sys.platform == "darwin":
            subprocess.run(["open", output_path], check=False)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running analysis: {e}")
        return False
    except FileNotFoundError:
        print("❌ Error: Make sure all model files are present in the models/ directory")
        return False
    
    return True

def main():
    print("🎾 Tennis Ball Perspective Correction Demo")
    print("=" * 50)
    print()
    print("This demo shows how to correct 3D perspective distortion in tennis ball tracking.")
    print("When viewing a tennis court from above, the ball should travel in straight lines")
    print("between bounces, not curved paths that result from 3D-to-2D projection.")
    print()
    
    # Check if required files exist
    required_files = [
        "main.py",
        "media/tennis.mp4",
        "models/ball_model.pt",
        "models/court_model.pt",
        "models/ctb_regr_bounce.cbm",
        "models/yolo12n.pt"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease ensure all model files and input video are present.")
        return
    
    print("Available modes:")
    print("1. original - Shows original trajectory with 3D perspective distortion")
    print("2. corrected - Shows corrected straight-line trajectory")
    print("3. comparison - Shows both trajectories side by side")
    print()
    
    mode = input("Enter mode (original/corrected/comparison) [comparison]: ").strip().lower()
    if not mode:
        mode = "comparison"
    
    if mode not in ["original", "corrected", "comparison"]:
        print(f"❌ Invalid mode: {mode}")
        return
    
    # Ask for bounce detection sensitivity if using correction
    sensitivity = "medium"
    smooth_tracking = True
    
    if mode in ["corrected", "comparison"]:
        print()
        print("Bounce detection sensitivity:")
        print("- low: Conservative, fewer bounces (good for noisy tracking)")
        print("- medium: Balanced (recommended)")
        print("- high: Sensitive, more bounces detected")
        print("- max: Very sensitive, may include false positives")
        print()
        
        sensitivity = input("Enter sensitivity (low/medium/high/max) [medium]: ").strip().lower()
        if not sensitivity:
            sensitivity = "medium"
        
        if sensitivity not in ["low", "medium", "high", "max"]:
            print(f"❌ Invalid sensitivity: {sensitivity}")
            return
    
    # Ask about tracking improvements
    print()
    print("Ball tracking improvements:")
    print("- Removes outliers that cause sudden position jumps")
    print("- Applies smoothing to reduce jitter") 
    print("- Interpolates small gaps in tracking")
    print()
    
    smooth_input = input("Apply improved tracking? (y/n) [y]: ").strip().lower()
    smooth_tracking = smooth_input != 'n'
    
    print()
    success = run_tennis_analysis(mode=mode, sensitivity=sensitivity, smooth_tracking=smooth_tracking)
    
    if success:
        print()
        print("🎾 Perspective Correction Explanation:")
        print("=" * 40)
        print("The original ball tracking projects the 3D ball position onto the 2D court.")
        print("This creates curved trajectories because we're seeing the 'shadow' of the")
        print("ball's parabolic flight path on the ground.")
        print()
        print("The corrected version detects bounce points and creates straight lines")
        print("between them, showing how the ball would look from a true top-down view.")
        print("This is more accurate for analyzing ball placement and court coverage.")

if __name__ == "__main__":
    main()
