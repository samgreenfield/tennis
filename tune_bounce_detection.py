#!/usr/bin/env python3
"""
Bounce Detection Tuning Tool

This script helps you test and tune bounce detection parameters to catch more bounces.
"""

import sys
import os
import pickle
import numpy as np

# Add the utils to the path
sys.path.append('.')
from utils import *

def load_ball_tracking_data(stub_path='stubs/ball_stub.pkl'):
    """Load ball tracking data from stub file"""
    if not os.path.exists(stub_path):
        print(f"❌ Ball tracking stub not found: {stub_path}")
        print("Run the main analysis first to generate tracking data.")
        return None, None
    
    with open(stub_path, 'rb') as stub:
        ball_track, dists = pickle.load(stub)
    
    return ball_track, dists

def test_bounce_detection():
    """Test bounce detection with different parameters"""
    print("🎾 Bounce Detection Tuning Tool")
    print("=" * 50)
    
    # Load ball tracking data
    ball_track, dists = load_ball_tracking_data()
    if ball_track is None:
        return
    
    print(f"📊 Loaded {len(ball_track)} frames of ball tracking data")
    
    # Count valid points
    valid_count = sum(1 for pt in ball_track if pt is not None)
    print(f"📍 {valid_count} frames with valid ball positions ({valid_count/len(ball_track)*100:.1f}%)")
    
    # We need to map the ball points using homography for bounce detection
    # For testing, we'll simulate this by using the raw ball positions
    # In real use, this would come from map_ball_points()
    
    print("\n🔍 Testing bounce detection with different sensitivity levels:")
    print("-" * 60)
    
    sensitivities = ['low', 'medium', 'high', 'max']
    
    for sensitivity in sensitivities:
        bounces = detect_bounces_with_sensitivity(ball_track, sensitivity)
        print(f"{sensitivity.ljust(8)}: {len(bounces):2d} bounces detected at frames: {bounces}")
    
    # Test custom parameters
    print("\n🔧 Testing with custom parameters:")
    print("-" * 40)
    
    # Very sensitive settings
    custom_bounces = detect_bounces_advanced(
        ball_track, 
        min_segment_length=1,  # Allow bounces very close together
        angle_threshold=np.pi/8,  # Very small angle changes
        use_multiple_methods=True,
        debug=True
    )
    
    print(f"\nCustom settings: {len(custom_bounces)} bounces detected")
    
    # Create analysis visualization if matplotlib is available
    try:
        analyze_bounce_detection(ball_track, "bounce_detection_analysis.png")
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")
    
    # Interactive parameter tuning
    print("\n🎯 Interactive Parameter Tuning")
    print("=" * 40)
    print("Adjust parameters to find optimal bounce detection:")
    
    while True:
        print("\nCurrent options:")
        print("1. Test angle threshold")
        print("2. Test minimum segment length")
        print("3. Test with debug output")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            try:
                angle_deg = float(input("Enter angle threshold in degrees (15-90): "))
                angle_rad = np.radians(angle_deg)
                bounces = detect_bounces_advanced(
                    ball_track, 
                    angle_threshold=angle_rad,
                    use_multiple_methods=True
                )
                print(f"With {angle_deg}° threshold: {len(bounces)} bounces at frames {bounces}")
            except ValueError:
                print("❌ Invalid input")
        
        elif choice == "2":
            try:
                min_len = int(input("Enter minimum segment length (1-10): "))
                bounces = detect_bounces_advanced(
                    ball_track, 
                    min_segment_length=min_len,
                    use_multiple_methods=True
                )
                print(f"With min_segment_length={min_len}: {len(bounces)} bounces at frames {bounces}")
            except ValueError:
                print("❌ Invalid input")
        
        elif choice == "3":
            bounces = detect_bounces_advanced(
                ball_track, 
                min_segment_length=3,
                angle_threshold=np.pi/6,
                use_multiple_methods=True,
                debug=True
            )
            print(f"Debug run: {len(bounces)} bounces detected")
        
        elif choice == "4":
            break
        
        else:
            print("❌ Invalid choice")
    
    print("\n🎾 Recommendations:")
    print("=" * 20)
    print("• If you're missing bounces: try 'high' or 'max' sensitivity")
    print("• If you're getting false positives: try 'low' sensitivity") 
    print("• For most cases: 'medium' sensitivity works well")
    print("• Use --debug_bounces to see detailed detection info")

def main():
    if not os.path.exists('utils'):
        print("❌ This script must be run from the tennis project root directory")
        return
    
    test_bounce_detection()

if __name__ == "__main__":
    main()
