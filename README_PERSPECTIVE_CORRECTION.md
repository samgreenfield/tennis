# Tennis Ball Perspective Correction

This project now includes perspective correction to fix the 3D-to-2D distortion in tennis ball tracking when viewing the court from above.

## The Problem

When tracking a tennis ball and projecting its 3D position onto a 2D court view, you see curved trajectories even when the ball is traveling in straight lines. This happens because:

1. The ball follows a parabolic path through 3D space
2. The homography projection maps this 3D path onto the 2D court plane
3. You're essentially seeing the "shadow" of the ball's flight on the ground
4. This creates distorted, curved paths that don't represent the true ball movement

## The Solution

The perspective correction system:

1. **Detects bounce points** using trajectory analysis
2. **Creates straight-line segments** between bounces
3. **Interpolates ball positions** along these straight lines
4. **Shows the true top-down view** of ball movement

## Usage

### Basic Commands

```bash
# Original (distorted) trajectory
python main.py --input_path media/tennis.mp4

# Corrected (straight-line) trajectory with improved tracking
python main.py --input_path media/tennis.mp4 --correct_perspective --outlier_removal advanced --smoothing

# Show both trajectories for comparison
python main.py --input_path media/tennis.mp4 --correct_perspective --show_comparison --bounce_sensitivity high

# Full improvements (recommended)
python main.py --input_path media/tennis.mp4 \
    --correct_perspective \
    --outlier_removal advanced \
    --smoothing \
    --interpolate_gaps \
    --debug_tracking
```

### New Command Line Options

- `--correct_perspective`: Use perspective-corrected trajectory for top-down view
- `--show_comparison`: Show both original (red) and corrected (green) trajectories with bounce points (yellow)
- `--bounce_sensitivity {low,medium,high,max}`: Control bounce detection sensitivity
- `--outlier_removal {basic,advanced}`: Choose outlier removal method
- `--smoothing`: Apply temporal smoothing to reduce tracking jitter
- `--interpolate_gaps`: Fill small gaps in ball tracking
- `--max_velocity PIXELS`: Maximum allowed ball velocity (default: 150 pixels/frame)
- `--debug_tracking`: Show detailed tracking improvement information

### Interactive Demo

Run the demo script for a guided experience:

```bash
python demo_perspective_correction.py
```

### Ball Tracking Improvements Demo

Test and compare tracking improvements:

```bash
python demo_tracking_improvements.py
```

## Visual Explanation

### Before Correction (Original)
- Ball appears to follow curved paths between bounces
- Trajectory is distorted by 3D perspective projection
- Sudden position jumps due to tracking outliers
- Difficult to analyze true ball placement and court coverage

### After Correction
- Ball travels in straight lines between bounces
- True top-down view of ball movement
- Smooth tracking without sudden jumps
- Accurate representation for tactical analysis

### Comparison View
- **Red circles**: Original trajectory (with distortion)
- **Green circles**: Corrected trajectory (straight lines)
- **Yellow circles**: Detected bounce points

## Technical Details

### Ball Tracking Improvements
The system now includes advanced outlier removal and smoothing:

1. **Velocity-based filtering**: Removes points with unrealistic speed
2. **Acceleration analysis**: Detects sudden direction changes
3. **Temporal consistency**: Ensures points fit with surrounding trajectory  
4. **Smoothing**: Reduces jitter and noise in tracking
5. **Gap interpolation**: Fills small missing segments

### Bounce Detection
The system uses multiple methods to detect bounces:

1. **Velocity analysis**: Detects sudden speed changes
2. **Direction analysis**: Identifies significant angle changes
3. **Advanced trajectory analysis**: Combines multiple factors for robust detection

### Trajectory Correction
1. Identifies segments between bounce points
2. Creates linear interpolation between valid endpoints
3. Maintains original timing while correcting spatial distortion

### Benefits
- More accurate court coverage analysis
- Better understanding of player positioning  
- Improved shot trajectory visualization
- Correct representation for tactical analysis
- Smoother tracking without sudden position jumps
- Reduced noise and tracking artifacts

## Files Modified

- `main.py`: Added command line options and logic for tracking improvements
- `utils/utils.py`: Added advanced outlier removal, smoothing, and trajectory correction functions
- `demo_perspective_correction.py`: Interactive demonstration script
- `demo_tracking_improvements.py`: Ball tracking improvement testing tool

## Example Output

The improved system provides:
- Straight ball paths between bounces (as they appear in real tennis)
- Smooth tracking without sudden position jumps
- Accurate landing positions
- True court coverage patterns
- Proper spatial relationships for analysis
- Reduced tracking noise and artifacts
