# Improved Bounce Detection Guide

## The Problem with Missing Bounces

The original bounce detection was conservative and could miss bounces due to:

1. **High angle threshold** - Only detected very sharp direction changes
2. **Minimum segment filtering** - Removed bounces that were close together  
3. **Single detection method** - Relied only on direction changes

## New Bounce Detection System

### Multiple Detection Methods

1. **Direction Change Analysis**: Detects angle changes in ball trajectory
2. **Velocity Change Analysis**: Identifies sudden speed changes (spikes/drops)
3. **Y-Direction Analysis**: Finds local minima (ball touching ground)

### Sensitivity Levels

| Sensitivity | Angle Threshold | Min Segment | Methods | Best For |
|-------------|----------------|-------------|---------|----------|
| **low**     | 60°            | 8 frames    | Direction only | Noisy tracking |
| **medium**  | 45°            | 5 frames    | All methods | General use |
| **high**    | 30°            | 3 frames    | All methods | Clean tracking |
| **max**     | 22.5°          | 2 frames    | All methods | Maximum detection |

## Usage Examples

### Basic Usage
```bash
# Use medium sensitivity (default)
python main.py --input_path media/tennis.mp4 --correct_perspective

# High sensitivity to catch more bounces
python main.py --input_path media/tennis.mp4 --correct_perspective --bounce_sensitivity high

# Maximum sensitivity 
python main.py --input_path media/tennis.mp4 --correct_perspective --bounce_sensitivity max

# Show debug information
python main.py --input_path media/tennis.mp4 --correct_perspective --debug_bounces
```

### Comparison View
```bash
# Compare original vs corrected with bounce points marked
python main.py --input_path media/tennis.mp4 --correct_perspective --show_comparison --bounce_sensitivity high
```

### Demo Script
```bash
# Interactive demo with sensitivity selection
python demo_perspective_correction.py
```

### Bounce Detection Tuning
```bash
# Analyze and tune bounce detection parameters
python tune_bounce_detection.py
```

## Troubleshooting Missing Bounces

### Step 1: Check Current Detection
```bash
python main.py --input_path media/tennis.mp4 --correct_perspective --show_comparison --debug_bounces
```

Look for console output like:
```
🔍 Using bounce detection sensitivity: medium  
🎾 Detected 3 bounces at frames: [145, 289, 456]
Direction bounce at frame 145, angle: 67.2°
Velocity bounce at frame 289, dv: 23.4
```

### Step 2: Increase Sensitivity
If you see fewer bounces than expected:

```bash
# Try high sensitivity
python main.py --input_path media/tennis.mp4 --correct_perspective --bounce_sensitivity high --debug_bounces

# Or maximum sensitivity
python main.py --input_path media/tennis.mp4 --correct_perspective --bounce_sensitivity max --debug_bounces
```

### Step 3: Use Tuning Tool
```bash
python tune_bounce_detection.py
```

This will:
- Show bounce detection with all sensitivity levels
- Let you test custom parameters interactively
- Create visualization showing detected bounces

### Step 4: Manual Parameter Tuning

If the preset sensitivities don't work, modify the detection function in `utils/utils.py`:

```python
# In detect_bounces_advanced(), try these changes:

# More sensitive angle detection
angle_threshold=np.pi/8,  # 22.5 degrees instead of 45

# Allow closer bounces
min_segment_length=1,     # Instead of 5

# More sensitive velocity detection  
velocity_threshold = np.std(velocities) * 1.0  # Instead of 1.5
```

## Visual Debugging

### Comparison View Legend
- **Red circles**: Original trajectory (with 3D perspective distortion)
- **Green circles**: Corrected trajectory (straight lines between bounces)  
- **Yellow circles**: Detected bounce points

### Expected Results
For a typical tennis rally, you should see:
- 2-6 bounces depending on rally length
- Bounces roughly where the ball hits the ground
- Straight green lines connecting bounce points

## Common Issues and Solutions

### Issue: No bounces detected
**Solution**: Increase sensitivity to "high" or "max"

### Issue: Too many false bounces  
**Solution**: Decrease sensitivity to "low" or increase `min_segment_length`

### Issue: Bounces in wrong locations
**Solution**: Check that court detection is working properly first

### Issue: Missing bounces at video start/end
**Solution**: This is normal - detection needs a few frames for analysis

## Technical Details

### Detection Logic
1. **Valid Points**: Extract non-None ball positions
2. **Direction Analysis**: Calculate angle between trajectory segments  
3. **Velocity Analysis**: Find sudden speed changes
4. **Y-Analysis**: Locate vertical minima (ball hitting ground)
5. **Filtering**: Remove bounces too close together
6. **Combination**: Merge results from multiple methods

### Parameters Explained
- `angle_threshold`: Minimum direction change to consider a bounce (radians)
- `min_segment_length`: Minimum frames between bounces (prevents noise)
- `use_multiple_methods`: Whether to combine all detection methods
- `debug`: Print detailed detection information

### Performance Tips
- Start with "medium" sensitivity
- Use "high" for clean, high-quality ball tracking
- Use "low" for noisy or low-quality tracking
- Use debug mode to understand what's being detected
