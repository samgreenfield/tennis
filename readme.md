# Tennis Video Analysis

This project analyzes tennis match videos to detect and track the ball and players, map their positions onto a virtual court, and generate visualizations and processed videos. It uses deep learning models for ball and player detection, homography for court mapping, and various utilities for smoothing, interpolation, and visualization.

## Features

- Detects and tracks tennis ball and players frame-by-frame
- Maps positions to a canonical court using homography
- Smooths and interpolates trajectories
- Visualizes results on both the original and virtual court
- Outputs a processed video with overlays

## Installation

1. **Clone the repository** (if you haven't already):

    ```bash
    git clone https://github.com/samgreenfield/tennis.git
    cd tennis
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Place your input video** in the `media/` directory (default: `media/tennis.mp4`).

2. **Run the main script**:

    ```bash
    python main.py
    ```

    By default, this will process `media/tennis.mp4` and output the result to `media/processed.mp4`. 

    **Optional arguments:**

    - `--input_path`: Path to input video (default: `media/tennis.mp4`)
    - `--output_path`: Path to save processed video (default: `media/processed.mp4`)
    - `--ball_model_path`: Path to ball detection model (default: `models/ball_model.pt`)
    - `--court_model_path`: Path to court detection model (default: `models/court_model.pt`)
    - `--player_tracking_model_path`: Path to player detection model (default: `models/yolo12m.pt`)
    - `--device`: Device to use for inference (`cpu`, `cuda`, or `mps`)

    Example:

    ```bash
    python main.py --input_path media/tennis.mp4 --output_path media/processed.mp4 --device cuda
    ```
    On first run, a YOLO model is downloaded based on `player_tracking_model_path` (default YOLOv12m). When an inference is performed on ball, court, or player positions, it will be stored as a stub automatically. In future runs, the same stub will be used if it exists in `stubs/xyz_stub.pkl`.

## Example Input and Output
Input:
https://github.com/user-attachments/assets/bf89e0ff-1c1f-4921-ba31-527653238944

Output:
https://github.com/user-attachments/assets/d19209ca-215d-4e8c-95fc-33eda524afde

## Credit
Credit to @yastrebksv (https://github.com/yastrebksv) for ball and court detection pretrained models and post-processing.

