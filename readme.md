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
    git clone <your-repo-url>
    cd <repo-folder>
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    The main dependencies include:
    - `torch`
    - `opencv-python`
    - `numpy`
    - `tqdm`
    - `ultralytics` (for YOLO models)

3. **Download or place the required models** in the `models/` directory:
    - `ball_model.pt`
    - `court_model.pt`
    - `yolo12m.pt` (or another YOLO model)

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

## Input and Output Videos

- **Input:**  
  `media/tennis.mp4` — Raw tennis match video.

- **Output:**  
  `media/processed.mp4` — Video with detected ball, players, and court overlays.