# Scene Cut Detection using Histogram Analysis

A Python implementation for detecting scene cuts in videos using histogram-based analysis with support for multiple distance metrics (L2, CORRELATION, CUMULATIVE).

## Installation

### 1. Create a Virtual Environment

```bash
python3 -m venv IMPR_ex1_venv
```

### 2. Activate the Virtual Environment

**On macOS/Linux:**
```bash
source IMPR_ex1_venv/bin/activate
```

**On Windows:**
```bash
IMPR_ex1_venv\Scripts\activate
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Required Packages

Only two packages are needed:

```bash
pip install opencv-python matplotlib
```

That's it! These two packages will automatically install all necessary dependencies (numpy, pillow, etc.).

## Usage

### Project Structure

```
.
├── Exercise Inputs/          # Input video files
│   ├── video1_category1.mp4
│   ├── video2_category1.mp4
│   ├── video3_category2.mp4
│   └── video4_category2.mp4
├── outputs/                  # Output directory (auto-created)
│   └── cat{category}/
│       └── vid{video_num}/
│           ├── histogram_differences_cat{X}_vid{Y}.png
│           ├── scene_cut_frames_cat{X}_vid{Y}.png
│           ├── histograms_comparison_cat{X}_vid{Y}.png
│           ├── analysis_complete_cat{X}_vid{Y}.png
│           ├── peak_frames/
│           │   └── {frame_num}_frame_cat{X}_vid{Y}_{primary|secondary}.png
│           └── peak_histograms/
│               └── {frame_num}_histogram_cat{X}_vid{Y}_{primary|secondary}.png
├── ex1.py                    # Main implementation with API entry point
└── README.md                 # This file
```

### Running the Script

The simplest way to run the analysis on all 4 videos:

```bash
python ex1.py
```

This will:
1. Process all 4 videos in the `Exercise Inputs/` directory
2. Detect scene cuts using histogram analysis
3. Generate analysis plots and save them to the `outputs/` directory
4. Print results to the console

### Using the API

If you want to use the scene cut detection in your own code:

```python
from ex1 import main

# Run scene cut detection on a video
video_path = "Exercise Inputs/video1_category1.mp4"
video_type = 1  # Category 1 or 2

scene_cut = main(video_path, video_type)
print(f"Scene cut detected at frames: {scene_cut[0]} -> {scene_cut[1]}")
```

**Return Value:** Tuple of `(last_frame_of_scene1, first_frame_of_scene2)`

## Function Call Flow

This diagram shows the chronological execution flow of the `main()` function and all helper functions called:

```
main(video_path, video_type)
│
├─ 1. video_to_frames(video_path)
│  └─ load_video(video_path)
│
├─ 2. convert_frames_to_grayscale(frames)
│
├─ 3. compute_histograms(grayscale_frames)
│  └─ compute_histogram(frame_gray) [called for each frame]
│
├─ 4. compute_all_histogram_differences(histograms, distance_metric)
│  └─ compute_histogram_difference(hist1, hist2, distance_metric) [called for each pair]
│
├─ 5. find_peaks(differences)
│
├─ 5b. extract_video_info(video_path)
│
├─ 6. print_top_peaks(differences, video_name, top_n=3)
│
├─ 7. save_analysis_figure(video_path, grayscale_frames, differences, scene_cut, video_type, distance_metric)
│  ├─ extract_video_info(video_path)
│  ├─ ensure_output_directory(video_type, video_num)
│  ├─ compute_histogram(grayscale_frames[cut_frame1])
│  ├─ compute_histogram(grayscale_frames[cut_frame2])
│  ├─ plot_histogram_on_axis(ax, hist1, cut_frame1, color='steelblue', ...)
│  ├─ plot_histogram_on_axis(ax, hist2, cut_frame2, color='forestgreen', ...)
│  └─ save_plot(fig, plot_name) [called 4 times: histogram_differences, scene_cut_frames, histograms_comparison, analysis_complete]
│
├─ 8. save_analysis_items_around_peak(video_path, grayscale_frames, scene_cut, video_type, item_type='frames', ...)
│  ├─ extract_video_info(video_path)
│  ├─ ensure_output_directory(video_type, video_num, subfolder='peak_frames')
│  └─ [for each frame in range]:
│     └─ save_items_for_cut(cut, cut_label) [called for primary and optional secondary cuts]
│        └─ imshow(grayscale_frames[frame_idx], cmap='gray') [for each frame]
│
└─ 9. save_analysis_items_around_peak(video_path, grayscale_frames, scene_cut, video_type, item_type='histograms', ...)
   ├─ extract_video_info(video_path)
   ├─ ensure_output_directory(video_type, video_num, subfolder='peak_histograms')
   └─ [for each frame in range]:
      └─ save_items_for_cut(cut, cut_label) [called for primary and optional secondary cuts]
         ├─ compute_histogram(grayscale_frames[frame_idx])
         └─ bar(range(256), hist, ...) [histogram plot for each frame]
```

## Distance Metrics

The implementation supports three distance metrics for histogram comparison:

- **L2 (Euclidean Distance):** Used for Category 1 videos. Measures overall histogram differences.
- **CORRELATION:** Compares histogram shape, ignoring brightness variations.
- **CUMULATIVE:** Distance between cumulative distribution functions (CDFs). Robust to localization artifacts like oversharpening. Used for Category 2 videos.

## Output Files

For each video, the following files are generated in `outputs/cat{X}/vid{Y}/`:

1. **histogram_differences_cat{X}_vid{Y}.png** - Plot showing histogram differences across all frame transitions with detected scene cut marked
2. **scene_cut_frames_cat{X}_vid{Y}.png** - Side-by-side frames at the scene cut boundary
3. **histograms_comparison_cat{X}_vid{Y}.png** - Side-by-side histograms of the two frames at the cut
4. **analysis_complete_cat{X}_vid{Y}.png** - Comprehensive figure with all 4 plots combined
5. **peak_frames/** - Individual frame images around detected cuts
6. **peak_histograms/** - Individual histogram plots around detected cuts

## Configuration

Video-specific settings are configured in the `main()` function:

**Distance Metrics by Category:**
```python
metrics_by_category = {
    1: 'L2',
    2: 'CUMULATIVE'
}
```

**Additional Cuts for Analysis:**
Specific videos have optional secondary cuts defined for comparison:
- **video4_category2.mp4:** Additional cut at frames (74, 75)
- **video3_category2.mp4:** Additional cut at frames (34, 35)

These settings can be modified in the `main()` function to adjust behavior for different videos.

## Notes

- Requires Python 3.6+
- All videos should be in the `Exercise Inputs/` directory
- Output directory is automatically created if it doesn't exist
- The implementation is optimized for detecting abrupt scene transitions
