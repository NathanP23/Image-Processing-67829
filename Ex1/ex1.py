"""
Scene Cut Detection using Histogram Analysis

Main Execution Flow (in main() function):
    1) video_to_frames() - Extract frames from video
    2) convert_frames_to_grayscale() - Convert RGB frames to grayscale
    3) compute_histograms() - Compute histograms for all frames
    4) compute_all_histogram_differences() - Compute differences between consecutive histograms
    5) find_peaks() - Detect scene cuts by finding peak differences
    6) print_top_peaks() - Print top N peaks for analysis
    7) save_analysis_figure() - Save individual and comprehensive analysis plots
    8) save_analysis_items_around_peak() - Save frames around detected cuts
    9) save_analysis_items_around_peak() - Save histograms around detected cuts

Helper Functions (supporting the main flow):
    - extract_video_info() - Extract video name and number from path
    - ensure_output_directory() - Create and return output directory path
    - plot_histogram_on_axis() - Reusable histogram plotting on matplotlib axis
    - compute_histogram() - Compute histogram for a single frame
    - compute_histogram_difference() - Compute difference between two histograms (L2, CORRELATION, CUMULATIVE)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os


# ===== HELPER FUNCTIONS FOR CODE REUSABILITY =====

def extract_video_info(video_path: str) -> Tuple[str, str]:
    """
    Extract video name and video number from video path.

    :param video_path: path to video file
    :return: tuple of (video_name, video_number)
             e.g., ("video1_category1", "1")
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_num = ''.join(filter(str.isdigit, video_name.split('_')[0]))
    return video_name, video_num


def ensure_output_directory(video_type: int, video_num: str, subfolder: str = "") -> str:
    """
    Create and return the output directory path.

    :param video_type: category of the video (1 or 2)
    :param video_num: video number as string
    :param subfolder: optional subfolder name (e.g., "peak_frames", "peak_histograms")
    :return: full path to output directory
    """
    output_base = os.path.join("outputs", f"cat{video_type}", f"vid{video_num}")
    if subfolder:
        output_base = os.path.join(output_base, subfolder)
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    return output_base


def plot_histogram_on_axis(ax, hist: np.ndarray, frame_idx: int, color: str = 'steelblue',
                           title_prefix: str = "") -> None:
    """
    Plot a histogram on a given matplotlib axis (reusable for multiple layouts).

    :param ax: matplotlib axis to plot on
    :param hist: normalized histogram array
    :param frame_idx: frame number (for title)
    :param color: color for the bars
    :param title_prefix: prefix for title (e.g., "Last Frame of Scene 1", "First Frame of Scene 2")
    """
    ax.bar(range(256), hist, width=1.0, color=color, alpha=0.7)
    ax.set_xlabel('Pixel Intensity', fontsize=10)
    ax.set_ylabel('Normalized Frequency', fontsize=10)
    ax.set_title(f'Histogram - {title_prefix} (Frame {frame_idx})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


def save_analysis_items_around_peak(video_path: str, grayscale_frames: List[np.ndarray],
                                    scene_cut: Tuple[int, int], video_type: int,
                                    item_type: str = 'frames',
                                    frames_before: int = 20, frames_after: int = 20,
                                    additional_cut: Tuple[int, int] = None) -> None:
    """
    Generic function to save frames or histograms around detected scene cut(s).
    Can analyze one or two scene cuts (primary and optional secondary).

    :param video_path: path to video file
    :param grayscale_frames: list of grayscale frames
    :param scene_cut: tuple of (last_frame_scene1, first_frame_scene2) - primary cut
    :param video_type: category of the video (1 or 2)
    :param item_type: type of items to save ('frames' or 'histograms')
    :param frames_before: number of frames to analyze before the peak (default: 20)
    :param frames_after: number of frames to analyze after the peak (default: 20)
    :param additional_cut: optional second tuple (last_frame_scene1, first_frame_scene2) to analyze
    """
    # Extract video information
    _, video_num = extract_video_info(video_path)

    # Create output directory structure
    subfolder = "peak_frames" if item_type == 'frames' else "peak_histograms"
    output_dir = ensure_output_directory(video_type, video_num, subfolder=subfolder)

    # Helper function to save items around a specific cut
    def save_items_for_cut(cut: Tuple[int, int], cut_label: str = ""):
        """Save frames or histograms around a specific scene cut."""
        peak_frame = cut[1]  # First frame of scene 2

        # Calculate frame range
        start_frame = max(0, peak_frame - frames_before)
        end_frame = min(len(grayscale_frames), peak_frame + frames_after + 1)

        # Save each frame or histogram
        for frame_idx in range(start_frame, end_frame):
            fig = plt.figure(figsize=(8, 6) if item_type == 'frames' else (10, 5))
            ax = fig.add_subplot(111)

            if item_type == 'frames':
                # Save frame image
                ax.imshow(grayscale_frames[frame_idx], cmap='gray')
                title_suffix = f" ({cut_label})" if cut_label else ""
                ax.set_title(f'Frame {frame_idx}{title_suffix}', fontsize=12, fontweight='bold')
                ax.axis('off')
            else:
                # Save histogram
                hist = compute_histogram(grayscale_frames[frame_idx])
                ax.bar(range(256), hist, width=1.0, color='steelblue', alpha=0.7)
                ax.set_xlabel('Pixel Intensity', fontsize=10)
                ax.set_ylabel('Normalized Frequency', fontsize=10)
                title_suffix = f" ({cut_label})" if cut_label else ""
                ax.set_title(f'Histogram - Frame {frame_idx}{title_suffix}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

            # Save with standardized naming
            filename_suffix = f"_{cut_label}" if cut_label else ""
            file_prefix = "frame" if item_type == 'frames' else "histogram"
            filename = f"{frame_idx}_{file_prefix}_cat{video_type}_vid{video_num}{filename_suffix}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)

    # Save items for primary scene cut
    save_items_for_cut(scene_cut, cut_label="primary")

    # Save items for additional scene cut if provided
    if additional_cut is not None:
        save_items_for_cut(additional_cut, cut_label="secondary")


# ===== CORE FUNCTIONS =====

def load_video(video_path: str) -> cv2.VideoCapture:
    """
    Load a video file.

    :param video_path: path to video file
    :return: cv2.VideoCapture object
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    return cap


def video_to_frames(video_path: str) -> List[np.ndarray]:
    """
    Extract all frames from a video file.

    :param video_path: path to video file
    :return: list of frames (BGR format)
    """
    cap = load_video(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def convert_frames_to_grayscale(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert RGB/BGR frames to grayscale.

    :param frames: list of BGR frames
    :return: list of grayscale frames
    """
    grayscale_frames = []
    for frame in frames:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_frames.append(frame_gray)

    return grayscale_frames


def save_frames(grayscale_frames: List[np.ndarray], output_dir: str = "frames",
                video_id: int = 1, category_id: int = 1) -> None:
    """
    Save grayscale frames to disk.

    :param grayscale_frames: list of grayscale frames
    :param output_dir: directory to save frames
    :param video_id: video identifier
    :param category_id: category identifier
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for frame_idx, frame in enumerate(grayscale_frames):
        filename = f"vid{video_id}_cat{category_id}_frame{frame_idx}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)


def compute_histogram(frame_gray: np.ndarray) -> np.ndarray:
    """
    Compute histogram for a grayscale frame.

    :param frame_gray: grayscale frame
    :return: normalized histogram
    """
    hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    # Normalize histogram
    hist = hist / np.sum(hist)
    return hist


def compute_histograms(grayscale_frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute histograms for all grayscale frames.

    :param grayscale_frames: list of grayscale frames
    :return: list of normalized histograms
    """
    histograms = []
    for frame in grayscale_frames:
        hist = compute_histogram(frame)
        histograms.append(hist)

    return histograms


def compute_histogram_difference(hist1: np.ndarray, hist2: np.ndarray,
                                distance_metric: str = 'L2') -> float:
    """
    Compute the difference between two histograms.

    :param hist1: first histogram
    :param hist2: second histogram
    :param distance_metric: distance metric ('L2', 'CORRELATION', or 'CUMULATIVE')
    :return: difference metric
    """
    if distance_metric == 'L2':
        # Euclidean distance (L2 norm)
        return float(np.sqrt(np.sum((hist1 - hist2) ** 2)))
    elif distance_metric == 'CORRELATION':
        # Using OpenCV's correlation method
        correlation = cv2.compareHist(hist1.astype(np.float32),
                                     hist2.astype(np.float32),
                                     cv2.HISTCMP_CORREL)
        return float(1 - correlation)
    elif distance_metric == 'CUMULATIVE':
        # Distance between cumulative histograms (robust to oversharpening)
        cum_hist1 = np.cumsum(hist1)
        cum_hist2 = np.cumsum(hist2)
        # Normalize cumulative histograms to [0, 1]
        cum_hist1 = cum_hist1 / cum_hist1[-1]
        cum_hist2 = cum_hist2 / cum_hist2[-1]
        # Compute L2 distance between cumulative histograms
        return float(np.sqrt(np.sum((cum_hist1 - cum_hist2) ** 2)))
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}. Must be 'L2', 'CORRELATION', or 'CUMULATIVE'")


def compute_all_histogram_differences(histograms: List[np.ndarray],
                                     distance_metric: str = 'L2') -> List[float]:
    """
    Compute differences between all consecutive histogram pairs.

    :param histograms: list of normalized histograms
    :param distance_metric: distance metric ('L2' or 'CORRELATION')
    :return: list of differences between consecutive frames
    """
    differences = []

    for i in range(len(histograms) - 1):
        diff = compute_histogram_difference(histograms[i], histograms[i + 1], distance_metric)
        differences.append(diff)

    return differences


def find_peaks(differences: List[float], percentile: int = 90) -> Tuple[int, int]:
    """
    Find the peak (scene cut) in histogram differences.
    Uses a percentile-based threshold to identify significant changes.

    :param differences: list of histogram differences
    :param percentile: percentile threshold for peak detection (0-100)
    :return: tuple of (last_frame_scene1, first_frame_scene2)
    """
    if not differences:
        raise ValueError("No differences provided")

    # Find maximum difference
    max_idx = int(np.argmax(differences))

    return (max_idx, max_idx + 1)


def plot_histogram_differences(differences: List[float],
                               output_file: str = None,
                               title: str = "Histogram Differences") -> None:
    """
    Plot the histogram differences over all frame transitions.

    :param differences: list of histogram differences
    :param output_file: optional file to save plot
    :param title: title of the plot
    """
    plt.figure(figsize=(12, 5))
    plt.plot(differences, linewidth=1.5)
    plt.xlabel('Frame Transition Index')
    plt.ylabel('Histogram Difference (L2 Distance)')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Mark the peak
    max_idx = np.argmax(differences)
    plt.plot(max_idx, differences[max_idx], 'ro', markersize=10, label='Detected Scene Cut')
    plt.legend()

    if output_file:
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

    plt.close()


def visualize_scene_cut(grayscale_frames: List[np.ndarray],
                        cut_frame1: int, cut_frame2: int,
                        output_file: str = None) -> None:
    """
    Visualize frames around the detected scene cut.

    :param grayscale_frames: list of grayscale frames
    :param cut_frame1: last frame of first scene
    :param cut_frame2: first frame of second scene
    :param output_file: optional file to save visualization
    """
    frames_to_show = [cut_frame1 - 1, cut_frame1, cut_frame2, cut_frame2 + 1]
    frames_to_show = [f for f in frames_to_show if 0 <= f < len(grayscale_frames)]

    fig, axes = plt.subplots(1, len(frames_to_show), figsize=(15, 4))
    if len(frames_to_show) == 1:
        axes = [axes]

    for i, frame_idx in enumerate(frames_to_show):
        axes[i].imshow(grayscale_frames[frame_idx], cmap='gray')
        axes[i].set_title(f"Frame {frame_idx}")
        axes[i].axis('off')

    fig.suptitle(f'Scene Cut: Transition from Frame {cut_frame1} to {cut_frame2}')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")

    plt.close()


def save_analysis_figure(video_path: str, grayscale_frames: List[np.ndarray],
                        differences: List[float], scene_cut: Tuple[int, int],
                        video_type: int, distance_metric: str = 'L2') -> None:
    """
    Save individual analysis plots and a comprehensive figure showing:
    - Histogram differences plot with detected scene cut
    - Frames around the scene cut
    - Histograms of the last frame of scene 1 and first frame of scene 2

    :param video_path: path to video file
    :param grayscale_frames: list of grayscale frames
    :param differences: list of histogram differences
    :param scene_cut: tuple of (last_frame_scene1, first_frame_scene2)
    :param video_type: category of the video (1 or 2)
    :param distance_metric: distance metric used ('L2' or 'CORRELATION')
    """
    # Extract video information
    video_name, video_num = extract_video_info(video_path)
    cut_frame1, cut_frame2 = scene_cut

    # Create output directory structure
    output_base = ensure_output_directory(video_type, video_num)

    # Helper function to save individual plot
    def save_plot(fig, plot_name):
        """Save a single plot with standardized naming."""
        filename = f"{plot_name}_cat{video_type}_vid{video_num}.png"
        filepath = os.path.join(output_base, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ===== PLOT 1: Histogram Differences =====
    fig1 = plt.figure(figsize=(12, 5))
    ax1 = fig1.add_subplot(111)
    ax1.plot(differences, linewidth=1.5, color='steelblue')
    max_idx = np.argmax(differences)
    ax1.plot(max_idx, differences[max_idx], 'ro', markersize=12,
            label=f'Scene Cut (frame {cut_frame1}->{cut_frame2})')
    ax1.set_xlabel('Frame Transition Index', fontsize=11)
    ax1.set_ylabel(f'Histogram Difference ({distance_metric} Distance)', fontsize=11)
    ax1.set_title(f'Histogram Differences - {video_name} (Metric: {distance_metric})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    save_plot(fig1, "histogram_differences")

    # ===== PLOT 2: Frames Around Scene Cut =====
    frames_to_show = [cut_frame1, cut_frame2]
    frames_to_show = [f for f in frames_to_show if 0 <= f < len(grayscale_frames)]

    n_frames = len(frames_to_show)
    fig2, axes = plt.subplots(1, n_frames, figsize=(10, 4))
    if n_frames == 1:
        axes = [axes]

    for i, frame_idx in enumerate(frames_to_show):
        axes[i].imshow(grayscale_frames[frame_idx], cmap='gray')
        axes[i].set_title(f"Frame {frame_idx}", fontsize=10)
        axes[i].axis('off')

    fig2.suptitle(f'Scene Cut Frames - {video_name}', fontsize=12, fontweight='bold')
    fig2.tight_layout()
    save_plot(fig2, "scene_cut_frames")

    # ===== PLOT 3 & 4: Histograms of Scene Cut Frames (Side-by-side) =====
    hist1 = compute_histogram(grayscale_frames[cut_frame1])
    hist2 = compute_histogram(grayscale_frames[cut_frame2])

    fig_histograms, axes_hist = plt.subplots(1, 2, figsize=(14, 5))

    plot_histogram_on_axis(axes_hist[0], hist1, cut_frame1, color='steelblue',
                          title_prefix='Last Frame of Scene 1')
    plot_histogram_on_axis(axes_hist[1], hist2, cut_frame2, color='forestgreen',
                          title_prefix='First Frame of Scene 2')

    fig_histograms.suptitle(f'Histograms Around Scene Cut - {video_name}', fontsize=12, fontweight='bold')
    fig_histograms.tight_layout()
    save_plot(fig_histograms, "histograms_comparison")

    # ===== COMPREHENSIVE FIGURE: All plots together =====
    fig_comprehensive = plt.figure(figsize=(16, 10))
    gs = fig_comprehensive.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Re-create all plots in the comprehensive figure
    ax1_comp = fig_comprehensive.add_subplot(gs[0, :])
    ax1_comp.plot(differences, linewidth=1.5, color='steelblue')
    ax1_comp.plot(max_idx, differences[max_idx], 'ro', markersize=12,
                 label=f'Scene Cut (frame {cut_frame1}->{cut_frame2})')
    ax1_comp.set_xlabel('Frame Transition Index', fontsize=11)
    ax1_comp.set_ylabel(f'Histogram Difference ({distance_metric} Distance)', fontsize=11)
    ax1_comp.set_title(f'Histogram Differences - {video_name} (Metric: {distance_metric})', fontsize=13, fontweight='bold')
    ax1_comp.grid(True, alpha=0.3)
    ax1_comp.legend(fontsize=10)

    gs_frames = gs[1, :].subgridspec(1, n_frames)
    for i, frame_idx in enumerate(frames_to_show):
        ax = fig_comprehensive.add_subplot(gs_frames[0, i])
        ax.imshow(grayscale_frames[frame_idx], cmap='gray')
        ax.set_title(f"Frame {frame_idx}", fontsize=10)
        ax.axis('off')

    ax3_comp = fig_comprehensive.add_subplot(gs[2, 0])
    plot_histogram_on_axis(ax3_comp, hist1, cut_frame1, color='steelblue',
                          title_prefix='Last Frame of Scene 1')

    ax4_comp = fig_comprehensive.add_subplot(gs[2, 1])
    plot_histogram_on_axis(ax4_comp, hist2, cut_frame2, color='forestgreen',
                          title_prefix='First Frame of Scene 2')

    fig_comprehensive.suptitle(f'Complete Analysis - {video_name}', fontsize=14, fontweight='bold', y=0.995)
    save_plot(fig_comprehensive, "analysis_complete")


def print_top_peaks(differences: List[float], video_name: str, top_n: int = 5) -> None:
    """
    Print the top N biggest peaks in histogram differences with their frame transitions.

    :param differences: list of histogram differences
    :param video_name: name of the video (for display)
    :param top_n: number of top peaks to print (default: 5)
    """
    # Get indices sorted by peak value (descending)
    sorted_indices = np.argsort(differences)[::-1]

    print(f"        {'='*70}")
    print(f"        Top {top_n} Peaks - {video_name}")
    print(f"        {'='*70}")
    print(f"        {'Rank':<6} {'Value':<12} {'Transition':<20} {'Frame Range'}")
    print(f"        {'-'*70}")

    for rank, idx in enumerate(sorted_indices[:top_n], 1):
        peak_value = differences[idx]
        frame_from = idx
        frame_to = idx + 1
        transition_str = f"Frame {frame_from} → {frame_to}"
        frame_range = f"[{frame_from}, {frame_to}]"

        print(f"        {rank:<6} {peak_value:<12.6f} {transition_str:<20} {frame_range}")

    print(f"        {'='*70}")


def main(video_path: str, video_type: int) -> Tuple[int, int]:
    """
    Main entry point for scene cut detection.

    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which
             the scene cut was detected (last frame of first scene, first frame of second scene)
    """
    # Define histogram distance metric for each category
    metrics_by_category = {
        1: 'L2',
        2: 'CUMULATIVE'
    }

    # Step 1: Load and extract frames
    frames = video_to_frames(video_path)

    # Step 2: Convert to grayscale
    grayscale_frames = convert_frames_to_grayscale(frames)

    # Step 3: Compute histograms
    histograms = compute_histograms(grayscale_frames)

    # Step 4: Compute histogram differences
    differences = compute_all_histogram_differences(histograms, distance_metric=metrics_by_category[video_type])

    # Step 5: Find scene cut
    scene_cut = find_peaks(differences)

    # Step 5b: Print top top_n peaks for analysis
    video_name, _ = extract_video_info(video_path)
    print_top_peaks(differences, video_name, top_n=3)

    # Step 6: Save a comprehensive analysis figure
    save_analysis_figure(video_path, grayscale_frames, differences, scene_cut, video_type, distance_metric=metrics_by_category[video_type])

    # Step 7: Save individual frames around the detected scene cut(s)
    # Define additional cuts for specific videos to analyze multiple peaks
    additional_cut = None
    if video_path.endswith("video4_category2.mp4"):
        additional_cut = (74, 75) # Video 4: Analyze the actual scene cut at frame 74 (before the detected false positive at 123)
    elif video_path.endswith("video3_category2.mp4"):
        additional_cut = (34, 35) # Video 3: Analyze the fade transition at frame 34 (before the detected correct cut at 174)

    # Step 8: Save individual frames and histograms around the detected scene cut(s)
    save_analysis_items_around_peak(video_path, grayscale_frames, scene_cut, video_type,
                                    item_type='frames', frames_before=5, frames_after=5,
                                    additional_cut=additional_cut)
    save_analysis_items_around_peak(video_path, grayscale_frames, scene_cut, video_type,
                                    item_type='histograms', frames_before=5, frames_after=5,
                                    additional_cut=additional_cut)

    return scene_cut


if __name__ == "__main__":
    # List of videos to process: (video_path, video_type)
    videos_to_process = [
        ("Exercise Inputs/video1_category1.mp4", 1),
        ("Exercise Inputs/video2_category1.mp4", 1),
        ("Exercise Inputs/video3_category2.mp4", 2),
        ("Exercise Inputs/video4_category2.mp4", 2),
    ]

    print("\n" + "=" * 70)
    print("BATCH SCENE CUT DETECTION ANALYSIS")
    print("=" * 70)

    results = {}

    # Process all videos by calling main() API function
    for video_path, video_type in videos_to_process:
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        try:
            video_name, _ = extract_video_info(video_path)
            print(f"\n    Sending: {video_name} (Category {video_type}) to provided main() API function)")

            # Call main() API function - it handles all logic
            scene_cut = main(video_path, video_type)
            results[video_name] = scene_cut
            print(f"    Received: cut Scene {scene_cut[0]} -> {scene_cut[1]} from provided main() API function")

        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    for video_name, (frame1, frame2) in results.items():
        print(f"    {video_name}: Scene cut at frames {frame1} -> {frame2}")
    print("=" * 70)
