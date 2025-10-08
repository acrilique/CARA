#!/usr/bin/env python3
"""
CARA vs Librosa Onset Detection Comparison Tool

This script compares CARA's onset detection output (from test_onset) with librosa's
implementation, providing comprehensive analysis and validation.

Usage:
    python compare_cara_librosa_onset.py [cara_output.txt] [audio_file.wav]

If no arguments provided, uses default paths:
    - CARA output: ../outputs/onset_envelope.txt
    - Audio file: files/black_woodpecker.wav
"""

import sys
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

def load_cara_onset(cara_file):
    """
    Load CARA's onset envelope from text file.
    
    Args:
        cara_file (str): Path to CARA's onset output file
        
    Returns:
        np.ndarray: CARA onset envelope
    """
    try:
        # Load the file, skipping header lines that start with #
        data = np.loadtxt(cara_file, comments='#')
        
        # Check if data is 2D (frame_index, onset_strength) or 1D (onset_strength only)
        if data.ndim == 2 and data.shape[1] == 2:
            # Extract only the onset strength values (second column)
            cara_onset = data[:, 1]
            print(f"✅ Loaded CARA onset: {len(cara_onset)} frames from {cara_file} (2-column format)")
        elif data.ndim == 1:
            # Data is already 1D onset strength values
            cara_onset = data
            print(f"✅ Loaded CARA onset: {len(cara_onset)} frames from {cara_file} (1-column format)")
        else:
            print(f"❌ Unexpected data format in {cara_file}: shape {data.shape}")
            return None
            
        return cara_onset
    except Exception as e:
        print(f"❌ Error loading CARA onset from {cara_file}: {e}")
        return None

def compute_librosa_onset(audio_file, use_corrected_params=True):
    """
    Compute librosa onset envelope with parameters matching CARA.
    
    Args:
        audio_file (str): Path to audio file
        use_corrected_params (bool): Use center=False to match CARA's frame count
        
    Returns:
        tuple: (onset_envelope, sample_rate, stft_frames)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        print(f"📁 Loaded audio: {len(y)} samples, {sr} Hz, {len(y)/sr:.3f}s")
        
        # STFT parameters (matching CARA's likely configuration)
        n_fft = 2048
        hop_length = 512
        n_mels = 128
        
        # Use center=False to match CARA's frame count (no padding)
        center = not use_corrected_params
        
        print(f"🔧 STFT parameters: n_fft={n_fft}, hop_length={hop_length}, center={center}")
        
        # Compute STFT
        stft_complex = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, center=center)
        stft_power = np.abs(stft_complex)**2
        
        print(f"📊 STFT shape: {stft_power.shape} (freq_bins x time_frames)")
        
        # Build mel filterbank (using Slaney scale, matching CARA's fix)
        mel_filters = librosa.filters.mel(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
            fmax=sr / 2.0,
            htk=False,  # Use Slaney scale (CARA's fix)
            norm=None   # No normalization to match CARA
        )
        
        # Apply mel filterbank
        mel_spec = np.dot(mel_filters, stft_power)
        print(f"🎵 Mel spectrogram shape: {mel_spec.shape}")
        
        # Convert to dB
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(
            S=mel_db,
            sr=sr,
            hop_length=hop_length,
            aggregate=np.median,  # CARA likely uses median
            lag=1,
            max_size=1,
            center=False  # No centering to match CARA
        )
        
        print(f"🎯 Librosa onset envelope: {len(onset_env)} frames")
        
        return onset_env, sr, stft_power.shape[1]
        
    except Exception as e:
        print(f"❌ Error computing librosa onset from {audio_file}: {e}")
        return None, None, None

def analyze_performance(cara_onset, librosa_onset):
    """
    Analyze performance metrics between CARA and librosa onset envelopes.
    
    Args:
        cara_onset (np.ndarray): CARA onset envelope
        librosa_onset (np.ndarray): Librosa onset envelope
        
    Returns:
        dict: Performance metrics
    """
    # Ensure same length for comparison
    min_len = min(len(cara_onset), len(librosa_onset))
    cara_trimmed = cara_onset[:min_len]
    librosa_trimmed = librosa_onset[:min_len]
    
    # Calculate metrics
    mae = np.mean(np.abs(cara_trimmed - librosa_trimmed))
    rmse = np.sqrt(np.mean((cara_trimmed - librosa_trimmed)**2))
    max_diff = np.max(np.abs(cara_trimmed - librosa_trimmed))
    correlation = np.corrcoef(cara_trimmed, librosa_trimmed)[0, 1]
    
    # Additional metrics
    bias = np.mean(cara_trimmed - librosa_trimmed)
    scale_ratio = np.mean(cara_trimmed) / np.mean(librosa_trimmed) if np.mean(librosa_trimmed) != 0 else 1.0
    
    # Count problematic frames
    large_diff_threshold = 0.05
    problematic_frames = np.sum(np.abs(cara_trimmed - librosa_trimmed) > large_diff_threshold)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'max_diff': max_diff,
        'correlation': correlation,
        'bias': bias,
        'scale_ratio': scale_ratio,
        'problematic_frames': problematic_frames,
        'total_frames': min_len,
        'cara_mean': np.mean(cara_trimmed),
        'librosa_mean': np.mean(librosa_trimmed),
        'cara_std': np.std(cara_trimmed),
        'librosa_std': np.std(librosa_trimmed)
    }

def create_comparison_plot(cara_onset, librosa_onset, metrics, output_path="onset_comparison.png"):
    """
    Create comprehensive comparison plot.
    
    Args:
        cara_onset (np.ndarray): CARA onset envelope
        librosa_onset (np.ndarray): Librosa onset envelope
        metrics (dict): Performance metrics
        output_path (str): Output plot path
    """
    # Trim to same length
    min_len = min(len(cara_onset), len(librosa_onset))
    cara_trimmed = cara_onset[:min_len]
    librosa_trimmed = librosa_onset[:min_len]
    
    # Create time axis (assuming 512 hop length, 22050 sample rate)
    time_frames = np.arange(min_len)
    time_seconds = time_frames * 512 / 22050  # Approximate
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Onset envelopes comparison
    axes[0, 0].plot(time_seconds, cara_trimmed, 'b-', label='CARA', alpha=0.8, linewidth=1.5)
    axes[0, 0].plot(time_seconds, librosa_trimmed, 'r--', label='Librosa', alpha=0.8, linewidth=1.5)
    axes[0, 0].set_title(f'Onset Envelopes Comparison\nMAE: {metrics["mae"]:.6f}, Correlation: {metrics["correlation"]:.6f}')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Onset Strength')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Difference plot
    diff = cara_trimmed - librosa_trimmed
    axes[0, 1].plot(time_seconds, diff, 'g-', alpha=0.8, linewidth=1)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].set_title(f'Difference (CARA - Librosa)\nRMSE: {metrics["rmse"]:.6f}, Max: {metrics["max_diff"]:.6f}')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Difference')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot
    axes[1, 0].scatter(librosa_trimmed, cara_trimmed, alpha=0.6, s=20)
    # Perfect correlation line
    min_val = min(np.min(librosa_trimmed), np.min(cara_trimmed))
    max_val = max(np.max(librosa_trimmed), np.max(cara_trimmed))
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect correlation')
    axes[1, 0].set_title(f'CARA vs Librosa Scatter\nCorrelation: {metrics["correlation"]:.6f}')
    axes[1, 0].set_xlabel('Librosa Onset Strength')
    axes[1, 0].set_ylabel('CARA Onset Strength')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistics summary
    axes[1, 1].axis('off')
    
    stats_text = f"""
PERFORMANCE METRICS

📊 Accuracy Metrics:
   MAE: {metrics['mae']:.6f}
   RMSE: {metrics['rmse']:.6f}
   Max Diff: {metrics['max_diff']:.6f}
   Correlation: {metrics['correlation']:.6f}

📈 Statistics:
   Bias: {metrics['bias']:.6f}
   Scale Ratio: {metrics['scale_ratio']:.6f}
   Problematic Frames: {metrics['problematic_frames']}/{metrics['total_frames']}

🔧 Data Summary:
   CARA Mean: {metrics['cara_mean']:.6f}
   Librosa Mean: {metrics['librosa_mean']:.6f}
   CARA Std: {metrics['cara_std']:.6f}
   Librosa Std: {metrics['librosa_std']:.6f}
"""
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Comparison plot saved to: {output_path}")

def print_summary_report(metrics, cara_file, audio_file):
    """
    Print comprehensive summary report.
    
    Args:
        metrics (dict): Performance metrics
        cara_file (str): CARA output file path
        audio_file (str): Audio file path
    """

    print("\n" + "="*80)
    print("🎯 CARA vs LIBROSA ONSET DETECTION COMPARISON REPORT")
    print("="*80)
    
    print(f"📁 Input Files:")
    print(f"   CARA Output: {cara_file}")
    print(f"   Audio File: {audio_file}")
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Mean Absolute Error (MAE): {metrics['mae']:.6f}")
    print(f"   Root Mean Square Error (RMSE): {metrics['rmse']:.6f}")
    print(f"   Maximum Difference: {metrics['max_diff']:.6f}")
    print(f"   Correlation Coefficient: {metrics['correlation']:.6f}")
    print(f"   Systematic Bias: {metrics['bias']:.6f}")
    print(f"   Scale Ratio (CARA/Librosa): {metrics['scale_ratio']:.6f}")
    
    print(f"\n🔍 DETAILED ANALYSIS:")
    print(f"   Total Frames Compared: {metrics['total_frames']}")
    print(f"   Problematic Frames (>0.05 diff): {metrics['problematic_frames']}")
    print(f"   Problematic Frame Rate: {metrics['problematic_frames']/metrics['total_frames']*100:.1f}%")
    
    print(f"\n📈 STATISTICAL SUMMARY:")
    print(f"   CARA - Mean: {metrics['cara_mean']:.6f}, Std: {metrics['cara_std']:.6f}")
    print(f"   Librosa - Mean: {metrics['librosa_mean']:.6f}, Std: {metrics['librosa_std']:.6f}")
    
    print("="*80)

def main():
    """
    Main function to run the comparison.
    """
    # Parse command line arguments
    if len(sys.argv) >= 3:
        cara_file = sys.argv[1]
        audio_file = sys.argv[2]
    elif len(sys.argv) == 2:
        cara_file = sys.argv[1]
        audio_file = "files/black_woodpecker.wav"
    else:
        cara_file = "../outputs/onset_envelope.txt"
        audio_file = "files/black_woodpecker.wav"
    
    print("🎯 CARA vs Librosa Onset Detection Comparison")
    print("="*60)
    
    # Check if files exist
    if not os.path.exists(cara_file):
        print(f"❌ CARA output file not found: {cara_file}")
        print("💡 Run CARA's test_onset first to generate the output file")
        return 1
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return 1
    
    # Load CARA onset
    cara_onset = load_cara_onset(cara_file)
    if cara_onset is None:
        return 1
    
    # Compute librosa onset
    librosa_onset, sr, stft_frames = compute_librosa_onset(audio_file, use_corrected_params=True)
    if librosa_onset is None:
        return 1
    
    # Analyze performance
    metrics = analyze_performance(cara_onset, librosa_onset)
    
    # Create output directory
    output_dir = Path("../outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison plot
    plot_path = output_dir / "cara_librosa_onset_comparison.png"
    create_comparison_plot(cara_onset, librosa_onset, metrics, str(plot_path))
    
    # Save librosa reference for future use
    librosa_ref_path = output_dir / "librosa_onset_reference.txt"
    np.savetxt(librosa_ref_path, librosa_onset, fmt="%.8f", 
               header=f"Librosa Onset Strength Reference\nFrames: {len(librosa_onset)}\nParameters: n_fft=2048, hop_length=512, center=False, Slaney mel scale")
    print(f"💾 Librosa reference saved to: {librosa_ref_path}")
    
    # Print summary report
    print_summary_report(metrics, cara_file, audio_file)
    
    # Return success
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
