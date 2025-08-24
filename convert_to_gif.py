"""
Convert MP4 videos to GIFs for GitHub display
"""
from moviepy import VideoFileClip
import os

def convert_mp4_to_gif(mp4_path, gif_path, fps=10, resize_factor=0.5):
    """
    Convert MP4 to GIF with optimization for GitHub
    Args:
        mp4_path (str): Path to input MP4 file
        gif_path (str): Path to output GIF file
        fps (int): Frames per second for GIF (lower = smaller file)
        resize_factor (float): Scale factor to reduce size (0.5 = half size)
    """
    print(f"Converting {mp4_path} to {gif_path}...")
    
    # Load video clip
    clip = VideoFileClip(mp4_path)
    
    # Resize to reduce file size
    if resize_factor != 1.0:
        clip = clip.resized(resize_factor)
    
    # Convert to GIF with reduced fps
    clip.write_gif(gif_path, fps=fps, logger=None)
    clip.close()
    
    # Get file sizes
    mp4_size = os.path.getsize(mp4_path) / (1024 * 1024)  # MB
    gif_size = os.path.getsize(gif_path) / (1024 * 1024)  # MB
    
    print(f"✓ Conversion complete!")
    print(f"  Original MP4: {mp4_size:.1f} MB")
    print(f"  Generated GIF: {gif_size:.1f} MB")

def main():
    videos_dir = "videos"
    
    # Convert before training video
    before_mp4 = os.path.join(videos_dir, "before_training.mp4")
    before_gif = os.path.join(videos_dir, "before_training.gif")
    
    if os.path.exists(before_mp4):
        convert_mp4_to_gif(before_mp4, before_gif)
    else:
        print(f"Warning: {before_mp4} not found")
    
    # Convert after training video
    after_mp4 = os.path.join(videos_dir, "after_training.mp4")
    after_gif = os.path.join(videos_dir, "after_training.gif")
    
    if os.path.exists(after_mp4):
        convert_mp4_to_gif(after_mp4, after_gif)
    else:
        print(f"Warning: {after_mp4} not found")
    
    print("\nAll conversions complete!")

if __name__ == "__main__":
    main()
