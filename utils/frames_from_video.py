from video_utils import extract_frames

extract_frames(
    video_path="data/raw/bici-01.mp4",
    output_folder="data/frames/test_bici01",
    every_n_frames=1                            # Test n=15 per annotazione manuale
)
