from utils.video_utils import extract_frames

extract_frames(
    video_path="data/raw/bici-01.mp4",
    output_folder="data/frames/bici01-annotazione",
    every_n_frames=15
)
