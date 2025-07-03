from utils.video_utils import extract_frames

extract_frames(
    video_path="data/raw/trekking-02.mp4",
    output_folder="data/frames/trekking02-annotazione",
    every_n_frames=15
)
