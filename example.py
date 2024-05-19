
from decord import VideoReader
from vtgllm.processors.video_processor import ToTHWC, ToUint8, load_video
from PIL import Image
import os
import numpy as np

target_dir = 'yourpath/sample_images'

video, _ = load_video(
    video_path='yourpath/QVhighlights/videos/val/NUsG9BgSes0_60.0_210.0.mp4',
    n_frms=8,
    sampling ="uniform", return_msg = True
)

video = video.transpose(0, 1)

for i in range(8):
    print(video[i].numpy())
    im = Image.fromarray(video[i].permute(1,2,0).numpy().astype(np.uint8))
    im.save(os.path.join(target_dir, f'{i}.jpg'))



