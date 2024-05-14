from torch.utils.data import Dataset
import os
from PIL import Image
import torch


class VideoDataset(Dataset):
    def __init__(self, root_dir, phase="train", transform=None, num_frames=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.phase = phase
        self.video, self.label = self._load_videos()

    def _load_videos(self):
        # Load videos and labels
        videos, labels = [], []

        video_folders = os.listdir(os.path.join(self.root_dir, self.phase))

        class_id = 0
        for folder in video_folders:
            all_video_paths = os.listdir(
                os.path.join(self.root_dir, self.phase, folder)
            )
            for video_path in all_video_paths:
                frames = []
                video_folders = os.path.join(
                    self.root_dir, self.phase, folder, video_path
                )
                for frame in os.listdir(video_folders):
                    frames.append(os.path.join(video_folders, frame))
                if self.num_frames:
                    frames = self._uniform_sampling(frames, self.num_frames)
                videos.append(frames)
                labels.append(class_id)
            class_id += 1

        return videos, labels

    def _uniform_sampling(self, frames, num_frames):
        stride = max(1, len(frames) // num_frames)
        sampled = [frames[i] for i in range(0, len(frames), stride)]
        return sampled[:num_frames]

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        video_frames = self.video[idx]
        label = self.label[idx]

        images = []
        for frame_path in video_frames:
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        data = torch.stack(images, dim=0)
        data = data.permute(1, 0, 2, 3)
        return data, label


"""
dataset/
    train/
        Normal/
            video1/
                frame1.jpg
                frame2.jpg
                ...
            video2/
                frame1.jpg
                frame2.jpg
                ...
            ...
        Abnormal/
            video1/
                frame1.jpg
                frame2.jpg
                ...
            video2/
                frame1.jpg
                frame2.jpg
                ...
            ...
    test/
        Normal/
            video1/
                frame1.jpg
                frame2.jpg
                ...
            video2/
                frame1.jpg
                frame2.jpg
                ...
            ...
        Abnormal/
            video1/
                frame1.jpg
                frame2.jpg
                ...
            video2/
                frame1.jpg
                frame2.jpg
                ...
            ...
"""
