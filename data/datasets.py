import glob
import os

import h5py
import numpy as np
import torch
from torchvision import transforms

from utils.image_utils import decompress_image


class BehaviorCloningDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size=(224, 224), crop_scale=1.2):
        self.data_root = data_root
        self.image_size = image_size
        self.episodes = glob.glob(os.path.join(self.data_root, '*.hdf5'))
        self.num_episodes = len(self.episodes)
        assert crop_scale >= 1.0, "crop_scale should be >= 1.0"
        self.image_transforms = transforms.Compose([
            transforms.Resize((int(self.image_size[0] * crop_scale), int(self.image_size[1] * crop_scale))),
            transforms.RandomCrop(self.image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
        ])
        self.max = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) * np.pi
        self.min = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0]) * np.pi


    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):

        episode_file = self.episodes[idx]
        with h5py.File(episode_file, 'r') as hf:
            observations = hf['joint_positions'][:]
            total_steps = observations.shape[0] - 1 
            time_step = np.random.randint(0, total_steps)
            png_image = hf['video'][time_step]

            observation = observations[time_step]
            observation = (observation - self.min) / (self.max - self.min) * 2.0 - 1.0
            action = observations[time_step + 1]
            action = (action - self.min) / (self.max - self.min) * 2.0 - 1.0
            
            image = decompress_image(png_image)
            image = self.image_transforms(image)
            image = torch.permute(image, (1, 2, 0))
            return {'observations': observation, 'actions': action, 
                    'images': image}
        # return {
        #     'observation': self.observations[idx],
        #     'action': self.actions[idx]
        # }

if __name__ == '__main__':
    dataset = BehaviorCloningDataset(data_root='/mnt/pfs/dataset/screwdriver-il-converted')
    print("Dataset loaded successfully.")
    print("Number of episodes:", len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
    batch = next(iter(dataloader))
    print("Batch observations shape:", batch['observations'].shape)
    print("Batch actions shape:", batch['actions'].shape)
    print("Batch images shape:", batch['images'].shape)
    # test load time
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    for i in range(10):
        batch = next(iter(dataloader))
        print(f"Iteration {i}: Batch observations shape:", batch['observations'].shape)