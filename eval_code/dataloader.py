import pickle

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


def read_fer2013_data(data_path, dataset_type, batch_size, num_workers):
    # data preprocessing
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image_transforms = {
            'train': transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            'test': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ])
        }

    dataset = ImageFolder(data_path, image_transforms[dataset_type])
    loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size, shuffle=True if 'train' == dataset_type else False,
                    num_workers=num_workers, pin_memory=True
                )
    return loader
