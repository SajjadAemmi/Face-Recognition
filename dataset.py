from PIL import Image

import torch
from torchvision import datasets, transforms, models

import config


def gray2rgb(image):
    return image.repeat(3, 1, 1)
    # rgbimg = Image.new("RGB", image.size)
    # rgbimg.paste(image)
    # return rgbimg


def mnist(subset='train'):
    transform = transforms.Compose([transforms.Lambda(gray2rgb),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #                      std=[0.229, 0.224, 0.225]),
                                ])

    if subset == 'train':
        train_dataset = datasets.MNIST('datasets', train=True, download=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

        train_dataloader = torch.utils.data.DataLoader(train_set, num_workers=config.num_workers, shuffle=True, batch_size=config.batch_size)
        val_dataloader = torch.utils.data.DataLoader(val_set, num_workers=config.num_workers, shuffle=True, batch_size=config.batch_size)

        print('train', train_dataloader.dataset)
        print('val', val_dataloader.dataset)
        return train_dataloader, val_dataloader, train_dataset.classes

    elif subset == 'test':
        test_dataset = datasets.MNIST('datasets', train=False, download=True, transform=transform)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=config.num_workers, shuffle=True, batch_size=config.batch_size)

        print('test', test_dataloader.dataset)
        return test_dataloader, test_dataset.classes


def cfar100(subset='train'):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224,224)),
                                    transforms.RandomHorizontalFlip()
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #                      std=[0.229, 0.224, 0.225]),
                                ])

    if subset == 'train':
        train_dataset = datasets.CIFAR100('datasets', train=True, download=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(train_dataset, [40000, 10000])

        train_dataloader = torch.utils.data.DataLoader(train_set, num_workers=config.num_workers, shuffle=True, batch_size=config.batch_size)
        val_dataloader = torch.utils.data.DataLoader(val_set, num_workers=config.num_workers, shuffle=True, batch_size=config.batch_size)

        print('train', train_dataloader.dataset)
        print('val', val_dataloader.dataset)
        return train_dataloader, val_dataloader, train_dataset.classes

    elif subset == 'test':
        test_dataset = datasets.MNIST('datasets', train=False, download=True, transform=transform)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=config.num_workers, shuffle=True, batch_size=config.batch_size)

        print('test', test_dataloader.dataset)
        return test_dataloader, test_dataset.classes
