from torchvision import datasets, transforms
import numpy as np

class CIFAR10WithAlbumentations(datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        
    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
            image = np.transpose(image, [2, 0, 1])
            

        return image, label
        