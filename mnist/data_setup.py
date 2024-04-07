"""
Contains functionality for creating Pytorch DataLoaders for MNIST data.
"""
from pathlib import Path
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

BATCH_SIZE = 32

def create_dataloaders(
    data_dir,
    transform=ToTensor,
    batch_size=BATCH_SIZE
):
    data_path = Path(data_dir)
    mnist_data_path = data_path / "MNIST"
    
    download_mnist = True
    
    if mnist_data_path.is_dir():
        print('MNIST data already exists. Skipping download...')
        download_mnist = False
    else:
        print('MNIST data folder not found. Downloading MNIST....')
        download_mnist = True
    
    train_data = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download_mnist,
        transform=transform(),
        target_transform=None
    )
    
    test_data = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download_mnist,
        transform=transform()
    )

    class_names = train_data.classes

    # Turn datasets into batches (iterable)
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader, class_names
