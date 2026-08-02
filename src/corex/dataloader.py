from torch.utils.data import DataLoader

from .dataset import TextDataset


def create_dataloader(
    path: str,
    batch_size: int,
    shuffle: bool = True
):

    dataset = TextDataset(path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
