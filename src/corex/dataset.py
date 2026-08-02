from pathlib import Path

from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, path: str):

        self.path = Path(path)

        with open(
            self.path,
            "r",
            encoding="utf-8"
        ) as file:

            self.lines = [
                line.strip()
                for line in file
                if line.strip()
            ]


    def __len__(self):

        return len(self.lines)


    def __getitem__(self, index):

        return self.lines[index]
