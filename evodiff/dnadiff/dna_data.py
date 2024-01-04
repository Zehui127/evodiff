from torch.utils.data import Dataset, random_split
import pandas as pd
from torch import Generator


class EPDDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        self.labels = pd.read_csv(self.data_path)["Sequence"].values

        print(f"data loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{len(self.labels)}")

    def __getitem__(self, index):
        label = self.labels[index]
        return (label,)


    def __len__(self):
        return len(self.labels)

def setup_epd_dataset(data_path,train_prop,valid_prop,seed=50):
    generator = Generator().manual_seed(seed)
    dataset = EPDDataset(data_path)
    train_size = int(train_prop * len(dataset))
    val_size = int(valid_prop * len(dataset))
    test_size = len(dataset)-train_size-val_size
    (
        train_dataset,
        test_dataset,
        val_dataset,
    ) =  random_split(dataset = dataset, lengths = [train_size, test_size, val_size],generator=generator)
    return train_dataset, test_dataset, val_dataset
