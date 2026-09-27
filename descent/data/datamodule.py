"""Lightning data module: split files and data loaders."""

import os
from copy import deepcopy
from typing import Optional

from easydict import EasyDict
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from descent.utils import data_utils as D
from descent.utils import pylogger

log = pylogger.get_pylogger(__name__)


class DataModule(LightningDataModule):
    """Builds the train/val/test split files of the selected airports and their data loaders."""

    def __init__(self, dataset: Dataset, extra_params: EasyDict):
        """Stores the dataset template and the loader and split parameters.

        Args:
            dataset: Dataset object; copied for each split.
            extra_params: Loader and split parameters (see configs/data/default.yaml).
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.dataset = dataset
        self.eparams = extra_params
        self.data_prep = self.eparams.data_prep
        self.supported_airports = self.eparams.supported_airports

        suffix = f"{self.data_prep.split_type}_{self.data_prep.exp_suffix}"
        self.split_path = {
            split: f"{self.data_prep.out_split_dir}/{split}_{suffix}.txt"
            for split in ["train", "val", "test"]
        }

    def prepare_data(self):
        """Writes the split lists of the run to its output directory.

        Seen airports contribute to train/val/test, unseen airports only to test. Blacklisted scene
        directories are removed.
        """
        seen_airports = self.data_prep.seen_airports
        unseen_airports = self.data_prep.unseen_airports
        assert len(seen_airports) > 0, f"Train airport list is empty: {seen_airports}"
        assert len(seen_airports) == len(set(seen_airports)), f"Duplicate airports {seen_airports}"
        assert all(
            airport in self.supported_airports for airport in seen_airports + unseen_airports
        ), f"Unsupported airport. Supported ones are {self.supported_airports}"
        assert all(airport not in seen_airports for airport in unseen_airports), (
            f"'Unseen' airports {unseen_airports} overlap with 'seen' airports {seen_airports}"
        )

        def read_split(split, airport):
            """Reads the shard list of one airport and split."""
            with open(
                f"{self.data_prep.splits_dir}/{split}_splits/{airport}_{self.data_prep.split_type}.txt"
            ) as fp:
                airport_list = [line.rstrip() for line in fp]
            return airport_list[: int(len(airport_list) * self.data_prep.to_process)]

        split_lists = {"train": [], "val": [], "test": []}
        for airport in seen_airports:
            for split in split_lists:
                split_lists[split] += read_split(split, airport)
        for airport in unseen_airports:
            split_lists["test"] += read_split("test", airport)

        blacklist = D.flatten_blacklist(D.load_blacklist(self.data_prep, self.supported_airports))
        os.makedirs(self.data_prep.out_split_dir, exist_ok=True)
        for split, file_list in split_lists.items():
            with open(self.split_path[split], "w") as fp:
                fp.write("\n".join(D.remove_blacklisted(blacklist, file_list)))

    def setup(self, stage: Optional[str] = None):
        """Builds the datasets of the stage (fit: train/val, test: test) once."""
        splits = ["test"] if stage == "test" else ["train", "val"]
        for split in splits:
            if getattr(self, f"data_{split}") is not None:
                continue
            data = deepcopy(self.dataset)
            data.set_split_list(self.split_path[split])
            data.prepare_data(split=split)
            setattr(self, f"data_{split}", data)

    def get_dataloader(self, data: Dataset, shuffle: bool):
        """Data loader with the configured batch size and worker settings."""
        num_workers = self.eparams.num_workers
        return DataLoader(
            dataset=data,
            batch_size=self.eparams.batch_size,
            num_workers=num_workers,
            pin_memory=self.eparams.pin_memory,
            shuffle=shuffle,
            collate_fn=self.dataset.collate_batch,
            persistent_workers=self.eparams.persistent_workers and num_workers > 0,
            prefetch_factor=self.eparams.prefetch_factor if num_workers > 0 else None,
            # The last incomplete batch is dropped, also for the reported test numbers.
            drop_last=True,
        )

    def train_dataloader(self):
        """Shuffled training loader."""
        return self.get_dataloader(self.data_train, shuffle=True)

    def val_dataloader(self):
        """Validation loader."""
        return self.get_dataloader(self.data_val, shuffle=False)

    def test_dataloader(self):
        """Test loader."""
        return self.get_dataloader(self.data_test, shuffle=False)
