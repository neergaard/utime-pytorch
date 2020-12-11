import math
import os
import random
import warnings
from itertools import compress
from pprint import pprint

import numpy as np
import torch
import pytorch_lightning as pl
from h5py import File
from joblib import delayed
from joblib import Memory
from joblib import Parallel
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from tqdm import tqdm

import utils

# try:
#     from utils.parallel_bar import ParallelExecutor
#     from utils.h5_utils import load_h5_data
# except ImportError:
#     from utils.h5_utils import load_h5_data
#     from utils.parallel_bar import ParallelExecutor

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
SCALERS = {"robust": preprocessing.RobustScaler, "standard": preprocessing.StandardScaler}


def get_class_sequence_idx(hypnogram, selected_sequences):
    d = {
        "w": [idx for idx, hyp in enumerate(hypnogram) if (hyp == 0).any() and idx in selected_sequences],
        "n1": [idx for idx, hyp in enumerate(hypnogram) if (hyp == 1).any() and idx in selected_sequences],
        "n2": [idx for idx, hyp in enumerate(hypnogram) if (hyp == 2).any() and idx in selected_sequences],
        "n3": [idx for idx, hyp in enumerate(hypnogram) if (hyp == 3).any() and idx in selected_sequences],
        "r": [idx for idx, hyp in enumerate(hypnogram) if (hyp == 4).any() and idx in selected_sequences],
    }
    return d


def get_unknown_stage(onehot_hypnogram):
    return onehot_hypnogram.sum(axis=1) == 0


def get_stable_stage(hypnogram, stage, adjustment=30):
    """
    Args:
        hypnogram (array_like): hypnogram with sleep stage labels
        stage (int): sleep stage label ({'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4})
        adjusted (int, optional): Controls the amount of bracketing surrounding a period of stable sleep.
        E.g. if adjustment=30, each period of stable sleep needs to be bracketed by 30 s.
    Returns:
        stable_periods: a list of range objects where each range describes a period of stable sleep stage.
    """
    from itertools import groupby
    from operator import itemgetter

    list_of_periods = []
    for k, g in groupby(enumerate(np.where(hypnogram == stage)[0]), lambda x: x[0] - x[1]):
        list_of_periods.append(list(map(itemgetter(1), g)))
    stable_periods = [range(period[0] + adjustment, period[-1] + 1 - adjustment) for period in list_of_periods]
    # Some periods are empty and need to be removed
    stable_periods = list(filter(lambda x: list(x), stable_periods))

    return stable_periods


def get_stable_sleep_periods(hypnogram, adjustment=30):
    """Get periods of stable sleep uninterrupted by transitions

    Args:
        hypnogram (array-like): hypnogram vector or array with sleep stage labels
        adjustment (int): parameter controlling the amount of shift when selecting periods of stable sleep. E.g.
        if adjustment = 30, each period of stable sleep needs to be bracketed by 30 s of the same sleep stage.
    """
    hypnogram_shape = hypnogram.shape
    hypnogram = hypnogram.reshape(np.prod(hypnogram_shape))
    stable_periods = []
    stable_periods_bool = np.full(np.prod(hypnogram_shape), False)
    for stage in [0, 1, 2, 3, 4]:
        stable_periods.append(get_stable_stage(hypnogram, stage, adjustment))
        for period in stable_periods[-1]:
            stable_periods_bool[period] = True
    stable_periods_bool = stable_periods_bool.reshape(hypnogram_shape)

    return stable_periods_bool, stable_periods


def initialize_record(filename, scaling=None, overlap=True, adjustment=30):

    if scaling in SCALERS.keys():
        scaler = SCALERS[scaling]()
    else:
        scaler = None

    with File(filename, "r") as h5:
        # if "A2081_5 194244.h5" in filename:
        # print("Hej")
        N, C, T = h5["M"].shape
        hypnogram = h5["L"][:, :, ::30]
        hyp_shape = hypnogram.shape
        sequences_in_file = N

        if scaler:
            scaler.fit(h5["M"][:].transpose(1, 0, 2).reshape((C, N * T)).T)

        # Remember that the output array from the H5 has 50 % overlap between segments.
        # Use the following to split into even and odd
        if overlap:
            hyp_even = hypnogram[0::2]
            hyp_odd = hypnogram[1::2]
            if adjustment > 0:
                stable_sleep = np.full([v for idx, v in enumerate(hyp_shape) if idx != 1], False)
                stable_sleep[0::2] = get_stable_sleep_periods(hyp_even.argmax(axis=1), adjustment)[0]
                stable_sleep[1::2] = get_stable_sleep_periods(hyp_odd.argmax(axis=1), adjustment)[0]
            else:
                stable_sleep = np.full([v for idx, v in enumerate(hyp_shape) if idx != 1], True)
        else:
            if adjustment > 0:
                stable_sleep = get_stable_sleep_periods(h5["L"][:].argmax(axis=1), adjustment)[0]
            else:
                stable_sleep = np.full([v for idx, v in enumerate(hyp_shape) if idx != 1], True)

        # Remove unknown stage
        unknown_stage = get_unknown_stage(hypnogram)
        stable_sleep[unknown_stage] = False

        # Get bin counts
        if overlap:
            # hyp = h5["L"][::2].argmax(axis=1)[~get_unknown_stage(h5["L"][::2])][::30]
            hyp = hyp_even.argmax(axis=1)[~unknown_stage[::2] & stable_sleep[::2]]
        else:
            # hyp = h5["L"][:].argmax(axis=1)[~get_unknown_stage(h5["L"][:])][::30]
            hyp = hypnogram.argmax(axis=1)[~unknown_stage & stable_sleep]
        bin_counts = np.bincount(hyp, minlength=C)

    return hypnogram.argmax(1), sequences_in_file, scaler, stable_sleep, bin_counts


def load_psg_h5_data(filename, scaling=None):
    scaler = None

    if scaling:
        scaler = SCALERS[scaling]()

    with File(filename, "r") as h5:
        N, C, T = h5["M"].shape
        sequences_in_file = N

        if scaling:
            scaler.fit(h5["M"][:].transpose(1, 0, 2).reshape((C, N * T)).T)

    return sequences_in_file, scaler
    # X = h5['M'][:].astype('float32')
    # y = h5['L'][:].astype('float32')

    # sequences_in_file = X.shape[0]

    # return X, y, sequences_in_file


class SscWscPsgDataset(Dataset):
    def __init__(
        self,
        data_dir=None,
        n_jobs=-1,
        scaling=None,
        adjustment=30,
        n_records=None,
        overlap=True,
        beta=0.999,
        cv=None,
        cv_idx=None,
        eval_ratio=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.n_jobs = n_jobs
        self.scaling = scaling
        self.adjustment = adjustment
        self.n_records = n_records
        self.overlap = overlap
        self.beta = beta
        self.cv = cv
        self.cv_idx = cv_idx
        self.eval_ratio = eval_ratio

        self.records = sorted(os.listdir(self.data_dir))[: self.n_records]
        # self.data = {r: [] for r in self.records}
        self.index_to_record = []
        self.index_to_record_class = {"w": [], "n1": [], "n2": [], "n3": [], "r": []}
        # self.record_to_index = []
        self.record_indices = {r: None for r in self.records}
        self.scalers = {r: None for r in self.records}
        self.stable_sleep = {r: None for r in self.records}
        self.record_class_indices = {r: None for r in self.records}
        # self.batch_indices = []
        # self.current_record_idx = -1
        # self.current_record = None
        # self.loaded_record = None
        # self.current_position = None
        # data = load_psg_h5_data(os.path.join(self.data_dir, self.records[0]))
        self.cache_dir = "data/.cache"
        memory = Memory(self.cache_dir, mmap_mode="r", verbose=0)
        get_data = memory.cache(initialize_record)

        # Get information about the data
        print(f"Loading mmap data using {n_jobs} workers:")
        data = utils.ParallelExecutor(n_jobs=n_jobs, prefer="threads")(total=len(self.records))(
            delayed(get_data)(
                filename=os.path.join(self.data_dir, record),
                scaling=self.scaling,
                adjustment=self.adjustment,
                overlap=self.overlap,
            )
            for record in self.records
        )
        # for record, d in zip(tqdm(self.records, desc='Processing'), data):
        #     seqs_in_file = d[2]
        #     self.data[record] = {'data': d[0], 'target': d[1]}
        self.n_classes = 5
        cum_class_counts = np.zeros(self.n_classes, dtype=np.int64)
        for record, (hypnogram, sequences_in_file, scaler, stable_sleep, class_counts) in zip(
            tqdm(self.records, desc="Processing"), data
        ):
            # Some sequences are all unstable sleep, which interferes with the loss calculations.
            # This selects sequences where at least one epoch is sleep.
            select_sequences = np.where(stable_sleep.squeeze().any(axis=1))[0]
            self.record_indices[record] = select_sequences  # np.arange(sequences_in_file)
            self.record_class_indices[record] = get_class_sequence_idx(hypnogram, select_sequences)
            self.index_to_record.extend(
                [{"record": record, "idx": x} for x in select_sequences]
            )  # range(sequences_in_file)])
            for c in self.index_to_record_class.keys():
                self.index_to_record_class[c].extend(
                    [
                        {
                            "idx": [
                                idx
                                for idx, i2r in enumerate(self.index_to_record)
                                if i2r["idx"] == x and record == i2r["record"]
                            ][0],
                            "record": record,
                            "record_idx": x,
                        }
                        for x in self.record_class_indices[record][c]
                    ]
                )
            self.scalers[record] = scaler
            self.stable_sleep[record] = stable_sleep
            cum_class_counts += class_counts

        # Define the class-balanced weights. We normalize the class counts to the lowest value as the numerator
        # otherwise will dominate the expression
        # (see https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)
        self.cb_weights_norm = (1 - self.beta) / (1 - self.beta ** (cum_class_counts / cum_class_counts.min()))
        self.effective_samples = 1 / self.cb_weights_norm
        self.cb_weights = self.cb_weights_norm * self.n_classes / self.cb_weights_norm.sum()
        print("")
        print(f"Class counts: {cum_class_counts}")
        print(f"Beta: {self.beta}")
        print(f"CB weights norm: {self.cb_weights_norm}")
        print(f"Effective samples: {self.effective_samples}")
        print(f"CB weights: {self.cb_weights}")
        print("")
        print("Finished loading data")

    def shuffle_records(self):
        random.shuffle(self.records)

    def split_data(self):
        n_records = len(self.records)
        self.shuffle_records()
        # if self.cv is None:
        n_eval = int(n_records * self.eval_ratio)
        n_train = n_records - n_eval
        train_idx = np.arange(n_eval, n_records)
        eval_idx = np.arange(0, n_eval)
        # train_data = SscWscPsgSubset(self, np.arange(n_eval, n_records), name="Train")
        # eval_data = SscWscPsgSubset(self, np.arange(0, n_eval), name="Validation")
        # else:
        if self.cv:
            # from sklearn.model_selection import KFold, StratifiedKFold

            # kf = KFold(n_splits=np.abs(self.cv))
            ssc_idx = ["SSC" in s for s in np.array(self.records)[train_idx]]
            kf = sklearn.model_selection.StratifiedKFold(n_splits=np.abs(self.cv))
            if self.cv > 0:
                # train_idx, eval_idx = list(kf.split(np.arange(n_records)))[self.cv_idx]
                _, train_idx = list(kf.split(train_idx, ssc_idx))[self.cv_idx]
            else:
                # eval_idx, train_idx = list(kf.split(np.arange(n_records)))[self.cv_idx]
                # train_idx, _ = list(kf.split(np.arange(n_records)))[self.cv_idx]
                train_idx, _ = list(kf.split(train_idx, ssc_idx))[self.cv_idx]
            print("\n")
            print(f"Running {np.abs(self.cv)}-fold cross-validation procedure.")
            print(f"Current split: {self.cv_idx}")
            print(f"Eval record indices: {eval_idx}")
            print(f"Train record indices: {train_idx}")
            print(f"Number of train/eval records: {len(train_idx)}/{len(eval_idx)}")
            print("\n")
        train_data = SscWscPsgSubset(self, train_idx, name="Train")
        eval_data = SscWscPsgSubset(self, eval_idx, name="Validation")

        return train_data, eval_data

    def __len__(self):
        # if isinstance(self.index_to_record, dict):
        #     return sum([len(v) for v in self.index_to_record.values()])
        # else:
        return len(self.index_to_record)

    def __getitem__(self, idx):

        try:
            # Grab data
            current_record = self.index_to_record[idx]["record"]
            current_sequence = self.index_to_record[idx]["idx"]
            scaler = self.scalers[current_record]
            stable_sleep = np.array(self.stable_sleep[current_record][current_sequence]).squeeze()

            # Grab data
            with File(os.path.join(self.data_dir, current_record), "r") as f:
                x = f["M"][current_sequence].astype("float32")
                t = f["L"][current_sequence, :, ::30].astype("uint8").squeeze()
            # x = self.data[current_record]['data'][current_sequence]
            # t = self.data[current_record]['target'][current_sequence]

        except IndexError:
            print("Bug")

        if np.isnan(x).any():
            print("NaNs detected!")

        if scaler:
            x = scaler.transform(x.T).T  # (n_channels, n_samples)

        return x, t, current_record, current_sequence, stable_sleep

    def __str__(self):
        s = f"""
======================================
SSC-WSC PSG Dataset Dataset
--------------------------------------
Data directory: {self.data_dir}
Number of records: {len(self.records)}
======================================
"""

        return s


# def collate_fn(batch):
#
#    x, t, w = (
#        np.stack([b[0] for b in batch]),
#        np.stack([b[1] for b in batch]),
#        np.stack([b[2] for b in batch])
#    )
#
#    return torch.FloatTensor(x), torch.IntTensor(t), torch.FloatTensor(w)


def collate_fn(batch):

    X, y = map(torch.FloatTensor, zip(*batch))

    return X, y
    # return torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(w)


class SscWscPsgSubset(Dataset):
    def __init__(self, dataset, record_indices, name="Train"):
        self.dataset = dataset
        self.record_indices = record_indices
        self.name = name
        self.records = [self.dataset.records[idx] for idx in self.record_indices]
        if self.name.lower() == "train":
            self.sequence_indices = self._get_subset_class_indices()
        else:
            self.sequence_indices = self._get_subset_indices()
        # self.sequence_indices = [idx for idx, v in enumerate(self.dataset.index_to_record) for r in self.records if v['record'] == r]# [idx for idx, v in enumerate(self.dataset.index_to_record) for r in self.records if v['record'] == r]
        # print("BAAAD BOIII")

    def _get_subset_class_indices(self):
        out = {k: None for k in self.dataset.index_to_record_class.keys()}
        for c in out.keys():
            t = list(map(lambda x: x["record"] in self.records, self.dataset.index_to_record_class[c]))
            out[c] = list(compress(range(len(t)), t))
        return out

    def _get_subset_indices(self):
        t = list(map(lambda x: x["record"] in self.records, self.dataset.index_to_record))
        return list(compress(range(len(t)), t))

    def __getitem__(self, idx):
        if isinstance(self.sequence_indices, dict):
            class_choice = np.random.choice(list(self.sequence_indices.keys()))
            sequence_choice = np.random.choice(self.sequence_indices[class_choice])
            return self.dataset[self.dataset.index_to_record_class[class_choice][sequence_choice]["idx"]]
        else:
            return self.dataset[self.sequence_indices[idx]]

    def __len__(self):
        if isinstance(self.sequence_indices, dict):
            return sum([len(v) for v in self.sequence_indices.values()])
        else:
            return len(self.sequence_indices)

    def __str__(self):
        s = f"""
======================================
SSC-WSC PSG Dataset - {self.name} partition
--------------------------------------
Data directory: {self.dataset.data_dir}
Number of records: {len(self.record_indices)}
First ten records: {self.records[:10]}
======================================
"""

        return s


class SscWscDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        cv=None,
        cv_idx=None,
        data_dir=None,
        eval_ratio=0.1,
        n_workers=0,
        n_jobs=-1,
        n_records=None,
        scaling="robust",
        adjustment=None,
        **kwargs,
    ):
        super().__init__()
        self.adjustment = adjustment
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.cv = cv
        self.cv_idx = cv_idx
        self.eval_ratio = eval_ratio
        self.n_jobs = n_jobs
        self.n_records = n_records
        self.n_workers = n_workers
        self.scaling = scaling
        self.data = {"train": os.path.join(data_dir, "train"), "test": os.path.join(data_dir, "test")}
        self.dataset_params = dict(
            # data_dir=self.data_dir,
            cv=self.cv,
            cv_idx=self.cv_idx,
            eval_ratio=self.eval_ratio,
            n_jobs=self.n_jobs,
            n_records=self.n_records,
            scaling=self.scaling,
            adjustment=self.adjustment,
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            dataset = SscWscPsgDataset(data_dir=self.data["train"], **self.dataset_params)
            self.train, self.eval = dataset.split_data()
        elif stage == "test":
            self.test = SscWscPsgDataset(data_dir=self.data["test"], overlap=False, **self.dataset_params)

    def train_dataloader(self):
        """Return training dataloader."""
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True,
            # drop_last=True,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return torch.utils.data.DataLoader(
            self.eval,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
            # drop_last=True,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return torch.utils.data.DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True,
        )

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        from argparse import ArgumentParser

        # DATASET specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        dataset_group = parser.add_argument_group("dataset")
        dataset_group.add_argument("--data_dir", default="data/ssc_wsc/5min", type=str)
        dataset_group.add_argument("--eval_ratio", default=0.1, type=float)
        dataset_group.add_argument("--n_jobs", default=-1, type=int)
        dataset_group.add_argument("--n_records", default=None, type=int)
        dataset_group.add_argument("--scaling", default="robust", type=str)
        dataset_group.add_argument("--adjustment", default=0, type=int)
        dataset_group.add_argument("--cv", default=None, type=int)
        dataset_group.add_argument("--cv_idx", default=None, type=int)

        # DATALOADER specific
        dataloader_group = parser.add_argument_group("dataloader")
        dataloader_group.add_argument("--batch_size", default=12, type=int)
        dataloader_group.add_argument("--n_workers", default=20, type=int)

        return parser


if __name__ == "__main__":

    np.random.seed(42)
    random.seed(42)

    # dataset_params = dict(data_dir="./data/raw/individual_encodings", n_jobs=1, scaling="robust", n_records=10)
    # dataset_params = dict(data_dir="./data/ssc_wsc/raw/5min", n_jobs=1, scaling="robust", n_records=10,)
    # dataset = SscWscPsgDataset(**dataset_params)
    dm_params = dict(
        batch_size=32,
        n_workers=0,
        data_dir="./data/ssc_wsc/5min",
        eval_ratio=0.1,
        n_records=None,
        scaling="robust",
        adjustment=15,
        n_jobs=-1,
    )
    dm = SscWscDataModule(**dm_params)
    dm.setup("fit")
    print(dm.train)
    pbar = tqdm(dm.train_dataloader())
    for idx, batch in enumerate(pbar):
        if idx == 0:
            print(batch)
    # pbar = tqdm(DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0, pin_memory=True))
    # for idx, (x, t) in enumerate(pbar):
    #     if idx == 0:
    #         print(x.shape)
    # print(eval_data)
    # pbar = tqdm(DataLoader(eval_data, batch_size=32, shuffle=True, num_workers=20, pin_memory=True))
    # for x, t in pbar:
    #     pass
