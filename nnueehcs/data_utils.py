from torch.utils.data import Dataset
from io import StringIO
import torch
import yaml
import re
import csv
import pandas as pd


class DatasetCommon():
    def __init__(self):
        pass

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (torch.tensor(self.input[idx]),
                torch.tensor(self.output[idx]))

    def input_as_torch_tensor(self):
        return torch.tensor(self.input)

    def output_as_torch_tensor(self):
        return torch.tensor(self.output)


class HDF5Dataset(DatasetCommon, Dataset):
    def __init__(self, path: str, group_name: str, 
                 input_dataset: str, output_dataset: str):
        super().__init__()
        self.path = path
        self.group_name = group_name
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.ipt_dataset, self.opt_dataset = self.get_datasets(path,
                                                               group_name,
                                                               input_dataset,
                                                               output_dataset
                                                               )
        assert len(self.ipt_dataset) == len(self.opt_dataset)
        self.len = len(self.ipt_dataset)

    def get_datasets(self, filename, group_name, ipt_dataset, opt_dataset):
        import h5py
        open_file = h5py.File(filename, 'r')
        group = open_file[group_name]
        ipt_dataset = group[ipt_dataset]
        opt_dataset = group[opt_dataset]
        if (ipt_dataset.shape[0] == 1):
            print(f"WARNING: Found left dimension of 1 in shape {ipt_dataset.shape},"
                  f" assuming this is not necessary and removing it."
                  f" Reshaping to {ipt_dataset.shape[1:]}"
                  )
            ipt_dataset = ipt_dataset[0]
            opt_dataset = opt_dataset[0]
        return ipt_dataset, opt_dataset

    @property
    def input(self):
        return self.ipt_dataset

    @property
    def output(self):
        return self.opt_dataset

    @property
    def shape(self):
        return self.ipt_dataset.shape


class ARFFDataSet(DatasetCommon, Dataset):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.input, self.output = self.read_arff_file(path)
        self.input, self.output = torch.tensor(self.input), torch.tensor(self.output)
        self.len = len(self.input)

    def read_arff_file(self, path):
        from scipy.io import arff
        import pandas as pd
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        return df.iloc[:, :-1].values, df.iloc[:, -1].values

    @property
    def shape(self):
        return self.input.shape


class CharacterDelimitedDataset(DatasetCommon, Dataset):
    def __init__(self, path: str, delimiter: str):
        super().__init__()
        self.path = path
        self.delimiter = delimiter
        self.input, self.output = self.read_file(path, delimiter)
        self.input, self.output = torch.tensor(self.input), torch.tensor(self.output)
        self.len = len(self.input)

    def read_file(self, path, delimiter):
        has_header = self.file_has_header(path, delimiter)
        if has_header:
            df = pd.read_csv(path, delimiter=delimiter)
        else:
            df = pd.read_csv(path, delimiter=delimiter, header=None)
        return df.iloc[:, :-1].values, df.iloc[:, -1].values


    def file_has_header(self, path, sep):
        if isinstance(path, str):
            with open(path, 'r') as file:
                sample_lines = [file.readline() for _ in range(5)]
        else:
            original_position = path.tell()
            path.seek(0)
            sample_lines = [path.readline() for _ in range(5)]
            path.seek(original_position)
    
        processed_lines = []
        for line in sample_lines:
            if sep == r'\s+':
                processed_line = re.sub(r'(?<=\S)\s+(?=\S)', ',', line.rstrip('\n'))
            else:
                processed_line = line.rstrip('\n').replace(sep, ',')
            processed_lines.append(processed_line)
    
        sample = '\n'.join(processed_lines)
    
        sniffer = csv.Sniffer()
        try:
            has_header = sniffer.has_header(sample)
        except csv.Error:
            has_header = False
    
        return has_header


    @property
    def shape(self):
        return self.input.shape


def read_dataset_from_yaml(filename: str, dataset_name: str):
    try:
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
    except TypeError:
        config = yaml.safe_load(filename)

    config = config['datasets']
    dset_details = config[dataset_name]
    if dset_details['format'] == 'hdf5':
        del dset_details['format']
        return HDF5Dataset(**dset_details)
    elif dset_details['format'] == 'arff':
        del dset_details['format']
        return ARFFDataSet(**dset_details)
    elif dset_details['format'] == 'character_delimited':
        del dset_details['format']
        return CharacterDelimitedDataset(**dset_details)
    else:
        raise ValueError(f"Unknown dataset format {dset_details['format']}")
