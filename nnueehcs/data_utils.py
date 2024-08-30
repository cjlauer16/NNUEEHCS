from torch.utils.data import Dataset
from io import StringIO
import torch
import yaml
import re
import csv
import pandas as pd
import numpy as np

percentile_re = re.compile(r'(?:\[(\d+),\s{0,1}(\d+)\],{0,1})')

class DatasetCommon():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.len = len(self.input)
            self._percentile_partition()
            self._dtype_conversion()
        cls.__init__ = new_init


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.input[idx],
                self.output[idx])

    def input_as_torch_tensor(self):
        return self.input

    def output_as_torch_tensor(self):
        return self.output

    def get_percentiles(self):
        try:
            percs = self.kwargs['percentiles']
            parsed = percentile_re.findall(percs)
            opt_percs = list()
            for p in parsed:
                lower, uper = int(p[0]), int(p[1])
                opt_percs.append((lower, uper))
            return opt_percs
        except KeyError:
            return [(0, 100)]


    def percentile_partition(self, percentiles):
        input_tensor = self.input_as_torch_tensor()
        output_tensor = self.output_as_torch_tensor()

        if len(output_tensor.shape) > 2:
            return input_tensor, output_tensor
        
        unique_percentiles = sorted(set(p for range_pair in percentiles for p in range_pair))
        percentile_values = torch.tensor([
            torch.quantile(output_tensor, q/100) for q in unique_percentiles
        ])
        
        percentile_dict = dict(zip(unique_percentiles, percentile_values))
        
        mask = torch.zeros(len(output_tensor), dtype=torch.bool)
        
        for lower, upper in percentiles:
            lower_value = percentile_dict[lower]
            upper_value = percentile_dict[upper]
            mask |= (output_tensor >= lower_value) & (output_tensor <= upper_value)
        
        partitioned_input = input_tensor[mask]
        partitioned_output = output_tensor[mask]
        
        return partitioned_input, partitioned_output


    def _percentile_partition(self):
        self.input, self.output = self.percentile_partition(self.get_percentiles())

    def _dtype_conversion(self):
        try:
            dt = self.kwargs['dtype']
            self.input = self.input.type(dtype=getattr(torch, dt))
            self.output = self.output.type(dtype=getattr(torch, dt))
        except KeyError:
            pass

    


class HDF5Dataset(DatasetCommon, Dataset):
    def __init__(self, path: str, group_name: str, 
                 input_dataset: str, output_dataset: str,
                 **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.group_name = group_name
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

        self.input, self.output = self.get_datasets(path,
                                                    group_name,
                                                    input_dataset,
                                                    output_dataset
                                                    )
        self.input = torch.tensor(self.input)
        self.output = torch.tensor(self.output)
        assert len(self.input) == len(self.output)

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
            ipt_dataset = torch.tensor(ipt_dataset[0])
            opt_dataset = torch.tensor(opt_dataset[0])
        return ipt_dataset, opt_dataset

    @property
    def shape(self):
        return self.input.shape


class ARFFDataSet(DatasetCommon, Dataset):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.input, self.output = self.read_arff_file(path)
        self.input, self.output = torch.tensor(self.input), torch.tensor(self.output)

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
    def __init__(self, path: str, delimiter: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.delimiter = delimiter
        self.input, self.output = self.read_file(path, delimiter)
        self.input, self.output = torch.tensor(self.input), torch.tensor(self.output)

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
