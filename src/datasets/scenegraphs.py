import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from globals import PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN

import src.utils.paths as paths
import src.utils.io_utils as io

ALLOWED_STRATEGIES = ['standard', 'uniform', 'tempered']

class SceneGraphs(Dataset):
    def __init__(self, problem, sampling_strategy='standard', input_size=224,
                 split='train'):
        
        if isinstance(sampling_strategy, str):
            assert sampling_strategy in ALLOWED_STRATEGIES
            # so sampling_strategy will be a list
            sampling_strategy_list = [sampling_strategy]
        else: 
            assert isinstance(sampling_strategy, list)
            assert set(sampling_strategy).issubset(set(ALLOWED_STRATEGIES))
            sampling_strategy_list = sampling_strategy
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.problem = problem
        self.input_size = input_size
        self.sampling_strategy_list = sampling_strategy_list
        self._load_data()
        self._load_shard(0)
        self.transforms = self._load_transforms()

    def _load_data(self):
        '''
            Loads all shard-independent data
        '''
        rv_info_list, metadata_dict, num_shards_list, shard_size_list, data_len_list = [], {}, [], [], []
        shard_num_to_sampling_strategy = []
        shard_num_to_sampling_shard_num = []

        for sampling_strategy in self.sampling_strategy_list:
            scene_paths = paths.scene_graph_data_paths(self.problem, self.split, sampling_strategy)

            for _, path in scene_paths.items():
                if not os.path.exists(path) and not os.path.exists(path.format(0)):
                    if 'student' not in path:
                        raise RuntimeError("Data path does not exist: [{}]. Generate using preprocessing script".format(path))

            rv_info = io.load_json(scene_paths['rv_info_path'])
            metadata = io.load_json(scene_paths['metadata_path'])
            num_shards = metadata['num_shards']
            shard_size = metadata['shard_size']
            data_len = metadata['data_len']

            rv_info_list.append(rv_info)
            metadata_dict[sampling_strategy] = metadata
            num_shards_list.append(num_shards)
            shard_num_to_sampling_strategy.extend([sampling_strategy]*num_shards)
            shard_num_to_sampling_shard_num.extend(range(num_shards))
            shard_size_list.append(shard_size)
            data_len_list.append(data_len)


        self.rv_info = rv_info_list[0]  # assume all of these are the same
        self.metadata_dict = metadata_dict
        self.num_shards = sum(num_shards_list)  # consider all shards
        self.shard_size_list = shard_size_list
        self.data_len = sum(data_len_list)
        self.shard_num_to_sampling_strategy = shard_num_to_sampling_strategy
        self.shard_num_to_sampling_shard_num = shard_num_to_sampling_shard_num

    def _load_transforms(self):
        return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])        

    def _load_shard(self, _shard_num):
        self.curr_shard = _shard_num

        sampling_strategy = self.shard_num_to_sampling_strategy[_shard_num]
        # we need to recover the actual shard_num for a single sampling strategy
        shard_num = self.shard_num_to_sampling_shard_num[_shard_num]
        scene_paths = paths.scene_graph_data_paths(self.problem, self.split, sampling_strategy)

        
        images_mat = io.loadmat(scene_paths['feat_images_path'].format(shard_num))
        self.raw_rvOrders = io.load_pickle(scene_paths['raw_rvOrder_path'].format(shard_num))

        # Shape: (n x 3 x 64 x 64)
        self.images = images_mat['images']
        self.tiers = images_mat['tiers'].squeeze()

        
        # Shape: (n x num_labels).  1 if label, 0 otherwise
        self.labels = io.load_np(scene_paths['feat_labels_path'].format(shard_num))

        rvOrders_mat = io.loadmat(scene_paths['feat_rvOrder_path'].format(shard_num))
        self.rvOrders = rvOrders_mat['rv_orders']
        self.rvOrders_lengths = rvOrders_mat['lengths'].squeeze()

    def shard_num_idx_from_idx(self, idx):
        if idx > len(self):
            raise ValueError('Index out of bounds. Dataset size {}. Accessing index {}'.format(len(self), idx))

        if len(self.shard_size_list) == 1:
            shard_size = self.shard_size_list[0]
            shard_num =  idx // shard_size
            new_idx = idx % shard_size
        else:
            shard_sizes = np.array(self.shard_size_list)
            shard_cumsum = np.cumsum(shard_sizes)
            shard_num = np.where(idx < shard_cumsum)[0][0]
            if shard_num == 0:
                new_idx = idx
            else:
                new_idx = idx - shard_cumsum[shard_num - 1]

        return shard_num, new_idx

    def __len__(self):
        return self.data_len

    def __getitem__(self, inp_idx):
        shard_num, idx = self.shard_num_idx_from_idx(inp_idx)
        
        if shard_num != self.curr_shard:
            self._load_shard(shard_num)

        return (self.transforms(self.images[idx]),
                self.labels[idx], self.tiers[idx],
                self.rvOrders[idx], self.rvOrders_lengths[idx])

    def get_raw_image(self, inp_idx):
        shard_num, idx = self.shard_num_idx_from_idx(inp_idx)
        if shard_num != self.curr_shard:
            self._load_shard(shard_num)

        return self.images[idx]

    @property
    def labels_size(self):
        return self.labels.shape[1]


# test sizes and data loading
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = SceneGraphs('scenegraph_0k', sampling_strategy='standard')
    y = dataset[0]
    
    # print(dataset.vocab_size)
    z = DataLoader(dataset, batch_size=64, shuffle=True)
    ctr = 0
    for b in z:
        for k in b:
            print(k.size())
        break
