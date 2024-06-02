import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


def check_data(data_list):
    for data in data_list:
        assert not torch.isnan(data.x).any() and not torch.isinf(data.x).any(), "NaNs or Infs found in node features"
        assert not torch.isnan(data.edge_index).any() and not torch.isinf(data.edge_index).any(), "NaNs or Infs found in edge index"
        assert not torch.isnan(data.edge_attr).any() and not torch.isinf(data.edge_attr).any(), "NaNs or Infs found in edge attributes"
        assert not torch.isnan(data.y).any() and not torch.isinf(data.y).any(), "NaNs or Infs found in targets"

class DiskDataset(Dataset):
    def __init__(self, directory):
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path)
        return data
    
    def get_loader(self, batch_size, shuffle=True) -> DataLoader:
        loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle)

        return loader



def model_data(saved_graphs, y):
    data_list = []
    for graph, target in zip(saved_graphs, y):
        edges = graph.edges(data=True)
        node_1_list = [x[0]  for x in edges]
        node_2_list = [x[1]  for x in edges]

        ## Row wise format
        edge_index = torch.tensor([node_1_list, node_2_list], dtype=torch.int64)

        ## Node features
        node_coords = [[attr['x'], attr['y']] for n, attr in graph.nodes(data=True)]
        node_features = torch.tensor(node_coords, dtype=torch.float)

        ## Weights
        weights_list = [x[2]['weight'] for x in edges]
        edge_weight = torch.tensor(weights_list, dtype=torch.float)

        target = torch.tensor([target], dtype=torch.float)  # Example target value

        input_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, y=target)
        data_list.append(input_data)

    check_data(data_list=data_list)

    return data_list
