from typing import List, Union, Tuple, Callable
import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, MessagePassing
import numpy as np


class DenseNet(nn.Sequential):
    def __init__(
        self,
        in_shape: Union[int, Tuple[int], Tuple[int, int, int]],
        out_size: int,
        hidden: List[int],
        activation: nn.Module = nn.ReLU,
        ) -> None:
        
        if isinstance(in_shape, int):
            in_shape = (in_shape, )
                
        in_features = [*in_shape, *hidden]
        out_features = [*hidden, out_size]
        block = []
        
        for blk_idx, (in_feat, out_feat) in enumerate(zip(in_features, out_features)):
            block.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            if blk_idx < len(in_features) - 1:
                block.append(activation())
        
        super().__init__(*block)
        
                    
                            
class NNConvEdgeEncoder(MessagePassing):        
    def __init__(
        self,
        node_shape: Union[int, Tuple[int], Tuple[int, int, int]],
        edge_shape: Union[int, Tuple[int], Tuple[int, int, int]],
        out_channels: int,
        hidden_channels: List[int],
        activation: nn.Module = nn.ReLU(),
        activate_last_layer: bool = True,        
        ) -> None:
        
        super().__init__(aggr='mean')  # Aggregation method can be 'mean', 'sum', or 'max'
        self.net = nn.ModuleList()
        self.activation = activation
        self.activate_last_layer = activate_last_layer
        
        if isinstance(node_shape, int):
            node_shape = (node_shape, )
        
        if isinstance(edge_shape, int):
            edge_shape = (edge_shape, )
        
        
        in_channels = [*node_shape, *hidden_channels]
        out_channels = [*hidden_channels, out_channels]
        
        for in_chan, out_chan in zip(in_channels, out_channels):
            self.net.append(
                NNConv(
                    in_channels=in_chan,
                    out_channels=out_chan,
                    nn=nn.Linear(
                        in_features=np.prod(edge_shape),
                        out_features=in_chan*out_chan
                    )
                )
            )
            
    def forward(self,
                node_space: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_space: torch.Tensor,
                ) -> torch.Tensor:
        
        for i, layer in enumerate(self.net):
            node_space = layer(node_space, edge_indices, edge_space)
            
            if i < len(self.net):
                node_space = self.activation(node_space)
                
        edge_embeddings = (node_space[edge_indices[0]] + node_space[edge_indices[1]]) / 2  
                
        return self.activation(edge_embeddings) if self.activate_last_layer else edge_embeddings
                            
                            
class AugmentedGraphEncoder(nn.Module):
    def __init__(
        self,
        global_shape: Union[int, Tuple[int,int,int]], 
        node_shape: Union[int, Tuple[int,int,int]],
        edge_shape: Union[int, Tuple[int,int,int]],
        features_dim: int,
        global_encoder: Callable[[], nn.Module],
        graph_encoder: Callable[[], nn.Module],
        ):
        super().__init__()
        
        self.global_encoder = global_encoder(
            in_shape=global_shape,
            out_size=features_dim,
        )
        
        self.graph_encoder = graph_encoder(
            node_shape=node_shape,
            edge_shape=edge_shape,
            out_channels=features_dim,
        )
        

    def forward(self, global_space, node_space, edge_indices, edge_space, batch=None):
        
        assert batch is not None or (batch is None and global_space.shape[0]==1),\
            "Batched input requires batch argument to be defined."
        
        if batch == None:
            batch = torch.zeros(len(edge_space), dtype=torch.long)
            
        
        global_space = self.global_encoder(global_space)
        graph_embedding = self.graph_encoder(
            node_space=node_space,
            edge_indices=edge_indices,
            edge_space=edge_space,
            )
        
        global_space = global_space[batch]        
        
        return global_space + graph_embedding

