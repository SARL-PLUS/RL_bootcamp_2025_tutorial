from typing import Union, Tuple, List
import torch.nn as nn


class DenseNet(nn.Sequential):
    def __init__(
        self,
        in_shape: Union[int, Tuple[int], Tuple[int, int, int]],
        features_dim: int,
        hidden: List[int],
        activation: nn.Module = nn.ReLU,
        ) -> None:
        
        if isinstance(in_shape, int):
            in_shape = (in_shape, )
                
        in_features = [*in_shape, *hidden]
        out_features = [*hidden, features_dim]
        block = []
        
        for blk_idx, (in_feat, out_feat) in enumerate(zip(in_features, out_features)):
            block.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            if blk_idx < len(in_features) - 1:
                block.append(activation())
        
        super().__init__(*block)
        
        
    def forward(self, input):
        return super().forward(input)