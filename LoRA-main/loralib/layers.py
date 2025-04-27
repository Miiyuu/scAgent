#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from torch.nn import ModuleDict


class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

# class MergedLinear(nn.Linear, LoRALayer):
#     # LoRA implemented in a dense layer
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         r: int = 0,
#         lora_alpha: int = 1,
#         lora_dropout: float = 0.,
#         enable_lora: List[bool] = [False],
#         fan_in_fan_out: bool = False,
#         merge_weights: bool = True,
#         num_lora:int=1,
#         **kwargs
#     ):
#         self.num_lora=num_lora
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
#                            merge_weights=merge_weights)
#         assert out_features % len(enable_lora) == 0, \
#             'The length of enable_lora must divide out_features'
#         self.enable_lora = enable_lora
#         self.fan_in_fan_out = fan_in_fan_out
#         # Actual trainable parameters
#         if r > 0 and any(enable_lora):
#             # self.lora_A = nn.Parameter(
#             #     self.weight.new_zeros((r * sum(enable_lora), in_features)))
#             # self.lora_B = nn.Parameter(
#             #     self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
#             # ) # weights for Conv1D with groups=sum(enable_lora)

#             # 初始化lora参数
#             for id in range(self.num_lora):
#                 setattr(self, f'lora_A_{id}', nn.Parameter(
#                     self.weight.new_zeros((r * sum(enable_lora), in_features))))
#                 setattr(self, f'lora_B_{id}', nn.Parameter(
#                     self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))))

#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#             # Compute the indices
#             self.lora_ind = self.weight.new_zeros(
#                 (out_features, ), dtype=torch.bool
#             ).view(len(enable_lora), -1)
#             self.lora_ind[enable_lora, :] = True
#             self.lora_ind = self.lora_ind.view(-1)

#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.T

#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         # if hasattr(self, 'lora_A'):
#         #     # initialize A the same way as the default for nn.Linear and B to zero
#         #     nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#         #     nn.init.zeros_(self.lora_B)
#         for id in range(self.num_lora):
#             # 假设每个id对应一个lora_A和lora_B参数
#             lora_A_name = f'lora_A_{id}'
#             lora_B_name = f'lora_B_{id}'
    
#             # 检查并初始化lora_A和lora_B参数
#             if hasattr(self, lora_A_name):
#                 nn.init.kaiming_uniform_(getattr(self, lora_A_name), a=math.sqrt(5))
#             if hasattr(self, lora_B_name):
#                 nn.init.zeros_(getattr(self, lora_B_name))

#     def zero_pad(self, x):
#         x = x.transpose(0, 1)
#         result = x.new_zeros((*x.shape[:-1], self.out_features))
#         result = result.view(-1, self.out_features)
#         result[:, self.lora_ind] = x.reshape(
#             -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
#         )
#         return result.view((*x.shape[:-1], self.out_features)).transpose(0, 1)

#     def train(self, mode: bool = True):
#         def T(w):
#             return w.T if self.fan_in_fan_out else w
#         nn.Linear.train(self, mode)

#         # if train(True) -> unmerge unless we already have them unmerged
#         # if train(False) -> merge unless we already have them merged
#         should = self.merged if mode else not self.merged

#         if self.merge_weights and should:
#             if self.r > 0 and any(self.enable_lora):

#                 for i in range(self.num_lora):
#                     lora_A_name = f'lora_A_{i}'
#                     lora_B_name = f'lora_B_{i}'
#                     delta_w = F.conv1d(
#                         getattr(self, lora_A_name).data.unsqueeze(0),
#                         getattr(self, lora_B_name).data.unsqueeze(-1),
#                         groups=sum(self.enable_lora)
#                     ).squeeze(0)
#                     # -1: W = W - delta_W (unmerge), +1: W = W + delta_W (merge)
#                     sign = -1 if mode else 1
#                     self.weight.data += 1/self.num_lora * sign * self.zero_pad(T(delta_w * self.scaling))
#             self.merged = not mode

#     def forward(self, x: torch.Tensor):
#         def T(w):
#             return w.T if self.fan_in_fan_out else w
#         if self.merged:
#             return F.linear(x, T(self.weight), bias=self.bias)
#         else:
#             print(x.shape)
#             result = F.linear(x, T(self.weight), bias=self.bias)
#             if self.r > 0:
#                 for i in range(self.num_lora):
#                     lora_A_name = f'lora_A_{i}'
#                     lora_B_name = f'lora_B_{i}'

#                     print(getattr(self, lora_A_name).shape)
#                     after_A = F.linear(self.lora_dropout(x), getattr(self, lora_A_name))
#                     print(after_A.shape)
#                     print(getattr(self, lora_B_name).shape)
#                     after_B = F.conv1d(
#                         after_A.transpose(-2, -1),
#                         getattr(self, lora_B_name).unsqueeze(-1),
#                         groups=sum(self.enable_lora)
#                     ).transpose(-2, -1)
#                     print(after_B.shape)

#                     if i == 0:
#                         after_B_sum = after_B
#                     else:
#                         after_B_sum += after_B
                        
#                 result += self.zero_pad(after_B_sum / self.num_lora) * self.scaling
#             return result

class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

class MergedLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        enable_lora: List[bool] = [False],
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora

        self.lora_num = lora_nums
        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight
        
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
            for i in range(self.lora_num):
                setattr(self, f'lora_A_{i}', nn.Parameter(
                    self.weight.new_zeros((r * sum(enable_lora), in_features))))
                setattr(self, f'lora_B_{i}', nn.Parameter(
                    self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))))

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        
        if hasattr(self, "lora_A_0"):
            for i in range(self.lora_num):
                nn.init.kaiming_uniform_(getattr(self, f"lora_A_{i}"), a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"lora_B_{i}"))

            nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def zero_pad(self, x):
        x = x.transpose(0, 1)
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features)).transpose(0, 1)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_route.train(mode)

    def eval(self):
        nn.Linear.eval(self)
        self.lora_route.eval()
  
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x: torch.Tensor, task_types=None):

        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            raise ImportError(":(")
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
            if self.r > 0:
                route_weight = nn.functional.softmax(self.lora_route(x), dim=-1, dtype=torch.float32).to(result.dtype)

                for i in range(self.lora_num):
                    lora_A_name = f'lora_A_{i}'
                    lora_B_name = f'lora_B_{i}'

                    # after_A = F.linear(self.lora_dropout(x), getattr(self, lora_A_name))
                    # after_B = F.conv1d(
                    #     F.linear(self.lora_dropout(x), getattr(self, lora_A_name)).transpose(-2, -1),
                    #     getattr(self, lora_B_name).unsqueeze(-1),
                    #     groups=sum(self.enable_lora)
                    # ).transpose(-2, -1)
                    result += self.zero_pad(torch.unsqueeze(route_weight[:,:,i], -1) * F.conv1d(
                        F.linear(self.lora_dropout(x), getattr(self, lora_A_name)).transpose(-2, -1),
                        getattr(self, lora_B_name).unsqueeze(-1),
                        groups=sum(self.enable_lora)
                    ).transpose(-2, -1) * self.scaling)

        return result


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)

# TODO: Change model to swtich among different LoRAs
class MoELoRA(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int,
            lora_alpha: int,
            lora_num: int,
            lora_dropout: float,
            enable_lora: List[bool],
            lora_ind
    ):
        super(MoELoRA, self).__init__()

        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_num = lora_num
        self.enable_lora = enable_lora

        self.in_features = in_features
        self.out_features = out_features

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
        for i in range(self.lora_num):
            setattr(self, f'lora_A_{i}', nn.Parameter(
                torch.zeros((r * sum(enable_lora), in_features))))
            setattr(self, f'lora_B_{i}', nn.Parameter(
                torch.zeros((out_features // len(enable_lora) * sum(enable_lora), r))))

        self.scaling = self.lora_alpha / self.r
        # Freezing the pre-trained weight matrix
        self.lora_ind = lora_ind

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.lora_num):
            nn.init.kaiming_uniform_(getattr(self, f"lora_A_{i}"), a=math.sqrt(5))
            nn.init.zeros_(getattr(self, f"lora_B_{i}"))

        nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        route_weight = nn.functional.softmax(self.lora_route(x), dim=-1)
        result = 0
        for i in range(self.lora_num):
            lora_A = getattr(self, f'lora_A_{i}')
            lora_B = getattr(self, f'lora_B_{i}')
            result += self.zero_pad(torch.unsqueeze(route_weight[:, :, i], -1) * F.conv1d(
                F.linear(self.lora_dropout(x), lora_A).transpose(-2, -1),
                lora_B.unsqueeze(-1),
                groups=sum(self.enable_lora)
            ).transpose(-2, -1) * self.scaling)
        return result

    def zero_pad(self, x):
        x = x.transpose(0, 1)
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features)).transpose(0, 1)


class MultiMergedLinear(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_nums: int = 2,
            lora_keys: List[str] = None,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            enable_lora: List[bool] = [False],
            **kwargs
    ):
        super(MultiMergedLinear, self).__init__(in_features, out_features, **kwargs)

        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'

        if lora_keys is None:
            raise ValueError("lora_keys must be provided and cannot be None.")

        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self.lora_keys = lora_keys

        self.weight.requires_grad = False
        # Compute the indices
        self.lora_ind = self.weight.new_zeros(
            (out_features,), dtype=torch.bool
        ).view(len(enable_lora), -1)
        self.lora_ind[enable_lora, :] = True
        self.lora_ind = self.lora_ind.view(-1)

        # Define MoELoRA modules in a ModuleDict
        self.moelora_modules = ModuleDict({
            key: MoELoRA(
                in_features, out_features, r, lora_alpha, lora_nums, lora_dropout, enable_lora, lora_ind=self.lora_ind
            ) for key in lora_keys
        })

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        super(MultiMergedLinear, self).reset_parameters()

    def zero_pad(self, x):
        x = x.transpose(0, 1)
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features)).transpose(0, 1)

    def forward(self, x: torch.Tensor, lora_key: str):
        result = F.linear(x, self.weight if not self.fan_in_fan_out else self.weight.T, bias=self.bias)

        if lora_key is not None and lora_key in self.moelora_modules:
            result += self.moelora_modules[lora_key](x)

        return result

