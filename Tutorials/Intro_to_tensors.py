# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:23:59 2021

@author: brand
"""

import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(f"Tensor from data: \n {x_data} \n")

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from numpy array: \n {x_np} \n")

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)   
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device} \n")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')

tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(f"Indexed tensor: \n {tensor} \n")

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"Tensor concatenated by three tensors: \n {t1} \n")

t1 = [[1,2], [1,3]]
t1 = torch.tensor(t1)
print(f"Tensor t1: \n {t1} \n")
print(f"t1.mul(t1): \n {t1.mul(t1)} \n")
print(f"t1.matmul(t1): \n {t1.matmul(t1)} \n")
print(f"t1.T: \n {t1.T} \n")
print(f"t1.add_(5): \n {t1.add_(5)} \n")