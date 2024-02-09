import torch
import torch.nn as nn
from torchvision import transforms

tensor_trans = transforms.ToTensor()

loss_func = nn.CrossEntropyLoss()
batch_size = 2

raw_output = torch.randn(3, 5)
input = torch.tensor([  [0.9, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.2],
                        [0.1, 0.9, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.2],
                        [0.1, 0.2, 0.9, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.2],
                        [0.1, 0.2, 0.3, 0.9, 0.2, 0.3, 0.1, 0.2, 0.3, 0.2],
                        [0.1, 0.2, 0.3, 0.1, 0.9, 0.3, 0.1, 0.2, 0.3, 0.2],
                        [0.1, 0.2, 0.3, 0.1, 0.2, 0.9, 0.1, 0.2, 0.3, 0.2],
                        [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.9, 0.2, 0.3, 0.2],
                        [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.9, 0.3, 0.2],
                        [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.9, 0.2],
                        [0.5, 0.6, 0.8, 0.5, 0.6, 0.8, 0.1, 0.5, 0.6, 0.9]], dtype=torch.float32) #2*10
input2 = torch.Tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3, 2],
                      [5, 6, 8, 5, 6, 8, 1, 5, 6, 8]])
target = torch.tensor([0, 8, 2, 2, 4, 1, 6, 5, 8, 0], dtype=torch.long)
loss = loss_func(input, target)
print("loss:{}".format(loss))


index1 = torch.argmax(input, dim=1)
index2 = torch.argmax(target)
tensor_true = index1 == target
num_true = torch.sum(tensor_true)
print(tensor_true)
print(num_true)