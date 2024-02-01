import numpy as np
from numpy import ndarray
from typing import Tuple, Union, List

from numpytorch import Tensor, nn
from numpytorch.grad_fn import GradFn
from numpytorch.functions import *
def im2col(input_data : Tensor, filter_h : int, filter_w : int, stride : int = 1, pad : int = 0) -> Tensor:
    batch_size, in_channels, in_height, in_width = input_data.shape
    out_h = (in_height + 2 * pad - filter_h) // stride + 1
    out_w = (in_width + 2 * pad - filter_w) // stride + 1

    img = input_data
    col = np.zeros((batch_size, in_channels, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_h * out_w, -1)
    return col

def col2im(col : Tensor, input_shape : Tensor, filter_h : int, filter_w : int, stride : int = 1, pad : int = 0)-> Tensor:
    batch_size, in_channels, in_height, in_width = input_shape
    out_h = (in_height + 2 * pad - filter_h) // stride + 1
    out_w = (in_width + 2 * pad - filter_w) // stride + 1
    col = col.reshape(batch_size, out_h, out_w, in_channels, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((batch_size, in_channels, in_height + 2 * pad + stride - 1, in_width + 2 * pad + stride - 1))
    for y_col in range(filter_h):
        y_max_col = y_col + stride * out_h
        for x_col in range(filter_w):
            x_max_col = x_col + stride * out_w
            img[:, :, y_col:y_max_col:stride, x_col:x_max_col:stride] += col[:, :, y_col, x_col, :, :]

    return img[:, :, pad:in_height + pad, pad:in_width + pad]


"""
Example model.
If you want to see how main.py works (before you finish the assignment),
try running it through this model.
"""
'''
class MNISTClassificationModel(nn.Module):
    def __init__(self) -> None:
        self.seq = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = reshape(x, (x.shape[0], -1))
        logits = self.seq(x)
        return logits
'''


class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight and bias initialization
        self.params = {
            'weight': nn.Parameter.new(out_channels, in_channels, kernel_size, kernel_size),
            'bias': nn.Parameter.new(out_channels),
            'weight_grad': np.zeros((out_channels, in_channels, kernel_size, kernel_size)),
            'bias_grad': np.zeros((out_channels,))
        }

    def forward(self, x: Tensor) -> Tensor:
        batch_size, in_channels, in_height, in_width = x.shape
        x_padded = np.pad(x.arr, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        x_col = im2col(x_padded, self.kernel_size, self.kernel_size, self.stride)
        weight_col = self.params['weight'].arr.reshape(self.out_channels, -1).T
        output_col = np.einsum('ij,jk->ik', x_col, weight_col) + self.params['bias'].arr.reshape(-1, 1).T

        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = output_col.reshape(batch_size, self.out_channels, out_height, out_width)

        return Tensor(output, requires_grad=x.requires_grad, is_leaf=False, grad_fn=Conv2dGradFn(x, self.params['weight'], self.params['bias'], self.stride, self.padding))

# Backward for convolution layer
class Conv2dGradFn(GradFn):
    def __init__(self, x: Tensor, w: Tensor, b: Tensor, stride: int, padding: int) -> None:
        super().__init__(x, w, b)
        self.stride = stride
        self.padding = padding

    def f_d(self, *args: Tensor) -> Tuple[ndarray, ndarray, ndarray]:
        x, w, b, y = args
        assert y.grad is not None

        dx = np.zeros_like(x.arr)
        dw = np.zeros_like(w.arr)
        db = np.zeros_like(b.arr)

        batch_size, in_channels, in_height, in_width = x.shape
        out_channels, _, kernel_size, _ = w.shape
        out_height = (in_height + 2 * self.padding - kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_size) // self.stride + 1

        col = im2col(x.arr, self.kernel_size, self.kernel_size, self.stride, self.padding)
        col_w = w.arr.reshape(out_channels, -1).T

        # Gradient with respect to x
        dx_col = np.dot(y.grad.reshape(batch_size, -1), col_w.T)
        dx = col2im(dx_col, x.shape, kernel_size, kernel_size, self.stride, self.padding)

        # Gradient with respect to w
        dw_col = np.dot(x.arr.reshape(batch_size, -1).T, y.grad.reshape(batch_size, -1))
        dw = dw_col.T.reshape(out_channels, in_channels, kernel_size, kernel_size)

        # Gradient with respect to b
        db = np.sum(y.grad, axis=(0, 2, 3)).reshape(b.shape)

        return (dx, dw, db)



# Max pooling layer
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, in_channels, out_height, out_width))

        for h in range(0, in_height - self.kernel_size + 1, self.stride):
            for w in range(0, in_width - self.kernel_size + 1, self.stride):
                output[:, :, h // self.stride, w // self.stride] = np.max(x.arr[:, :, h:h + self.kernel_size, w:w + self.kernel_size], axis=(2, 3))

        return Tensor(output, requires_grad=x.requires_grad, is_leaf=False, grad_fn=MaxPool2dGradFn(x, self.kernel_size, self.kernel_size, self.stride))

class MaxPool2dGradFn(GradFn):
    def __init__(self, x: Tensor, h: int, w: int, stride: int) -> None:
        super().__init__(x)
        self.h = h
        self.w = w
        self.stride = stride

    def f_d(self, *args: Tensor) -> Tuple[ndarray]:
        x, y = args
        assert y.grad is not None

        batch_size, in_channels, in_height, in_width = x.shape
        dx = np.zeros_like(x.arr)

        for h in range(0, in_height - self.h + 1, self.stride):
            for w in range(0, in_width - self.w + 1, self.stride):
                pool_region = x.arr[:, :, h:h + self.h, w:w + self.w]
                mask = (pool_region == np.max(pool_region, axis=(2, 3), keepdims=True))
                dx[:, :, h:h + self.h, w:w + self.w] += mask * y.grad[:, :, h // self.stride, w // self.stride, np.newaxis, np.newaxis]

        return (dx,)


# class Flatten(nn.Module):
#     def forward(self, x: Tensor) -> Tensor:
#         # Input shape: (batch_size, channels, height, width)
#         return x.view(x.size(0), -1)
#     def custom_view(self, x, new_shape):
#         return np.reshape(x, new_shape)


# Your model

class MNISTClassificationModel(nn.Module):
    def __init__(self) -> None:
        super(MNISTClassificationModel, self).__init__()

        # Convolutional Layers
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # self.conv1_1 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.relu1_1 = nn.ReLU()
        # self.conv1_2 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.relu1_2 = nn.ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # self.conv2_1 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.relu2_1 = nn.ReLU()
        # self.conv2_2 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.relu2_2 = nn.ReLU()
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x: Tensor) -> Tensor: 
        # Input shape: (batch_size, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)


        # Fully Connected Layers
        x = reshape(x, (x.shape[0], -1))  # Reshape
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        # Apply softmax to get probabilities
        return x



