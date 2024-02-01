import numpy as np
from numpy import ndarray
import math
from typing import Any, Optional, Tuple, Type, Union
from .tensor import *
from .grad_fn import *


def tensor(
    v: Union[Value, ndarray],
    requires_grad: bool = False
) -> Tensor:
    v = ndfy(v).copy()
    return Tensor(v, requires_grad=requires_grad)

def zeros(*args, requires_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.zeros(*args, **kwargs), requires_grad)

def ones(*args, requires_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.ones(*args, **kwargs), requires_grad)

def rand(*args, requires_grad: bool = False, **kwargs) -> Tensor:
    return Tensor(np.random.rand(*args, **kwargs), requires_grad)

def exp(x: Tensor) -> Tensor:
    return math.e ** x

def sigmoid_naive(x: Tensor) -> Tensor:
    return 1 / (1 + exp(-x))

def _new_tensor(x: Tensor, arr: ndarray, grad_fn: Type[GradFn], **kwargs: Any) -> Tensor:
    return Tensor(
        arr,
        requires_grad=x.requires_grad,
        is_leaf=not x.requires_grad,
        grad_fn=grad_fn(x, **kwargs) if x.requires_grad else None
    )

def log(x: Tensor) -> Tensor:
    return _new_tensor(x, np.log(x.arr), LogGradFn)

def sigmoid(x: Tensor) -> Tensor:
    return _new_tensor(x, 1 / (1 + np.exp(-x.arr)), SigmoidGradFn)

def sum(
    x: Tensor,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False
) -> Tensor:
    return _new_tensor(x, np.sum(x.arr, axis, keepdims=keepdims), SumGradFn,
                        axis=axis, keepdims=keepdims)

def mean(x: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    if axis is None:
        return sum(x) / x.size
    else:
        return sum(x, axis, keepdims) / x.shape[axis]

def relu(x: Tensor) -> Tensor:
    # relu 계산이 안돼서 isinstance(x, Tensor) 추가
    if not isinstance(x, Tensor):
        x = tensor(x)
    return _new_tensor(x, np.maximum(0, x.arr), ReLUGradFn)

def tanh(x: Tensor) -> Tensor:
    return _new_tensor(x, np.tanh(x.arr), TanhGradFn)

def reshape(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    return _new_tensor(x, x.arr.reshape(shape), ReshapeGradFn)

def one_hot(x: Tensor, n_label: int) -> Tensor:
    return tensor(np.eye(n_label)[x.arr])

    