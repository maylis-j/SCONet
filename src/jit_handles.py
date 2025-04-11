# From https://gist.github.com/jackroos/97b44dd1e603835057f64cf73563a7cd 
# adapted for SCONet
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import typing
from collections import Counter
import numpy as np
from numpy import prod
from itertools import zip_longest
from functools import partial

from numbers import Number
from typing import Any, Callable, List, Union

Handle = Callable[[List[Any], List[Any]], Union[typing.Counter[str], Number]]


def get_shape(val: object) -> typing.List[int]:
    """
    Get the shapes from a jit value object.
    Args:
        val (torch._C.Value): jit value object.
    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():  # pyre-ignore
        r = val.type().sizes()  # pyre-ignore
        if not r:
            r = [1]
        return r
    elif val.type().kind() in ("IntType", "FloatType"):
        return [1]
    else:
        raise ValueError()


def basic_binary_op_flop_jit(inputs, outputs, name):
    input_shapes = [get_shape(v) for v in inputs]
    # print(input_shapes)
    # for broadcasting
    input_shapes = [s[::-1] for s in input_shapes]
    max_shape = np.array(list(zip_longest(*input_shapes, fillvalue=1))).max(1)
    flop = prod(max_shape)
    flop_counter = Counter({name: flop})
    return flop_counter


def elementwise_flop_counter(input_scale: float = 1, output_scale: float = 0) -> Handle:
    """
    Count flops by
        input_tensor.numel() * input_scale + output_tensor.numel() * output_scale

    Args:
        input_scale: scale of the input tensor (first argument)
        output_scale: scale of the output tensor (first element in outputs)
    """

    def elementwise_flop(inputs: List[Any], outputs: List[Any]) -> Number:
        ret = 0
        if input_scale != 0:
            shape = get_shape(inputs[0])
            ret += input_scale * prod(shape)
        if output_scale != 0:
            shape = get_shape(outputs[0])
            ret += prod(shape)
        return ret

    return elementwise_flop


# Custom handles for SCONet
_SUPPORTED_OPS: typing.Dict[str, Handle] = {
    "aten::add": partial(basic_binary_op_flop_jit, name='aten::add'),
    "aten::add_": partial(basic_binary_op_flop_jit, name='aten::add_'),
    "aten::mul": partial(basic_binary_op_flop_jit, name='aten::mul'),
    "aten::sub": partial(basic_binary_op_flop_jit, name='aten::sub'),
    "aten::div": partial(basic_binary_op_flop_jit, name='aten::div'),
    "aten::div_": partial(basic_binary_op_flop_jit, name='aten::div_'),
    "aten::upsample_nearest3d": elementwise_flop_counter(0, 1), # from https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py
    "aten::max_pool3d": elementwise_flop_counter(1, 0),
    "aten::lt": partial(basic_binary_op_flop_jit, name='aten::lt'), #lt = less than element-wise
    # "torch_scatter::scatter_max": partial(basic_binary_op_flop_jit, name='aten::add'), # not 100% sure
    "aten::scatter_add_": partial(basic_binary_op_flop_jit, name='aten::scatter_add_'), # not 100% sure
}