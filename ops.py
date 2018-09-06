import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from utils import *


def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def conv2d(input_, output_dim, ks=7, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)


def deconv2d(input_, output_dim, ks=7, s=2, stddev=0.02, padding='SAME', name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def relu(tensor_in):
    if tensor_in is not None:
        return tf.nn.relu(tensor_in)
    else:
        return tensor_in


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def to_binary_tf(bar_or_track_bar, threshold=0.0, track_mode=False, melody=False):
    """Return the binarize tensor of the input tensor (be careful of the channel order!)"""
    if track_mode:
        # melody track
        if melody:
            melody_is_max = tf.equal(bar_or_track_bar, tf.reduce_max(bar_or_track_bar, axis=2, keep_dims=True))
            melody_pass_threshold = (bar_or_track_bar > threshold)
            out_tensor = tf.logical_and(melody_is_max, melody_pass_threshold)
        # non-melody track
        else:
            out_tensor = (bar_or_track_bar > threshold)
        return out_tensor
    else:
        if len(bar_or_track_bar.get_shape()) == 4:
            melody_track = tf.slice(bar_or_track_bar, [0, 0, 0, 0], [-1, -1, -1, 1])
            other_tracks = tf.slice(bar_or_track_bar, [0, 0, 0, 1], [-1, -1, -1, -1])
        elif len(bar_or_track_bar.get_shape()) == 5:
            melody_track = tf.slice(bar_or_track_bar, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 1])
            other_tracks = tf.slice(bar_or_track_bar, [0, 0, 0, 0, 1], [-1, -1, -1, -1, -1])
        # melody track
        melody_is_max = tf.equal(melody_track, tf.reduce_max(melody_track, axis=2, keep_dims=True))
        melody_pass_threshold = (melody_track > threshold)
        out_tensor_melody = tf.logical_and(melody_is_max, melody_pass_threshold)
        # other tracks
        out_tensor_others = (other_tracks > threshold)
        if len(bar_or_track_bar.get_shape()) == 4:
            return tf.concat([out_tensor_melody, out_tensor_others], 3)
        elif len(bar_or_track_bar.get_shape()) == 5:
            return tf.concat([out_tensor_melody, out_tensor_others], 4)


def to_chroma_tf(bar_or_track_bar, is_normalize=True):
    """Return the chroma tensor of the input tensor"""
    out_shape = tf.stack([tf.shape(bar_or_track_bar)[0], bar_or_track_bar.get_shape()[1], 12, 7,
                         bar_or_track_bar.get_shape()[3]])
    chroma = tf.reduce_sum(tf.reshape(tf.cast(bar_or_track_bar, tf.float32), out_shape), axis=3)
    if is_normalize:
        chroma_max = tf.reduce_max(chroma, axis=(1, 2, 3), keep_dims=True)
        chroma_min = tf.reduce_min(chroma, axis=(1, 2, 3), keep_dims=True)
        return tf.truediv(chroma - chroma_min, (chroma_max - chroma_min + 1e-15))
    else:
        return chroma


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keep_dims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track


def conv2d_musegan(tensor_in, out_channels, kernels, strides, stddev=0.02, name='conv2d', reuse=None, padding='VALID'):
    """
    Apply a 2D convolution layer on the input tensor and return the resulting tensor.

    Args:
        tensor_in (tensor): The input tensor.
        out_channels (int): The number of output channels.
        kernels (list of int): The size of the kernel. [kernel_height, kernel_width]
        strides (list of int): The stride of the sliding window. [stride_height, stride_width]
        stddev (float): The value passed to the truncated normal initializer for weights. Defaults to 0.02.
        name (str): The tenorflow variable scope. Defaults to 'conv2d'.
        reuse (bool): True to reuse weights and biases.
        padding (str): 'SAME' or 'VALID'. The type of padding algorithm to use. Defaults to 'VALID'.

    Returns:
        tensor: The resulting tensor.

    """
    if tensor_in is None:
        return None
    else:
        with tf.variable_scope(name, reuse=reuse):

            print('|   |---'+tf.get_variable_scope().name, tf.get_variable_scope().reuse)

            weights = tf.get_variable('weights', kernels+[tensor_in.get_shape()[-1], out_channels],
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(tensor_in, weights, strides=[1]+strides+[1], padding=padding)

            out_shape = tf.stack([tf.shape(tensor_in)[0]]+list(conv.get_shape()[1:]))

            return tf.reshape(tf.nn.bias_add(conv, biases), out_shape)


def deconv2d_musegan(tensor_in, out_shape, out_channels, kernels, strides, stddev=0.02, name='transconv2d', reuse=None,
                padding='VALID'):
    """
    Apply a 2D transposed convolution layer on the input tensor and return the resulting tensor.

    Args:
        tensor_in (tensor): The input tensor.
        out_shape (list of int): The output shape. [height, width]
        out_channels (int): The number of output channels.
        kernels (list of int): The size of the kernel.[kernel_height, kernel_width]
        strides (list of int): The stride of the sliding window. [stride_height, stride_width]
        stddev (float): The value passed to the truncated normal initializer for weights. Defaults to 0.02.
        name (str): The tenorflow variable scope. Defaults to 'transconv2d'.
        reuse (bool): True to reuse weights and biases.
        padding (str): 'SAME' or 'VALID'. The type of padding algorithm to use. Defaults to 'VALID'.

    Returns:
        tensor: The resulting tensor.

    """
    if tensor_in is None:
        return None
    else:
        with tf.variable_scope(name, reuse=reuse):

            print('|   |---'+tf.get_variable_scope().name, tf.get_variable_scope().reuse)

            # filter : [height, width, output_channels, in_channels]
            weights = tf.get_variable('weights', kernels+[out_channels, tensor_in.get_shape()[-1]],
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))

            output_shape = tf.stack([tf.shape(tensor_in)[0]]+out_shape+[out_channels])

            try:
                conv_transpose = tf.nn.conv2d_transpose(tensor_in, weights, output_shape=output_shape,
                                                        strides=[1]+strides+[1], padding=padding)
            except AttributeError:  # Support for verisons of TensorFlow before 0.7.0
                conv_transpose = tf.nn.deconv2d(tensor_in, weights, output_shape=output_shape, strides=[1]+strides+[1],
                                                padding=padding)

            return tf.reshape(tf.nn.bias_add(conv_transpose, biases), output_shape)