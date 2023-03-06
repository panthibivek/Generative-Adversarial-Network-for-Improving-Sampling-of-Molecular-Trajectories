
""" @author : Bivek Panthi
"""

import numpy as np
import tensorflow as tf

def random_shuffle(x_tensor, y_tensor):
    idx = tf.range(start=0, limit=tf.shape(x_tensor)[0], dtype=tf.int32)
    shuffled_idx = tf.random.shuffle(idx)
    x_tensor, y_tensor = tf.gather(x_tensor, shuffled_idx), tf.gather(y_tensor, shuffled_idx)
    return x_tensor, y_tensor

def random_generator(shape_):
    random_gaussian_vectors = tf.random.normal(shape=shape_)
    random_gaussian_vectors = tf.cast(random_gaussian_vectors, dtype=tf.float64)
    # combine_vector = tf.concat([random_gaussian_vectors], axis=1)
    return tf.cast(random_gaussian_vectors, dtype=tf.float64)

def unison_shuffle(arr1 : np.array, arr2 : np.array):
    assert len(arr1) == len(arr2)
    p = np.random.permutation(len(arr1))
    return arr1[p], arr2[p]

def __gen(reader):
    while True:
        temp = reader(2**16)
        if not temp: break
        yield temp

def buf_count_newlines(filename : str):
    with open(filename, "rb") as fb:
        count = sum(buf.count(b"\n") for buf in __gen(fb.raw.read))
    return count

def ExtractLines(filename : str, n : int, line_length : int = 14):
    with open(filename, 'r') as fp:
        x = fp.readlines()[n*line_length:(n+1)*line_length]
    output_filename = 'molecule{}.xyz'.format(n)
    with open(output_filename, "w") as text_file:
        text_file.writelines(x)