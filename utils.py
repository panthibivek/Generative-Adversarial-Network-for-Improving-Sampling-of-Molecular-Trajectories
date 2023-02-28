
""" @author : Bivek
"""

import numpy as np
import tensorflow as tf

def random_generator(shape_, true_y_values):
    random_gaussian_vectors = tf.random.normal(shape=shape_)
    random_gaussian_vectors = tf.cast(random_gaussian_vectors, dtype=tf.float64)
    combine_vector = tf.concat([random_gaussian_vectors, true_y_values], axis=1)
    return tf.cast(combine_vector, dtype=tf.float64)

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