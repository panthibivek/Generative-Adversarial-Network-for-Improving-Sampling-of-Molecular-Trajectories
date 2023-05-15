
""" @author : Bivek Panthi
"""

import numpy as np
import tensorflow as tf

def random_shuffle(x_tensor, y_tensor):
    """
    Function to shuffle two tensor in same order

    Parameters : 
        x_tensor            : first tensor to shuffle
        y_tensor            : second tensor to shuffle

    """
    idx = tf.range(start=0, limit=tf.shape(x_tensor)[0], dtype=tf.int32)
    shuffled_idx = tf.random.shuffle(idx)
    x_tensor, y_tensor = tf.gather(x_tensor, shuffled_idx), tf.gather(y_tensor, shuffled_idx)
    return x_tensor, y_tensor

def random_generator(shape_):
    """
    Function to generate random gaussian noise of shape 'shape_'

    Parameters : 
        shape_              : shape of the gaussian noise tensor
    """
    random_gaussian_vectors = tf.random.normal(shape=shape_)
    random_gaussian_vectors = tf.cast(random_gaussian_vectors, dtype=tf.float64)
    # combine_vector = tf.concat([random_gaussian_vectors], axis=1)
    return tf.cast(random_gaussian_vectors, dtype=tf.float64)

def unison_shuffle(arr1 : np.array, arr2 : np.array):
    """
    Function to shuffle two numpy arrays in same order

    Parameters : 
        arr1                : first numpy array to shuffle
        arr2                : second numpy array to shuffle
    """
    assert len(arr1) == len(arr2)
    p = np.random.permutation(len(arr1))
    return arr1[p], arr2[p]

def __gen(reader):
    while True:
        temp = reader(2**16)
        if not temp: break
        yield temp

def buf_count_newlines(filename : str):
    """
    Function to count the number of newline characters

    Parameters : 
        filename            : name of the file to count the newline
    """
    with open(filename, "rb") as fb:
        count = sum(buf.count(b"\n") for buf in __gen(fb.raw.read))
    return count

def ExtractLines(filename : str, n : int, line_length : int = 14):
    """
    Function to extract specific lines from a file

    Parameters : 
        filename            : name of the file to extract lines
        n                   : file sequence number. Used for structing the filename 
        line_length         : length of lines to extract
    """
    with open(filename, 'r') as fp:
        x = fp.readlines()[n*line_length:(n+1)*line_length]
    output_filename = 'molecule{:05d}.xyz'.format(n)
    with open(output_filename, "w") as text_file:
        text_file.writelines(x)