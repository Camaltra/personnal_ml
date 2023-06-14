import tensorflow as tf


def compute_gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    """
    Input size of (1, height, width, channels)
    Output size of (height * width, channels)
    """
    _, __, ___, channels = input_tensor.shape
    reshape_input = tf.reshape(input_tensor, [-1, channels])
    coef = reshape_input.shape[0]
    gram_matrix = tf.matmul(reshape_input, reshape_input, transpose_a=True)

    return gram_matrix / tf.cast(coef, tf.float32)
