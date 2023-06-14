import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf


def load_and_process_image(path: str) -> np.ndarray:
    image = tf.keras.applications.vgg19.preprocess_input(mpimg.imread(path))
    return np.expand_dims(image, axis=0)


def deprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Deprocess the VGG19 image processed
    """
    if len(img.shape) == 4:
        img = np.squeeze(img, 0)
    if len(img.shape) != 3:
        raise ValueError

    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 123.68

    img = img[:, :, ::-1]

    return np.clip(img, 0, 255).astype('uint8')