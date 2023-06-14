import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from typing import Tuple

from losses import compute_grads
from maths import compute_gram_matrix
from model import get_model, get_feature_representation
from preprocess import load_and_process_image, deprocess_image


def run_style_tranfer(
        style_path: str,
        content_path: str,
        num_iterations: int = 1000,
        style_weight: float = 1e-2,
        content_weight: float = 1e3,
) -> Tuple[np.ndarray, float]:
    model = get_model()

    style_image = load_and_process_image(style_path)
    content_image = load_and_process_image(content_path)

    style_features, content_feature = get_feature_representation(model, style_image, content_image)
    gram_style_features = [compute_gram_matrix(style_feature) for style_feature in style_features]

    input_image = load_and_process_image(content_path)
    input_image = tf.Variable(input_image, tf.float32)

    opt = tf.optimizers.legacy.Adam(learning_rate=10.0)

    best_loss, best_img = float('inf'), None
    loss_weight = (style_weight, content_weight)
    config = {
        "model": model,
        "loss_weight": loss_weight,
        "input_image": input_image,
        "gram_style_features": gram_style_features,
        "content_features": content_feature
    }

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    for i in range(num_iterations):
        grads, all_loss = compute_grads(config)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, input_image)])
        clipped = tf.clip_by_value(input_image, min_vals, max_vals)
        input_image.assign(clipped)

        numerical_loss = loss.numpy()

        if i % 100 == 0:
            print(f"Epochs {i} / {num_iterations}")

        if numerical_loss < best_loss:
            best_loss = numerical_loss
            best_img = input_image.numpy()

    return best_img, best_loss


if __name__ == "__main__":
    best, best_loss = run_style_tranfer(
        "./data/iris.jpg",
        "./data/turtle.jpg",
    )

    best_img = deprocess_image(best)
    plt.imshow(best_img)
    plt.show()
