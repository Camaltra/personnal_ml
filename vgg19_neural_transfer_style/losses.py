from maths import compute_gram_matrix
import tensorflow as tf

from typing import Any, Tuple

from constants import LEN_STYLE_LAYERS


def get_layer_content_loss(base_content: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(base_content - target))


def get_layer_style_loss(base_style: tf.Tensor, gram_target: tf.Tensor) -> tf.Tensor:
    gram_base_style = compute_gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_base_style - gram_target))


def get_content_loss(base_extracted_content_features: list[tf.Tensor], extracted_content_features: list[tf.Tensor]) -> tf.Tensor:
    weight = 1.0 / float(len(extracted_content_features))
    losses = []
    for base_extracted_content_feature, extracted_content_feature in zip(base_extracted_content_features, extracted_content_features):
        losses.append(weight * get_layer_content_loss(base_extracted_content_feature, extracted_content_feature))
    return tf.add_n(losses)


def get_style_loss(base_extracted_style_features: list[tf.Tensor], gram_extracted_style_features: list[tf.Tensor]) -> tf.Tensor:
    weight = 1.0 / float(len(gram_extracted_style_features))
    losses = []
    for base_extracted_style_feature, extracted_style_feature in zip(base_extracted_style_features, gram_extracted_style_features):
        losses.append(weight * get_layer_style_loss(base_extracted_style_feature, extracted_style_feature))
    return tf.add_n(losses)


def compute_loss(
        model: tf.keras.Model,
        loss_weight: Tuple[float, float],
        input_image: tf.Tensor,
        gram_style_features: list[tf.Tensor],
        content_features: list[tf.Tensor],
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    # Input Image is not really a tensor but a composit tensor.
    # It works even if we give tensor
    features_output = model(input_image)

    style_weight, content_weight = loss_weight
    input_style_feature = features_output[:LEN_STYLE_LAYERS]
    input_content_feature = features_output[LEN_STYLE_LAYERS:]

    style_loss = get_style_loss(input_style_feature, gram_style_features)
    content_loss = get_content_loss(input_content_feature, content_features)

    style_score = style_loss * style_weight
    content_score = content_loss * content_weight

    total_loss = style_score + content_score

    return total_loss, style_score, content_score


@tf.function()
def compute_grads(cfg: dict[str, Any]) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss, _, __ = all_loss
    return tape.gradient(total_loss, cfg['input_image']), all_loss
