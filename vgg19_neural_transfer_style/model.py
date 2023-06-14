import tensorflow as tf

from typing import Tuple

from constants import STYLE_LAYERS, CONTENT_LAYERS, LEN_STYLE_LAYERS


def get_model() -> tf.keras.Model:
    """
    Load a pre-trained model, and tweak-it to have access to intermediate layers
    Allow us ot get the features maps from these layers as output of the new model
    """
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]
    content_outputs = [vgg.get_layer(name).output for name in CONTENT_LAYERS]
    model_outputs = style_outputs + content_outputs
    return tf.keras.models.Model(vgg.input, model_outputs)


def get_feature_representation(
        model: tf.keras.Model,
        style_image: tf.Tensor,
        content_image: tf.Tensor
) -> Tuple[list[tf.Tensor], list[tf.Tensor]]:
    """
    Get the feature representation from the outputs layers of the model
    -> See LEN_STYLE_LAYERS | LEN_CONTENT_LAYERS to see the concerned ones
    """
    style_output = model(style_image)
    content_output = model(content_image)

    style_features_representation = [feature for feature in style_output[:LEN_STYLE_LAYERS]]
    content_features_representation = [feature for feature in content_output[LEN_STYLE_LAYERS:]]

    return style_features_representation, content_features_representation
