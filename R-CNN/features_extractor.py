import numpy as np
import tensorflow as tf
from typing import Tuple

outputs_layer = ["flatten", "fc1", "fc2"]
trainable_layers = ["fc1", "fc2"]


class FeatureExtractor:
    def __init__(self, train_dataset_path, valid_dataset_path):
        self.train_dataset_path = train_dataset_path
        self.valid_dataset_path = valid_dataset_path
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        vgg16 = tf.keras.applications.vgg16.VGG16(include_top=True, weights="imagenet")

        output = vgg16.layers[-3].output
        output_layer = tf.keras.layers.Dense(2, activation="linear", name="fc2")(output)
        model = tf.keras.models.Model(vgg16.input, output_layer)
        for layer in model.layers:
            if layer.name not in trainable_layers:
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return model

    def _get_and_process_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_dataset = np.load(self.train_dataset_path)
        valid_dataset = np.load(self.valid_dataset_path)
        X_train, y_train = train_dataset["images"], train_dataset["labels"]
        X_valid, y_valid = valid_dataset["images"], valid_dataset["labels"]

        for dataset in [X_train, y_valid, X_train, y_valid]:
            if type(dataset) != np.ndarray:
                raise TypeError("Array are not in the right type")

        y_train = tf.keras.utils.to_categorical(y_train, 2)
        y_valid = tf.keras.utils.to_categorical(y_valid, 2)

        train_shuffle = np.random.permutation(len(X_train))
        valid_shuffle = np.random.permutation(len(X_valid))

        return X_train[train_shuffle], y_train[train_shuffle], X_valid[valid_shuffle], y_valid[valid_shuffle]

    def _train(self, X_train, y_train, X_valid, y_valid):
        callback_functions = [
            tf.keras.callbacks.ModelCheckpoint(
                "./vgg16_airplaine.h5",
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            tf.keras.callbacks.EarlyStopping(patience=5)
        ]

        self.model.fit(x=X_train,
                  y=y_train,
                  epochs=50,
                  batch_size=32,
                  validation_data=(X_valid, y_valid),
                  callbacks=callback_functions,
                  verbose=1)

    def run(self) -> None:
        X_train, y_train, X_valid, y_valid = self._get_and_process_data()
        print("Fit the model to the datas")
        self._train(X_train, y_train, X_valid, y_valid)
        print("Save the model to the path ./vgg16_airplaine.h5")


if __name__ == "__main__":
    feature_extractor_model = FeatureExtractor("./data/train/numpy_dataset/processed_img-train.npz", "./data/valid/numpy_dataset/processed_img-valid.npz")
    feature_extractor_model.run()
