import tensorflow as tf
import numpy as np
import GPyOpt
from typing import Callable
from parameters_space import search_space

def build_model(
    num_features: int, layers: list[int], keeps_prob: float
) -> tf.keras.Model:
    layers = [
        tf.keras.layers.Dense(
            units=layers[0], activation="relu", input_shape=(num_features,)
        ),
        tf.keras.layers.Dropout(rate=keeps_prob),
        tf.keras.layers.Dense(units=layers[1], activation="relu"),
        tf.keras.layers.Dropout(rate=keeps_prob),
        tf.keras.layers.Dense(units=layers[2], activation="relu"),
        tf.keras.layers.Dense(units=layers[3], activation="softmax"),
    ]

    model = tf.keras.models.Sequential(layers)

    return model


def black_box_model(x: np.ndarray) -> float:
    learning_rate = x[0][0]
    layers = [int(x[0][1]), int(x[0][2]), int(x[0][3])] + [10]
    keeps_prob = x[0][4]
    beta_1 = x[0][5]
    beta_2 = x[0][6]
    batch_size = x[0][7]

    datasets = np.load("data/MNIST.npz")
    X_train = datasets["X_train"]
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = tf.keras.utils.to_categorical(datasets["Y_train"])
    X_valid = datasets["X_valid"]
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = tf.keras.utils.to_categorical(datasets["Y_valid"])

    model = build_model(X_train.shape[1], layers, keeps_prob)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(
            beta_1=beta_1, beta_2=beta_2, lr=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    filepath = f"models/lr_{np.round(learning_rate, 5)}-beta_1_{np.round(beta_1, 5)}-beta_2_{np.round(beta_2, 5)}-batchsize_{batch_size}-layer_1_{layers[0]}-layer_2_{layers[0]}-layer_3_{layers[2]}-keep_probs_{keeps_prob}.h5"

    callbacks = [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=4),
        tf.keras.callbacks.ModelCheckpoint(
            filepath, save_best_only=True, monitor="val_loss"
        ),
    ]

    history: tf.keras.callbacks.History = model.fit(
        X_train,
        Y_train,
        batch_size=int(batch_size),
        epochs=50,
        validation_data=(X_valid, Y_valid),
        callbacks=callbacks,
        verbose=2,
    )
    return -history.history["val_accuracy"][-1]


def optimize_model(model_build_function: Callable[[np.ndarray], float]) -> None:
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=model_build_function,
        domain=search_space,
        model_type="GP",
        initial_design_numdata=1,
        acquisition_type="EI",
        maximize=False,
        verbosity=True,
    )

    optimizer.run_optimization(
        max_iter=30,
        report_file="report_file.txt",
        models_file="model_file.txt",
        evaluations_file="evaluation_file.txt",
    )


if __name__ == "__main__":
    optimize_model(black_box_model)
