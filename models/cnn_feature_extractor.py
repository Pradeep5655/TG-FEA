"""
====================================================
CNN Feature Extractor
TG-FEA Project
====================================================
"""

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout
)

from tensorflow.keras.models import Model


def build_cnn_feature_extractor(input_shape=(115, 1)):

    inputs = Input(shape=input_shape, name="Input_Layer")

    x = Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="Conv1D_1"
    )(inputs)

    x = BatchNormalization(name="BatchNorm_1")(x)

    x = MaxPooling1D(
        pool_size=2,
        name="MaxPool_1"
    )(x)

    x = Dropout(
        rate=0.30,
        name="Dropout_1"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=x,
        name="CNN_Feature_Extractor"
    )

    return model
  
if __name__ == "__main__":

    model = build_cnn_feature_extractor()

    model.summary()