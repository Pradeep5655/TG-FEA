"""
====================================================
GRU Feature Extractor
TG-FEA Project
====================================================
"""

from tensorflow.keras.layers import (
    Input,
    GRU
)

from tensorflow.keras.models import Model


def build_gru(input_shape=(115,1)):

    inputs = Input(
        shape=input_shape,
        name="Input_Layer"
    )

    x = GRU(
        units=128,
        return_sequences=False,
        name="GRU_1"
    )(inputs)

    model = Model(
        inputs=inputs,
        outputs=x,
        name="GRU_Branch"
    )

    return model


if __name__ == "__main__":

    model = build_gru()

    model.summary()