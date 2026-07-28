"""
====================================================
BiLSTM Feature Extractor
TG-FEA Project
====================================================
"""

from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM
)

from tensorflow.keras.models import Model


def build_bilstm(input_shape=(115,1)):

    inputs = Input(
        shape=input_shape,
        name="Input_Layer"
    )

    x = Bidirectional(
        LSTM(
            units=64,
            return_sequences=False
        ),
        name="BiLSTM_1"
    )(inputs)

    model = Model(
        inputs=inputs,
        outputs=x,
        name="BiLSTM_Branch"
    )

    return model


if __name__ == "__main__":

    model = build_bilstm()

    model.summary()