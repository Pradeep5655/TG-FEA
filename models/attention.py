"""
====================================================
Three-Way Attention Fusion
TG-FEA Project
====================================================
"""

import tensorflow as tf

from tensorflow.keras.layers import (
    Layer,
    Dense
)


class ThreeWayAttention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Shared attention network
        self.attention_dense = Dense(
            units=128,
            activation="sigmoid",
            name="Attention_Dense"
        )
        super().build(input_shape)

    def call(self, inputs):
        z_cnn, z_lstm, z_gru = inputs

        # Compute attention masks
        a_cnn = self.attention_dense(z_cnn)
        a_lstm = self.attention_dense(z_lstm)
        a_gru = self.attention_dense(z_gru)

        # Apply attention
        z_cnn = a_cnn * z_cnn
        z_lstm = a_lstm * z_lstm
        z_gru = a_gru * z_gru

        # Fuse all branches
        fused = (z_cnn + z_lstm + z_gru) / 3.0

        return fused
  
if __name__ == "__main__":

    attention = ThreeWayAttention()

    x1 = tf.random.normal((2, 128))
    x2 = tf.random.normal((2, 128))
    x3 = tf.random.normal((2, 128))

    output = attention([x1, x2, x3])

    print("Output Shape:", output.shape)
    
