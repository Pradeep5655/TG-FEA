"""
====================================================
Hybrid TG-FEA Model
====================================================
"""

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Multiply
)

from tensorflow.keras.models import Model

from cnn_feature_extractor import build_cnn_feature_extractor
from bilstm import build_bilstm
from gru import build_gru
from attention import ThreeWayAttention


def build_hybrid_model(input_shape=(115, 1), num_classes=11):

    # ==========================================
    # Input Layer
    # ==========================================
    inputs = Input(
        shape=input_shape,
        name="Input_Layer"
    )

    # ==========================================
    # Feature Extraction Branches
    # ==========================================
    cnn_branch = build_cnn_feature_extractor(input_shape)
    bilstm_branch = build_bilstm(input_shape)
    gru_branch = build_gru(input_shape)

    z_cnn = cnn_branch(inputs)
    z_lstm = bilstm_branch(inputs)
    z_gru = gru_branch(inputs)

    # ==========================================
    # Three-Way Attention Fusion
    # ==========================================
    attention_layer = ThreeWayAttention()

    fused = attention_layer([
        z_cnn,
        z_lstm,
        z_gru
    ])

    # ==========================================
    # Trust Gate
    # ==========================================
    trust_gate = Dense(
        units=128,
        activation="sigmoid",
        name="Trust_Gate"
    )(fused)

    trusted_features = Multiply(
        name="Trusted_Features"
    )([
        fused,
        trust_gate
    ])

    # ==========================================
    # Classifier
    # ==========================================
    x = Dense(
        64,
        activation="relu",
        name="Dense_1"
    )(trusted_features)

    x = Dropout(
        0.30,
        name="Dropout_1"
    )(x)

    outputs = Dense(
        num_classes,
        activation="softmax",
        name="Output_Layer"
    )(x)

    # ==========================================
    # Build Model
    # ==========================================
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="TG_FEA_Model"
    )

    return model


# ==========================================
# Test Model
# ==========================================
if __name__ == "__main__":

    model = build_hybrid_model()

    model.summary()