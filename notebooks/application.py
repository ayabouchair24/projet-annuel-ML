
import streamlit as st
import numpy as np
import ctypes
import json
from PIL import Image
from pathlib import Path

BASE = Path(__file__).resolve().parent

# Load the DLL containing your teammate's MLP.cpp
lib = ctypes.CDLL(str(BASE / "libPA_V2.dll"))

int_ptr = ctypes.POINTER(ctypes.c_int32)
double_ptr = ctypes.POINTER(ctypes.c_double)

lib.create_mlp_model.argtypes = [int_ptr, ctypes.c_int32]
lib.create_mlp_model.restype = ctypes.c_void_p

lib.set_mlp_activation.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.set_mlp_activation.restype = None

lib.set_mlp_weights.argtypes = [ctypes.c_void_p, double_ptr]
lib.set_mlp_weights.restype = None

lib.predict_mlp_model.argtypes = [
    ctypes.c_void_p, double_ptr, ctypes.c_bool
]
lib.predict_mlp_model.restype = double_ptr

lib.destroy_mlp_model.argtypes = [ctypes.c_void_p]
lib.destroy_mlp_model.restype = None


def predict_flower(image, model_name):
    path = BASE / "models" / f"mlp_{model_name}.npz"

    with np.load(path) as data:
        layers = np.ascontiguousarray(data["layers"], dtype=np.int32)
        weights = np.ascontiguousarray(data["weights"], dtype=np.float64)
        meta = json.loads(str(data["meta"]))

    # 8x8 RGB, normalized to [-1, 1]
    image = image.convert("RGB").resize((8, 8))
    x = np.asarray(image, dtype=np.float64).flatten()
    x = np.ascontiguousarray(x / 127.5 - 1.0)

    model = lib.create_mlp_model(
        layers.ctypes.data_as(int_ptr), len(layers)
    )

    if not model:
        raise RuntimeError("MLP creation failed")

    try:
        activation = 0 if model_name == "tanh" else 1
        lib.set_mlp_activation(model, activation)

        lib.set_mlp_weights(
            model, weights.ctypes.data_as(double_ptr)
        )

        result_ptr = lib.predict_mlp_model(
            model, x.ctypes.data_as(double_ptr), True
        )

        scores = np.ctypeslib.as_array(
            result_ptr, shape=(int(layers[-1]),)
        ).copy()

    finally:
        lib.destroy_mlp_model(model)

    classes = meta["classes"]
    predicted_class = classes[int(np.argmax(scores))]
    return predicted_class, scores, meta


st.title("Flower Classification — MLP")

model_name = st.radio(
    "Activation function",
    ["tanh", "sigmoid"],
    horizontal=True
)

uploaded = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Selected image", width=300)

    if st.button("Classify"):
        label, scores, meta = predict_flower(image, model_name)

        st.success(f"Prediction: {label}")

        st.write("Model output scores:")
        for name, score in zip(meta["classes"], scores):
            st.write(f"{name}: {score:.4f}")

        st.caption(
            f"Saved test accuracy: "
            f"{meta['precision_test_propre']:.2%}"
        )
