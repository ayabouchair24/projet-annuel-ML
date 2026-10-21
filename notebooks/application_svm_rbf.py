import streamlit as st
import numpy as np
import ctypes
import json
from PIL import Image
from pathlib import Path

BASE = Path(__file__).resolve().parent
lib = ctypes.CDLL(str(BASE / 'libPA_V2.dll'))

int_ptr = ctypes.POINTER(ctypes.c_int32)
double_ptr = ctypes.POINTER(ctypes.c_double)

# ==================== MLP ====================
lib.create_mlp_model.argtypes = [int_ptr, ctypes.c_int32]
lib.create_mlp_model.restype = ctypes.c_void_p
lib.set_mlp_activation.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.set_mlp_activation.restype = None
lib.set_mlp_weights.argtypes = [ctypes.c_void_p, double_ptr]
lib.set_mlp_weights.restype = None
lib.predict_mlp_model.argtypes = [ctypes.c_void_p, double_ptr, ctypes.c_bool]
lib.predict_mlp_model.restype = double_ptr
lib.destroy_mlp_model.argtypes = [ctypes.c_void_p]
lib.destroy_mlp_model.restype = None


def predict_flower(image, model_name):
    path = BASE / 'models' / f'mlp_{model_name}.npz'
    with np.load(path) as data:
        layers = np.ascontiguousarray(data['layers'], dtype=np.int32)
        weights = np.ascontiguousarray(data['weights'], dtype=np.float64)
        meta = json.loads(str(data['meta']))

    image = image.convert('RGB').resize((8, 8))
    x = np.asarray(image, dtype=np.float64).flatten()
    x = np.ascontiguousarray(x / 127.5 - 1.0)
    model = lib.create_mlp_model(layers.ctypes.data_as(int_ptr), len(layers))
    if not model:
        raise RuntimeError('MLP creation failed')
    try:
        activation = 0 if model_name == 'tanh' else 1
        lib.set_mlp_activation(model, activation)
        lib.set_mlp_weights(model, weights.ctypes.data_as(double_ptr))
        result_ptr = lib.predict_mlp_model(model, x.ctypes.data_as(double_ptr), True)
        if not result_ptr:
            raise RuntimeError('MLP prediction failed')
        scores = np.ctypeslib.as_array(result_ptr, shape=(int(layers[-1]),)).copy()
    finally:
        lib.destroy_mlp_model(model)

    classes = meta['classes']
    return classes[int(np.argmax(scores))], scores, meta


# ==================== LINEAR MODEL ====================
lib.create_linear_model.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
lib.create_linear_model.restype = ctypes.c_void_p
lib.set_weights.argtypes = [ctypes.c_void_p, double_ptr]
lib.set_weights.restype = None
lib.predict_scores.argtypes = [ctypes.c_void_p, double_ptr, ctypes.c_int32, double_ptr]
lib.predict_scores.restype = None
lib.delete_linear_model.argtypes = [ctypes.c_void_p]
lib.delete_linear_model.restype = None


def predict_flower_linear(image):
    path = BASE / 'models' / 'modele_lineaire.npz'
    with np.load(path) as data:
        weights = np.ascontiguousarray(data['w'], dtype=np.float64)
        mu = np.asarray(data['mu'], dtype=np.float64)
        sd = np.asarray(data['sd'], dtype=np.float64)
        classes = data['classes'].tolist()
        size = int(data['size'])

    image = image.convert('RGB').resize((64, 64), Image.Resampling.BOX)
    x = np.asarray(image, dtype=np.float64)
    if size <= 0 or 64 % size != 0:
        raise ValueError(f'Linear model image size not supported: {size}')
    factor = 64 // size
    x = x.reshape(size, factor, size, factor, 3).mean(axis=(1, 3))
    x = (x / 255.0).reshape(-1)
    x = np.ascontiguousarray((x - mu) / sd, dtype=np.float64)
    n_in, n_out = len(x), len(classes)
    weights = np.ascontiguousarray(weights.ravel(), dtype=np.float64)
    if weights.size != n_out * (n_in + 1):
        raise ValueError(f'Invalid linear model weights: got {weights.size}, expected {n_out * (n_in + 1)}')
    model = lib.create_linear_model(n_in, n_out, 42)
    if not model:
        raise RuntimeError('Linear model creation failed')
    try:
        lib.set_weights(model, weights.ctypes.data_as(double_ptr))
        scores = np.zeros(n_out, dtype=np.float64)
        lib.predict_scores(model, x.ctypes.data_as(double_ptr), 1, scores.ctypes.data_as(double_ptr))
    finally:
        lib.delete_linear_model(model)
    return classes[int(np.argmax(scores))], scores, classes


# ==================== SVM + RBF EXPORTED FROM JUPYTER ====================

def image_to_features(image, taille):
    """Same RGB / Image.BOX / [-1, 1] preprocessing as images_utils.py."""
    im = image.convert('RGB').resize((taille, taille), Image.Resampling.BOX)
    return np.ascontiguousarray(np.asarray(im, dtype=np.float64).reshape(-1) / 127.5 - 1.0)


@st.cache_data(show_spinner=False)
def load_exported_model(algorithm):
    filename = 'svm_fleurs.npz' if algorithm == 'SVM' else 'rbf_fleurs.npz'
    path = BASE / 'models' / filename
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} introuvable. Exécute la cellule 7 (Export) du notebook, '
            'puis copie le fichier .npz dans models/.'
        )
    with np.load(path, allow_pickle=False) as data:
        model = {k: data[k].copy() for k in data.files}
    if str(model['algo']) != algorithm:
        raise ValueError(f'{filename} does not contain a {algorithm} model')
    model['config'] = json.loads(str(model['config']))
    model['classes'] = model['classes'].tolist()
    model['taille'] = int(model['taille'])
    model['dimensions'] = int(model['dimensions'])
    return model


def predict_flower_exported(image, algorithm):
    m = load_exported_model(algorithm)
    x = image_to_features(image, m['taille'])
    if x.size != m['dimensions']:
        raise ValueError(f'Wrong feature count: {x.size} instead of {m["dimensions"]}')

    gamma = float(m['config']['gamma'])
    scores = []
    for c in range(len(m['classes'])):
        if algorithm == 'SVM':
            sv = m[f'sv_{c}']
            coeffs = m[f'alphasY_{c}']
            bias = float(m[f'bias_{c}'])
            kernel_type = int(m['config']['kernel'])
            if kernel_type == 0:  # LINEAR
                kv = sv @ x
            elif kernel_type == 1:  # POLYNOMIAL: (1 + scalarProduct)^degree
                degree = int(m['config']['degree'])
                kv = np.power(1.0 + sv @ x, degree)
            elif kernel_type == 2:  # RBF: exp(-gamma * squared distance)
                distances2 = np.sum((sv - x) ** 2, axis=1)
                kv = np.exp(-gamma * distances2)
            else:
                raise ValueError(f'Unknown SVM kernel: {kernel_type}')
            score = bias + np.dot(coeffs, kv)
        else:
            centers = m[f'centers_{c}']
            weights = m[f'weights_{c}']
            distances2 = np.sum((centers - x) ** 2, axis=1)
            score = np.dot(weights, np.exp(-gamma * distances2))
        scores.append(float(score))

    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError('Non-finite SVM/RBF scores: check saved model and gamma')
    return m['classes'][int(np.argmax(scores))], scores, m['classes']


# ==================== STREAMLIT ====================
st.title('Flower Classification')
model_type = st.radio(
    'Choose a model',
    ['MLP', 'Linear Model', 'SVM', 'RBF'],
    horizontal=True,
)

if model_type == 'MLP':
    model_name = st.radio('Activation function', ['tanh', 'sigmoid'], horizontal=True)

uploaded = st.file_uploader('Upload a flower image', type=['jpg', 'jpeg', 'png'])
if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption='Selected image', width=300)

    if st.button('Classify'):
        try:
            if model_type == 'MLP':
                label, scores, meta = predict_flower(image, model_name)
                classes = meta['classes']
            elif model_type == 'Linear Model':
                label, scores, classes = predict_flower_linear(image)
            else:
                label, scores, classes = predict_flower_exported(image, model_type)

            st.success(f'Prediction: {label}')
            # These values are raw decision scores, NOT probabilities.
            with st.expander('Show model output scores'):
                for name, score in zip(classes, scores):
                    st.write(f'{name}: {float(score):.4f}')

            if model_type == 'MLP' and 'precision_test_propre' in meta:
                st.caption(f"Saved test accuracy: {meta['precision_test_propre']:.2%}")
        except Exception as exc:
            st.error(str(exc))
