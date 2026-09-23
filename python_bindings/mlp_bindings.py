"""Petite passerelle Python -> API C -> MLP C++.

Le calcul et l'entraînement du PMC restent en C++.
Python sert seulement à appeler l'API et à afficher les résultats.
"""

import ctypes
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIB_PATH = PROJECT_ROOT / "lib_cpp" / "build" / "libmlp.dylib"

if not LIB_PATH.exists():
    raise FileNotFoundError(
        f"Bibliothèque MLP introuvable : {LIB_PATH}\n"
        "Construisez d'abord : cmake -S lib_cpp -B lib_cpp/build && cmake --build lib_cpp/build"
    )

_lib = ctypes.CDLL(str(LIB_PATH))

_lib.create_mlp_model.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.create_mlp_model.restype = ctypes.c_void_p

_lib.destroy_mlp_model.argtypes = [ctypes.c_void_p]
_lib.destroy_mlp_model.restype = None

_lib.predict_mlp_model.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_bool,
]
_lib.predict_mlp_model.restype = ctypes.POINTER(ctypes.c_double)

_lib.train_mlp_model.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_bool,
]
_lib.train_mlp_model.restype = None

_lib.get_mlp_loss.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
]
_lib.get_mlp_loss.restype = ctypes.c_double

_lib.mlp_n_layers.argtypes = [ctypes.c_void_p]
_lib.mlp_n_layers.restype = ctypes.c_int

_lib.mlp_layer_sizes.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
]
_lib.mlp_layer_sizes.restype = None


class MLP:
    def __init__(self, layer_sizes):
        if len(layer_sizes) < 2:
            raise ValueError("Un PMC doit avoir au moins une entrée et une sortie.")

        self._handle = None
        sizes = (ctypes.c_int * len(layer_sizes))(*layer_sizes)
        self._handle = _lib.create_mlp_model(sizes, len(layer_sizes))

        if not self._handle:
            raise RuntimeError("Impossible de créer le PMC C++.")

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            _lib.destroy_mlp_model(handle)
            self._handle = None

    @property
    def layer_sizes(self):
        n_layers = _lib.mlp_n_layers(self._handle)
        result = (ctypes.c_int * n_layers)()
        _lib.mlp_layer_sizes(self._handle, result)
        return list(result)

    @property
    def n_inputs(self):
        return self.layer_sizes[0]

    @property
    def n_outputs(self):
        return self.layer_sizes[-1]

    def predict(self, x):
        if len(x) != self.n_inputs:
            raise ValueError(
                f"Le PMC attend {self.n_inputs} entrées, reçu {len(x)}."
            )

        x_array = (ctypes.c_double * self.n_inputs)(
            *[float(value) for value in x]
        )

        result = _lib.predict_mlp_model(
            self._handle,
            x_array,
            False,
        )

        if not result:
            raise RuntimeError("La prédiction C++ a échoué.")

        return [result[i] for i in range(self.n_outputs)]

    def train(self, X, Y, epochs, learning_rate):
        if len(X) == 0:
            raise ValueError("Le dataset d'entraînement est vide.")
        if len(X) != len(Y):
            raise ValueError("X et Y doivent avoir le même nombre de samples.")

        input_dim = self.n_inputs
        output_dim = self.n_outputs

        if any(len(sample) != input_dim for sample in X):
            raise ValueError("Une entrée n'a pas le bon nombre de features.")
        if any(len(target) != output_dim for target in Y):
            raise ValueError("Une cible n'a pas la bonne dimension.")

        x_flat = [float(v) for sample in X for v in sample]
        y_flat = [float(v) for target in Y for v in target]

        x_array = (ctypes.c_double * len(x_flat))(*x_flat)
        y_array = (ctypes.c_double * len(y_flat))(*y_flat)

        _lib.train_mlp_model(
            self._handle,
            x_array,
            y_array,
            len(X),
            input_dim,
            output_dim,
            float(learning_rate),
            int(epochs),
            False,
        )

    def loss(self, X, Y):
        if len(X) != len(Y) or not X:
            raise ValueError("X et Y doivent être non vides et de même taille.")

        input_dim = self.n_inputs
        output_dim = self.n_outputs

        x_flat = [float(v) for sample in X for v in sample]
        y_flat = [float(v) for target in Y for v in target]

        x_array = (ctypes.c_double * len(x_flat))(*x_flat)
        y_array = (ctypes.c_double * len(y_flat))(*y_flat)

        return _lib.get_mlp_loss(
            self._handle,
            x_array,
            y_array,
            len(X),
            input_dim,
            output_dim,
            False,
        )

    def accuracy_binary(self, X, Y):
        correct = 0
        for x, target in zip(X, Y):
            prediction = 1 if self.predict(x)[0] >= 0.0 else -1
            if prediction == int(target[0]):
                correct += 1
        return correct / len(X)
