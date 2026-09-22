import ctypes
import sys
from pathlib import Path


# ============================================================
# Chargement de la bibliothèque C++
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIB_PATH = PROJECT_ROOT / "lib_cpp" / "build" / "libmlp.dylib"

if not LIB_PATH.exists():
    raise FileNotFoundError(
        f"Bibliothèque MLP introuvable : {LIB_PATH}\n"
        "Construisez d'abord le projet avec : cmake --build lib_cpp/build"
    )

_lib = ctypes.CDLL(str(LIB_PATH))


# ============================================================
# Déclaration de l'API C
# ============================================================

_lib.create_mlp_model.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int
]
_lib.create_mlp_model.restype = ctypes.c_void_p

_lib.destroy_mlp_model.argtypes = [
    ctypes.c_void_p
]
_lib.destroy_mlp_model.restype = None

_lib.predict_mlp_model.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_bool
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
    ctypes.c_bool
]
_lib.train_mlp_model.restype = None

_lib.get_mlp_loss.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool
]
_lib.get_mlp_loss.restype = ctypes.c_double

_lib.mlp_n_layers.argtypes = [
    ctypes.c_void_p
]
_lib.mlp_n_layers.restype = ctypes.c_int

_lib.mlp_layer_sizes.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int)
]
_lib.mlp_layer_sizes.restype = None


# ============================================================
# Classe Python MLP
# ============================================================

class MLP:

    def __init__(self, layer_sizes):
        if len(layer_sizes) < 2:
            raise ValueError(
                "Le MLP doit contenir au moins une couche d'entrée "
                "et une couche de sortie."
            )

        self.layer_sizes_requested = list(layer_sizes)

        sizes = (ctypes.c_int * len(layer_sizes))(*layer_sizes)

        self._handle = _lib.create_mlp_model(
            sizes,
            len(layer_sizes)
        )

        if not self._handle:
            raise RuntimeError("Impossible de créer le modèle MLP.")

    def __del__(self):
        handle = getattr(self, "_handle", None)

        if handle:
            _lib.destroy_mlp_model(handle)
            self._handle = None

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    @property
    def layer_sizes(self):
        n = _lib.mlp_n_layers(self._handle)

        sizes = (ctypes.c_int * n)()

        _lib.mlp_layer_sizes(
            self._handle,
            sizes
        )

        return list(sizes)

    @property
    def n_inputs(self):
        return self.layer_sizes[0]

    @property
    def n_outputs(self):
        return self.layer_sizes[-1]

    # --------------------------------------------------------
    # Forward / prédiction
    # --------------------------------------------------------

    def predict(self, x, classification=False):

        if len(x) != self.n_inputs:
            raise ValueError(
                f"Le modèle attend {self.n_inputs} features, "
                f"mais {len(x)} ont été fournies."
            )

        x_array = (ctypes.c_double * self.n_inputs)(
            *[float(v) for v in x]
        )

        result = _lib.predict_mlp_model(
            self._handle,
            x_array,
            classification
        )

        if not result:
            raise RuntimeError("La prédiction C++ a échoué.")

        return [
            result[i]
            for i in range(self.n_outputs)
        ]

    # --------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------

    def train(
        self,
        X,
        Y,
        epochs,
        learning_rate,
        classification=False
    ):

        sample_count = len(X)

        if sample_count == 0:
            raise ValueError("Le dataset d'entraînement est vide.")

        if len(Y) != sample_count:
            raise ValueError(
                "X et Y doivent contenir le même nombre de samples."
            )

        input_dim = self.n_inputs
        output_dim = self.n_outputs

        X_flat = [
            float(value)
            for sample in X
            for value in sample
        ]

        Y_flat = [
            float(value)
            for sample in Y
            for value in sample
        ]

        X_array = (ctypes.c_double * len(X_flat))(*X_flat)
        Y_array = (ctypes.c_double * len(Y_flat))(*Y_flat)

        _lib.train_mlp_model(
            self._handle,
            X_array,
            Y_array,
            sample_count,
            input_dim,
            output_dim,
            float(learning_rate),
            int(epochs),
            classification
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    def loss(
        self,
        X,
        Y,
        classification=False
    ):

        sample_count = len(X)

        input_dim = self.n_inputs
        output_dim = self.n_outputs

        X_flat = [
            float(value)
            for sample in X
            for value in sample
        ]

        Y_flat = [
            float(value)
            for sample in Y
            for value in sample
        ]

        X_array = (ctypes.c_double * len(X_flat))(*X_flat)
        Y_array = (ctypes.c_double * len(Y_flat))(*Y_flat)

        return _lib.get_mlp_loss(
            self._handle,
            X_array,
            Y_array,
            sample_count,
            input_dim,
            output_dim,
            classification
        )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    def accuracy(self, X, Y):

        correct = 0

        for x, y in zip(X, Y):

            prediction = self.predict(
                x,
                classification=True
            )

            # Classification binaire
            if self.n_outputs == 1:
                predicted_class = prediction[0]
                true_class = float(y[0])

                if predicted_class == true_class:
                    correct += 1

            # Classification multiclasses one-hot
            else:
                predicted_class = max(
                    range(self.n_outputs),
                    key=lambda i: prediction[i]
                )

                true_class = max(
                    range(len(y)),
                    key=lambda i: y[i]
                )

                if predicted_class == true_class:
                    correct += 1

        return correct / len(X)
