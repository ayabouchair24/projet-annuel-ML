"""
mlp_wrapper.py : interface Python (ctypes) de la bibliothèque C++ MLP.

Le calcul (forward pass, rétropropagation) est fait EN C++ (MLP.cpp).
Ce fichier ne fait que : charger la bibliothèque, convertir les tableaux numpy,
suivre les courbes d'apprentissage, et sauvegarder / charger les modèles.

Compilation de la bibliothèque (Linux / macOS) :
    g++ -std=c++20 -shared -fPIC -O2 -o libmlp.so MLP.cpp
(sous Windows, CLion / Visual Studio produisent une .dll)

Exemple :
    from mlp_wrapper import MLP
    mlp = MLP([2, 3, 1], activation="tanh", task="classification", seed=42)
    hist = mlp.fit(X, Y, alpha=0.1, epochs=5000)
    mlp.predict_class(X)
    mlp.save("xor.npz")
    mlp2 = MLP.load("xor.npz")

Conventions :
  - Classification : on fournit toujours Y dans {-1, 1} (comme dans le notebook).
    Avec la sigmoïde, le wrapper convertit lui-même en {0, 1} (sortie dans ]0, 1[).
  - Régression : Y = valeurs réelles, la sortie du réseau est linéaire.
  - Y peut être de forme (n,) ou (n, nb_sorties).
"""
import ctypes
import json
import os
import sys

import numpy as np

_ACTIVATIONS = {"tanh": 0, "sigmoid": 1, "sigmoide": 1, "sigmoïde": 1}
_dbl_p = ctypes.POINTER(ctypes.c_double)
_lib = None


def _find_library(path=None):
    """Cherche la bibliothèque : chemin donné, variable MLP_LIB, ou à côté de ce fichier."""
    if path is None:
        path = os.environ.get("MLP_LIB")
    if path is not None:
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    names = ["libmlp.so", "libmlp.dylib", "mlp.dll", "libmlp.dll"]
    for folder in (here, os.getcwd()):
        for name in names:
            candidate = os.path.join(folder, name)
            if os.path.exists(candidate):
                return candidate
    raise FileNotFoundError(
        "Bibliothèque MLP introuvable. Compile-la avec :\n"
        "  g++ -std=c++20 -shared -fPIC -O2 -o libmlp.so MLP.cpp\n"
        "puis place-la à côté de mlp_wrapper.py, ou appelle load_library('chemin') "
        "ou définis la variable d'environnement MLP_LIB."
    )


def load_library(path=None):
    """Charge la bibliothèque C++ et déclare les types des fonctions (une seule fois)."""
    global _lib
    lib = ctypes.CDLL(os.path.abspath(_find_library(path)))
    i32, f64, c_bool, vp = ctypes.c_int32, ctypes.c_double, ctypes.c_bool, ctypes.c_void_p

    lib.create_mlp_model.restype = vp
    lib.create_mlp_model.argtypes = [ctypes.POINTER(i32), i32]
    lib.set_mlp_activation.argtypes = [vp, i32]
    lib.set_mlp_seed.argtypes = [ctypes.c_uint32]
    lib.predict_mlp_model.restype = _dbl_p
    lib.predict_mlp_model.argtypes = [vp, _dbl_p, c_bool]
    lib.train_mlp_model.argtypes = [vp, _dbl_p, _dbl_p, i32, i32, i32, f64, i32, c_bool]
    lib.get_mlp_loss.restype = f64
    lib.get_mlp_loss.argtypes = [vp, _dbl_p, _dbl_p, i32, i32, i32, c_bool]
    lib.get_mlp_num_weights.restype = i32
    lib.get_mlp_num_weights.argtypes = [vp]
    lib.get_mlp_weights.argtypes = [vp, _dbl_p]
    lib.set_mlp_weights.argtypes = [vp, _dbl_p]
    lib.destroy_mlp_model.argtypes = [vp]
    _lib = lib
    return lib


def _ptr(a):
    return a.ctypes.data_as(_dbl_p)


class MLP:
    """Perceptron Multi Couches (calcul en C++)."""

    def __init__(self, layers, activation="tanh", task="classification", seed=None):
        """
        layers     : nombre de neurones par couche SANS compter le biais, ex. [2, 3, 1]
        activation : "tanh" ou "sigmoid"
        task       : "classification" ou "regression"
        seed       : graine aléatoire (init des poids + tirage des exemples)
        """
        if _lib is None:
            load_library()
        if task not in ("classification", "regression"):
            raise ValueError("task doit valoir 'classification' ou 'regression'")
        if activation.lower() not in _ACTIVATIONS:
            raise ValueError("activation doit valoir 'tanh' ou 'sigmoid'")
        if len(layers) < 2:
            raise ValueError("il faut au moins une couche d'entrée et une couche de sortie")

        self.layers = [int(n) for n in layers]
        self.activation = "sigmoid" if _ACTIVATIONS[activation.lower()] == 1 else "tanh"
        self.task = task
        self.is_clf = task == "classification"
        self._handle = None
        self.meta = {}

        if seed is not None:
            _lib.set_mlp_seed(int(seed))  # attention : la graine est globale au processus (srand)
        arr = (ctypes.c_int32 * len(self.layers))(*self.layers)
        self._handle = _lib.create_mlp_model(arr, len(self.layers))
        _lib.set_mlp_activation(self._handle, _ACTIVATIONS[self.activation])

    # ------------------------------------------------------------------ outils internes
    @property
    def n_inputs(self):
        return self.layers[0]

    @property
    def n_outputs(self):
        return self.layers[-1]

    def _prep_X(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1) if X.shape[0] == self.n_inputs else X.reshape(-1, 1)
        if X.shape[1] != self.n_inputs:
            raise ValueError(f"X a {X.shape[1]} colonnes, le réseau attend {self.n_inputs} entrées")
        return X

    def _prep_Y(self, Y, n):
        Y = np.array(Y, dtype=np.float64).reshape(n, -1)
        if Y.shape[1] != self.n_outputs:
            raise ValueError(f"Y a {Y.shape[1]} colonnes, le réseau a {self.n_outputs} sorties")
        if self.is_clf and self.activation == "sigmoid":
            Y = (Y > 0).astype(np.float64)  # {-1, 1} -> {0, 1}
        return np.ascontiguousarray(Y)

    # ------------------------------------------------------------------ API principale
    def predict(self, X):
        """Sorties brutes du réseau, tableau (n, nb_sorties)."""
        X = self._prep_X(X)
        n = X.shape[0]
        out = np.empty((n, self.n_outputs))
        for k in range(n):
            p = _lib.predict_mlp_model(self._handle, _ptr(X[k]), self.is_clf)
            out[k] = [p[j] for j in range(self.n_outputs)]  # copie immédiate (le pointeur est interne au C++)
        return out

    def predict_class(self, X):
        """Classes prédites, au même format que Y (valeurs -1 / 1). Classification uniquement."""
        if not self.is_clf:
            raise ValueError("predict_class n'a de sens qu'en classification")
        p = self.predict(X)
        if self.n_outputs == 1:
            thr = 0.5 if self.activation == "sigmoid" else 0.0
            return np.where(p > thr, 1, -1)
        res = -np.ones_like(p)
        res[np.arange(len(p)), np.argmax(p, axis=1)] = 1
        return res

    def loss(self, X, Y):
        """Erreur quadratique moyenne (MSE), calculée en C++."""
        X = self._prep_X(X)
        Y = self._prep_Y(Y, X.shape[0])
        return _lib.get_mlp_loss(self._handle, _ptr(X), _ptr(Y), X.shape[0], X.shape[1],
                                 Y.shape[1], self.is_clf)

    def score(self, X, Y):
        """Classification : précision (0..1). Régression : erreur absolue moyenne."""
        X = self._prep_X(X)
        Y = np.array(Y, dtype=np.float64).reshape(X.shape[0], -1)
        if self.is_clf:
            return float(np.mean(np.all(self.predict_class(X) == np.where(Y > 0, 1, -1), axis=1)))
        return float(np.mean(np.abs(self.predict(X) - Y)))

    def fit(self, X, Y, alpha=0.01, epochs=1000, X_val=None, Y_val=None, log_every=None):
        """
        Entraîne le réseau (rétropropagation du gradient stochastique, en C++).
        Une époque = autant d'itérations que d'exemples (un exemple tiré au hasard par itération).

        Retourne un dictionnaire d'historique pour tracer les courbes :
            "epoch", "train_loss", "train_score" (+ "val_loss", "val_score" si X_val/Y_val donnés)
        On entraîne par paquets de 'log_every' époques et on mesure l'erreur entre les paquets.
        """
        X = self._prep_X(X)
        Y_in = Y
        Y = self._prep_Y(Y, X.shape[0])
        if log_every is None:
            log_every = max(1, epochs // 100)
        has_val = X_val is not None and Y_val is not None

        hist = {"epoch": [], "train_loss": [], "train_score": []}
        if has_val:
            hist.update(val_loss=[], val_score=[])

        def log(ep):
            hist["epoch"].append(ep)
            hist["train_loss"].append(self.loss(X, Y_in))
            hist["train_score"].append(self.score(X, Y_in))
            if has_val:
                hist["val_loss"].append(self.loss(X_val, Y_val))
                hist["val_score"].append(self.score(X_val, Y_val))

        log(0)
        done = 0
        while done < epochs:
            step = min(log_every, epochs - done)
            _lib.train_mlp_model(self._handle, _ptr(X), _ptr(Y), X.shape[0], X.shape[1],
                                 Y.shape[1], float(alpha), int(step), self.is_clf)
            done += step
            log(done)
        return hist

    # ------------------------------------------------------------------ poids / sauvegarde
    def get_weights(self):
        """Tous les poids aplatis (ordre : couche, neurone de départ (0 = biais), neurone d'arrivée)."""
        n = _lib.get_mlp_num_weights(self._handle)
        w = np.empty(n, dtype=np.float64)
        _lib.get_mlp_weights(self._handle, _ptr(w))
        return w

    def set_weights(self, w):
        w = np.ascontiguousarray(w, dtype=np.float64).ravel()
        if len(w) != _lib.get_mlp_num_weights(self._handle):
            raise ValueError("nombre de poids incompatible avec l'architecture")
        _lib.set_mlp_weights(self._handle, _ptr(w))

    @property
    def n_weights(self):
        return int(_lib.get_mlp_num_weights(self._handle))

    def save(self, path, meta=None):
        """
        Sauvegarde architecture + activation + tâche + poids dans un fichier .npz.
        meta : dictionnaire facultatif (ex. noms des classes, taille d'image du prétraitement), sauvegardé en JSON,
               pour que l'application puisse recharger le modèle ET savoir comment préparer les images.
        """
        if not str(path).endswith(".npz"):
            path = str(path) + ".npz"
        np.savez(path, layers=np.array(self.layers, dtype=np.int32),
                 activation=np.array(self.activation), task=np.array(self.task),
                 weights=self.get_weights(), meta=np.array(json.dumps(meta or {})))
        return path

    @classmethod
    def load(cls, path):
        """Recrée un modèle identique à celui sauvegardé (même prédictions)."""
        if not str(path).endswith(".npz"):
            path = str(path) + ".npz"
        d = np.load(path, allow_pickle=False)
        model = cls([int(n) for n in d["layers"]], activation=str(d["activation"]),
                    task=str(d["task"]))
        model.set_weights(d["weights"])
        model.meta = json.loads(str(d["meta"])) if "meta" in d.files else {}   # métadonnées sauvegardées
        return model

    # ------------------------------------------------------------------ mémoire
    def close(self):
        if self._handle is not None and _lib is not None:
            _lib.destroy_mlp_model(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (f"MLP(layers={self.layers}, activation='{self.activation}', "
                f"task='{self.task}', n_weights={self.n_weights})")
