"""
Assia

linear_model.py - Interface Python (ctypes) vers la bibliothèque C++ du modèle linéaire.

La bibliothèque doit être compilée avant (voir cpp/build.sh ou cpp/build.bat) :
    Linux  : liblinear_model.so
    Mac    : liblinear_model.dylib
    Windows: linear_model.dll
"""
import ctypes
import os
import sys
import numpy as np
from numpy.ctypeslib import ndpointer

_DBL = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


def _lib_name():
    if sys.platform.startswith("win"):
        return "linear_model.dll"
    if sys.platform == "darwin":
        return "liblinear_model.dylib"
    return "liblinear_model.so"


def load_library(lib_dir=None):
    """Charge la bibliothèque et déclare les types (argtypes / restype)."""
    if lib_dir is None:
        lib_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(lib_dir, _lib_name()))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bibliothèque introuvable : {path}. Compilez d'abord cpp/linear_model.cpp.")
    lib = ctypes.CDLL(path)

    lib.create_linear_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.create_linear_model.restype = ctypes.c_void_p

    lib.delete_linear_model.argtypes = [ctypes.c_void_p]
    lib.delete_linear_model.restype = None

    lib.get_n_in.argtypes = [ctypes.c_void_p]
    lib.get_n_in.restype = ctypes.c_int
    lib.get_n_out.argtypes = [ctypes.c_void_p]
    lib.get_n_out.restype = ctypes.c_int

    lib.get_weights.argtypes = [ctypes.c_void_p, _DBL]
    lib.get_weights.restype = None
    lib.set_weights.argtypes = [ctypes.c_void_p, _DBL]
    lib.set_weights.restype = None

    lib.predict_scores.argtypes = [ctypes.c_void_p, _DBL, ctypes.c_int, _DBL]
    lib.predict_scores.restype = None

    lib.train_rosenblatt.argtypes = [ctypes.c_void_p, _DBL, _DBL, ctypes.c_int,
                                     ctypes.c_double, ctypes.c_double,
                                     ctypes.c_int, ctypes.c_int, _DBL]
    lib.train_rosenblatt.restype = ctypes.c_int

    lib.train_pseudo_inverse.argtypes = [ctypes.c_void_p, _DBL, _DBL, ctypes.c_int, ctypes.c_double]
    lib.train_pseudo_inverse.restype = ctypes.c_int

    lib.save_linear_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.save_linear_model.restype = ctypes.c_int
    lib.load_linear_model.argtypes = [ctypes.c_char_p]
    lib.load_linear_model.restype = ctypes.c_void_p
    return lib


class LinearModel:
    """Modèle linéaire (n_in entrées, n_out sorties) dont le calcul est fait en C++."""

    def __init__(self, n_in=None, n_out=None, seed=0, lib=None, _handle=None):
        self.lib = lib if lib is not None else load_library()
        if _handle is not None:
            self._h = _handle
        else:
            self._h = self.lib.create_linear_model(int(n_in), int(n_out), int(seed))
            if not self._h:
                raise ValueError("Création du modèle impossible (n_in / n_out invalides).")
        self.n_in = self.lib.get_n_in(self._h)
        self.n_out = self.lib.get_n_out(self._h)

    # -- libération mémoire C++ --------------------------------------------
    def close(self):
        if getattr(self, "_h", None):
            self.lib.delete_linear_model(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- entraînement -----------------------------------------------------
    def fit_rosenblatt(self, X, Y, lr=0.01, momentum=0.0, epochs=100, seed=0):
        """Classification. Y : cibles -1/+1 de forme (n, n_out). Retourne l'historique d'erreur par epoch."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.ascontiguousarray(Y, dtype=np.float64).reshape(len(X), self.n_out)
        hist = np.zeros(epochs, dtype=np.float64)
        done = self.lib.train_rosenblatt(self._h, X, Y, len(X), lr, momentum, epochs, seed, hist)
        return hist[:done]

    def fit_pseudo_inverse(self, X, Y, ridge=1e-8):
        """Régression (ou classification par moindres carrés)."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.ascontiguousarray(Y, dtype=np.float64).reshape(len(X), self.n_out)
        status = self.lib.train_pseudo_inverse(self._h, X, Y, len(X), ridge)
        if status != 0:
            raise np.linalg.LinAlgError("Matrice singulière : augmentez le paramètre ridge.")
        return self

    # -- prédiction -------------------------------------------------------
    def predict_scores(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        out = np.zeros((len(X), self.n_out), dtype=np.float64)
        self.lib.predict_scores(self._h, X, len(X), out)
        return out

    def predict_class(self, X):
        """Classe prédite : signe si une seule sortie, argmax des scores sinon (pas de softmax)."""
        s = self.predict_scores(X)
        if self.n_out == 1:
            return np.where(s[:, 0] >= 0, 1, -1)
        return np.argmax(s, axis=1)

    def predict_regression(self, X):
        s = self.predict_scores(X)
        return s[:, 0] if self.n_out == 1 else s

    # -- poids, sauvegarde ------------------------------------------------
    def get_weights(self):
        w = np.zeros(self.n_out * (self.n_in + 1), dtype=np.float64)
        self.lib.get_weights(self._h, w)
        return w.reshape(self.n_out, self.n_in + 1)

    def set_weights(self, w):
        w = np.ascontiguousarray(w, dtype=np.float64)
        self.lib.set_weights(self._h, w)

    def save(self, path):
        if self.lib.save_linear_model(self._h, path.encode("utf-8")) != 0:
            raise IOError(f"Impossible d'écrire {path}")

    @classmethod
    def load(cls, path, lib=None):
        lib = lib if lib is not None else load_library()
        h = lib.load_linear_model(path.encode("utf-8"))
        if not h:
            raise IOError(f"Impossible de charger {path}")
        return cls(lib=lib, _handle=h)


def one_hot_pm1(labels, n_classes):
    """Étiquettes entières -> cibles -1/+1 (une colonne par classe)."""
    Y = -np.ones((len(labels), n_classes))
    Y[np.arange(len(labels)), np.asarray(labels, dtype=int)] = 1.0
    return Y
