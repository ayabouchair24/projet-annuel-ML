# Fichier de Nadine Zerguine : wrapper Python du MLP
import ctypes
import json
import numpy as np

lib = None
dbl_p = ctypes.POINTER(ctypes.c_double)


def load_library(path):
    global lib
    lib = ctypes.CDLL(path)
    i32, f64, boolean, vp = ctypes.c_int32, ctypes.c_double, ctypes.c_bool, ctypes.c_void_p

    lib.create_mlp_model.restype = vp
    lib.create_mlp_model.argtypes = [ctypes.POINTER(i32), i32]
    lib.set_mlp_seed.argtypes = [ctypes.c_uint32]
    lib.predict_mlp_model.restype = dbl_p
    lib.predict_mlp_model.argtypes = [vp, dbl_p, boolean]
    lib.train_mlp_model.argtypes = [vp, dbl_p, dbl_p, i32, i32, i32, f64, i32, boolean]
    lib.get_mlp_loss.restype = f64
    lib.get_mlp_loss.argtypes = [vp, dbl_p, dbl_p, i32, i32, i32, boolean]
    lib.get_mlp_num_weights.restype = i32
    lib.get_mlp_num_weights.argtypes = [vp]
    lib.get_mlp_weights.argtypes = [vp, dbl_p]
    lib.set_mlp_weights.argtypes = [vp, dbl_p]
    lib.destroy_mlp_model.argtypes = [vp]


def ptr(a):
    return a.ctypes.data_as(dbl_p)


class MLP:

    def __init__(self, layers, task="classification", seed=None):
        self.layers = list(layers)
        self.task = task
        self.is_clf = (task == "classification")
        self.meta = {}

        if seed is not None:
            lib.set_mlp_seed(seed)
        tableau = (ctypes.c_int32 * len(self.layers))(*self.layers)
        self.handle = lib.create_mlp_model(tableau, len(self.layers))

    @property
    def n_outputs(self):
        return self.layers[-1]

    @property
    def n_weights(self):
        return lib.get_mlp_num_weights(self.handle)

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        res = []
        for k in range(len(X)):
            p = lib.predict_mlp_model(self.handle, ptr(X[k]), self.is_clf)
            res.append([p[j] for j in range(self.n_outputs)])
        return np.array(res)

    def predict_class(self, X):
        p = self.predict(X)
        if self.n_outputs == 1:
            return np.where(p > 0, 1, -1)
        res = -np.ones_like(p)
        res[np.arange(len(p)), np.argmax(p, axis=1)] = 1
        return res

    def loss(self, X, Y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.ascontiguousarray(np.array(Y, dtype=np.float64).reshape(len(X), -1))
        return lib.get_mlp_loss(self.handle, ptr(X), ptr(Y), len(X), X.shape[1], Y.shape[1], self.is_clf)

    def score(self, X, Y):
        Y = np.array(Y, dtype=np.float64).reshape(len(X), -1)
        if self.is_clf:
            bonnes = np.all(self.predict_class(X) == np.where(Y > 0, 1, -1), axis=1)
            return float(np.mean(bonnes))
        return float(np.mean(np.abs(self.predict(X) - Y)))

    def fit(self, X, Y, alpha=0.01, epochs=1000, X_val=None, Y_val=None, log_every=None):
        Xp = np.ascontiguousarray(X, dtype=np.float64)
        Yp = np.ascontiguousarray(np.array(Y, dtype=np.float64).reshape(len(Xp), -1))
        if log_every is None:
            log_every = max(1, epochs // 100)

        hist = {"epoch": [], "train_loss": [], "train_score": []}
        if X_val is not None:
            hist["val_loss"] = []
            hist["val_score"] = []

        def noter(ep):
            hist["epoch"].append(ep)
            hist["train_loss"].append(self.loss(X, Y))
            hist["train_score"].append(self.score(X, Y))
            if X_val is not None:
                hist["val_loss"].append(self.loss(X_val, Y_val))
                hist["val_score"].append(self.score(X_val, Y_val))

        noter(0)
        fait = 0
        while fait < epochs:
            n = min(log_every, epochs - fait)
            lib.train_mlp_model(self.handle, ptr(Xp), ptr(Yp), len(Xp), Xp.shape[1], Yp.shape[1],
                                alpha, n, self.is_clf)
            fait += n
            noter(fait)
        return hist

    def get_weights(self):
        w = np.empty(self.n_weights, dtype=np.float64)
        lib.get_mlp_weights(self.handle, ptr(w))
        return w

    def set_weights(self, w):
        w = np.ascontiguousarray(w, dtype=np.float64)
        lib.set_mlp_weights(self.handle, ptr(w))

    def save(self, path, meta=None):
        if not str(path).endswith(".npz"):
            path = str(path) + ".npz"
        np.savez(path, layers=np.array(self.layers, dtype=np.int32),
                 activation=np.array("tanh"), task=np.array(self.task),
                 weights=self.get_weights(), meta=np.array(json.dumps(meta or {})))
        return path

    @classmethod
    def load(cls, path):
        if not str(path).endswith(".npz"):
            path = str(path) + ".npz"
        d = np.load(path)
        model = cls([int(n) for n in d["layers"]], task=str(d["task"]))
        model.set_weights(d["weights"])
        if "meta" in d.files:
            model.meta = json.loads(str(d["meta"]))
        return model

    def __repr__(self):
        return f"MLP(layers={self.layers}, task='{self.task}')"
