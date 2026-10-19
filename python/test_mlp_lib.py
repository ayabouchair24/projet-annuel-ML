"""
Vérification de la bibliothèque MLP (étape 1) sur les 10 cas de test du notebook.
Usage : python test_mlp_lib.py ./libmlp.so
Pour Tanh : labels dans {-1, 1}.  Pour Sigmoïde : labels dans {0, 1} (on convertit -1 -> 0).
"""
import ctypes
import sys
import numpy as np

lib = ctypes.CDLL(sys.argv[1])
dbl_p = ctypes.POINTER(ctypes.c_double)

lib.create_mlp_model.restype = ctypes.c_void_p
lib.create_mlp_model.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
lib.set_mlp_activation.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.set_mlp_seed.argtypes = [ctypes.c_uint32]
lib.predict_mlp_model.restype = dbl_p
lib.predict_mlp_model.argtypes = [ctypes.c_void_p, dbl_p, ctypes.c_bool]
lib.train_mlp_model.argtypes = [ctypes.c_void_p, dbl_p, dbl_p, ctypes.c_int32, ctypes.c_int32,
                                ctypes.c_int32, ctypes.c_double, ctypes.c_int32, ctypes.c_bool]
lib.get_mlp_loss.restype = ctypes.c_double
lib.get_mlp_loss.argtypes = [ctypes.c_void_p, dbl_p, dbl_p, ctypes.c_int32, ctypes.c_int32,
                             ctypes.c_int32, ctypes.c_bool]
lib.destroy_mlp_model.argtypes = [ctypes.c_void_p]


def P(a):
    return a.ctypes.data_as(dbl_p)


def run(name, npl, X, Y, clf, act, alpha, epochs):
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64).reshape(len(X), -1)
    if clf and act == 1:
        Y = (Y > 0).astype(np.float64)  # {-1,1} -> {0,1} pour la sigmoïde
    lib.set_mlp_seed(42)
    arr = (ctypes.c_int32 * len(npl))(*npl)
    m = lib.create_mlp_model(arr, len(npl))
    lib.set_mlp_activation(m, act)
    n, f, t = len(X), X.shape[1], Y.shape[1]
    l0 = lib.get_mlp_loss(m, P(X), P(Y), n, f, t, clf)
    lib.train_mlp_model(m, P(X), P(Y), n, f, t, alpha, epochs, clf)
    l1 = lib.get_mlp_loss(m, P(X), P(Y), n, f, t, clf)
    preds = np.array([[lib.predict_mlp_model(m, P(X[k]), clf)[j] for j in range(t)] for k in range(n)])
    if clf:
        thr = 0.5 if act == 1 else 0.0
        if t == 1:
            score = np.mean((preds > thr) == (Y > thr))
        else:  # multi-classes : la sortie la plus forte gagne
            score = np.mean(np.argmax(preds, 1) == np.argmax(Y, 1))
        label = "précision"
    else:
        score = np.mean(np.abs(preds - Y))
        label = "erreur moy."
    print(f"  {name:26s} {str(npl):14s} loss {l0:7.4f} -> {l1:7.4f}   {label} = {score:.3f}")
    lib.destroy_mlp_model(m)


rng = np.random.default_rng(0)
cases = []
# --- Classification ---
cases.append(("Linear simple", [2, 1], [[1, 1], [2, 3], [3, 3]], [1, -1, -1], True, 0.05, 2000))
Xlm = np.concatenate([rng.random((50, 2)) * 0.9 + [1, 1], rng.random((50, 2)) * 0.9 + [2, 2]])
Ylm = np.concatenate([np.ones(50), -np.ones(50)])
cases.append(("Linear multiple", [2, 1], Xlm, Ylm, True, 0.05, 1000))
cases.append(("XOR", [2, 3, 1], [[1, 0], [0, 1], [0, 0], [1, 1]], [1, 1, -1, -1], True, 0.1, 5000))
Xc = rng.random((500, 2)) * 2 - 1
Yc = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in Xc])
cases.append(("Cross", [2, 4, 1], Xc, Yc, True, 0.05, 3000))
Xm = rng.random((500, 2)) * 2 - 1
Ym = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
               [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
               [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
               [-1, -1, -1] for p in Xm])
keep = [not np.all(r == [-1, -1, -1]) for r in Ym]
cases.append(("Multi linear 3 classes", [2, 3], Xm[keep], Ym[keep], True, 0.05, 2000))
Xx = rng.random((1000, 2)) * 2 - 1
Yx = np.array([[1, -1, -1] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else
               [-1, 1, -1] if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else
               [-1, -1, 1] for p in Xx])
cases.append(("Multi Cross", [2, 16, 16, 3], Xx, Yx, True, 0.01, 3000))
# --- Régression ---
cases.append(("Reg. linear simple 2D", [1, 1], [[1], [2]], [2, 3], False, 0.05, 2000))
cases.append(("Reg. non linear simple 2D", [1, 4, 1], [[1], [2], [3]], [2, 3, 2.5], False, 0.01, 5000))
cases.append(("Reg. linear simple 3D", [2, 1], [[1, 1], [2, 2], [3, 1]], [2, 3, 2.5], False, 0.01, 3000))
cases.append(("Reg. linear tricky 3D", [2, 1], [[1, 1], [2, 2], [3, 3]], [1, 2, 3], False, 0.01, 3000))
cases.append(("Reg. non linear simple 3D", [2, 4, 1], [[1, 0], [0, 1], [1, 1], [0, 0]], [2, 1, -2, -1], False, 0.01, 8000))

for act, nom in ((0, "TANH"), (1, "SIGMOÏDE")):
    print(f"\n=== Activation : {nom} ===")
    for name, npl, X, Y, clf, alpha, ep in cases:
        run(name, npl, X, Y, clf, act, alpha, ep)
