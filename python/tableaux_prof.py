"""
Remplit les tableaux demandés par le prof (cas de test + dataset) pour le MLP.
Mêmes données, mêmes graines, mêmes paramètres que le notebook MLP : les résultats sont donc ceux du notebook.
Usage (depuis le dossier python/) :  python tableaux_prof.py
Écrit results/MLP/tableau_cas_de_test.csv et results/MLP/tableau_dataset.csv
Train Loss (MSE) : erreur quadratique moyenne sur le jeu d'entraînement (sigmoïde : cibles {0, 1} ; tanh : cibles {-1, 1}).
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mlp_wrapper import MLP
import images_utils as iu

RES = os.path.join(HERE, "..", "results", "MLP")
NOM = {"tanh": "Tanh", "sigmoid": "Sigmoïde"}


def ligne(cas, layers, act, seed, alpha, epochs, X, Y, Xt=None, Yt=None):
    m = MLP(layers, activation=act, task="classification", seed=seed)
    m.fit(X, Y, alpha=alpha, epochs=epochs, log_every=epochs)
    return [cas, f"{tuple(layers)} {NOM[act]}", seed, alpha, epochs, f"{m.loss(X, Y):.4f}",
            f"{m.score(X, Y):.1%}", "" if Xt is None else f"{m.score(Xt, Yt):.1%}"]


def cas_de_test():
    L = []
    X = np.array([[1, 1], [2, 3], [3, 3]]); Y = np.array([1, -1, -1])
    for act, s in (("tanh", 1), ("sigmoid", 0)):
        L.append(ligne("Linéaire simple", [2, 1], act, s, 0.1, 1000, X, Y))
    np.random.seed(0)
    X = np.concatenate([np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])
    for act in ("tanh", "sigmoid"):
        L.append(ligne("Modèle linéaire (multiple)", [2, 1], act, 0, 0.1, 500, X, Y))
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]]); Y = np.array([1, 1, -1, -1])
    for act in ("tanh", "sigmoid"):
        L.append(ligne("XOR", [2, 2, 1], act, 0, 0.3, 10000, X, Y))
    np.random.seed(0)
    def cross(n):
        X = np.random.random((n, 2)) * 2.0 - 1.0
        return X, np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])
    X, Y = cross(500); Xt, Yt = cross(200)
    for act, s in (("tanh", 0), ("sigmoid", 1)):
        L.append(ligne("Cross", [2, 4, 1], act, s, 0.05, 4000, X, Y, Xt, Yt))
    np.random.seed(0)
    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, -1, -1] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else
                  [-1, 1, -1] if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else
                  [-1, -1, 1] for p in X])
    for act, a in (("tanh", 0.02), ("sigmoid", 0.1)):
        L.append(ligne("Multi-cross", [2, 32, 32, 3], act, 1, a, 3000, X[:700], Y[:700], X[700:], Y[700:]))
    return L


def dataset():
    D = os.path.join(HERE, "..", "dataset")
    CL = iu.lister_classes_dataset(D); K = len(CL)
    Xtr, ytr, _, _ = iu.charger_dataset(D, "train", CL, 32)
    Xte, yte, _, _ = iu.charger_dataset(D, "test", CL, 32)
    idx_tr, idx_te = iu.nettoyer_train_test(Xtr, Xte)
    d_meme, d_autre = iu.plus_proches_tolerants(Xte, yte, Xtr, ytr)
    idx_te = idx_te[~(d_meme < 24.0)[idx_te]]
    A = iu.charger_dataset(D, "train", CL, 8)[0]; B = iu.charger_dataset(D, "test", CL, 8)[0]
    Xa, Xb = iu.vers_vecteurs(A[idx_tr]), iu.vers_vecteurs(B[idx_te])
    Ya, Yb = iu.etiquettes_pm1(ytr[idx_tr], K), iu.etiquettes_pm1(yte[idx_te], K)
    L = []
    for act, h, a in (("tanh", 32, 0.008), ("sigmoid", 32, 0.16)):
        L.append(ligne(f"Dataset fleurs ({len(Xa)} train / {len(Xb)} test)", [Xa.shape[1], h, K], act, 0, a, 21, Xa, Ya, Xb, Yb))
    return L


if __name__ == "__main__":
    tete = ["Cas de test", "Modèle / Structure", "Seed", "Learning rate", "Epoch", "Train Loss (MSE)", "Train Accuracy", "Test Accuracy"]
    for nom, f in (("tableau_cas_de_test.csv", cas_de_test), ("tableau_dataset.csv", dataset)):
        lignes = f()
        with open(os.path.join(RES, nom), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(tete); w.writerows(lignes)
        print("\n" + nom)
        print(" | ".join(tete))
        for l in lignes:
            print(" | ".join(str(x) for x in l))
