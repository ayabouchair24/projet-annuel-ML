"""
Tests du wrapper mlp_wrapper.py.
Usage : MLP_LIB=./libmlp.so python test_mlp_wrapper.py
"""
import os
import tempfile

import numpy as np

from mlp_wrapper import MLP

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'OK' if cond else 'ECHEC'}] {name} {detail}")


XOR_X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
XOR_Y = np.array([1, 1, -1, -1])

# 1. XOR avec chaque activation : le réseau apprend, la précision monte à 100 %
for act in ("tanh", "sigmoid"):
    m = MLP([2, 3, 1], activation=act, task="classification", seed=42)
    h = m.fit(XOR_X, XOR_Y, alpha=0.1, epochs=5000)
    check(f"XOR {act}", m.score(XOR_X, XOR_Y) == 1.0 and h["train_loss"][-1] < h["train_loss"][0],
          f"loss {h['train_loss'][0]:.3f} -> {h['train_loss'][-1]:.4f}, {len(h['epoch'])} points d'historique")

# 2. Nombre de poids = formule du cours : sum_l (d^(l-1) + 1) * d^l
m = MLP([2, 4, 3], seed=1)
check("nombre de poids", m.n_weights == (2 + 1) * 4 + (4 + 1) * 3, f"n_weights={m.n_weights}")

# 3. Régression non linéaire 2D (1,4,1) : sortie linéaire
mr = MLP([1, 4, 1], activation="tanh", task="regression", seed=3)
mr.fit(np.array([[1], [2], [3]]), np.array([2, 3, 2.5]), alpha=0.01, epochs=8000)
err = mr.score(np.array([[1], [2], [3]]), np.array([2, 3, 2.5]))
check("régression (1,4,1)", err < 0.1, f"erreur moyenne={err:.4f}")

# 4. Multi-classes (2,3) sur le cas "Multi linear 3 classes"
rng = np.random.default_rng(0)
X = rng.random((500, 2)) * 2 - 1
Y = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
              [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
              [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
              [-1, -1, -1] for p in X])
keep = ~np.all(Y == -1, axis=1)
X, Y = X[keep], Y[keep]
m3 = MLP([2, 3], task="classification", seed=5)
Xtr, Ytr, Xte, Yte = X[:150], Y[:150], X[150:], Y[150:]
h = m3.fit(Xtr, Ytr, alpha=0.05, epochs=1000, X_val=Xte, Y_val=Yte, log_every=100)
check("3 classes + validation", m3.score(Xte, Yte) > 0.95 and "val_loss" in h,
      f"précision test={m3.score(Xte, Yte):.3f}, clés historique={sorted(h)}")

# 5. Sauvegarde / chargement : prédictions strictement identiques
for act in ("tanh", "sigmoid"):
    m = MLP([2, 3, 1], activation=act, seed=42)
    m.fit(XOR_X, XOR_Y, alpha=0.1, epochs=2000)
    with tempfile.TemporaryDirectory() as d:
        path = m.save(os.path.join(d, "xor"))
        m2 = MLP.load(path)
    same = np.array_equal(m.predict(XOR_X), m2.predict(XOR_X))
    check(f"save/load {act}", same and m2.activation == act and m2.layers == [2, 3, 1],
          f"{m2}")

# 6. Sauvegarde / chargement d'un modèle de régression
with tempfile.TemporaryDirectory() as d:
    path = mr.save(os.path.join(d, "reg.npz"))
    mr2 = MLP.load(path)
check("save/load régression", np.array_equal(mr.predict([[1], [2], [3]]), mr2.predict([[1], [2], [3]])))

# 6b. Métadonnées sauvegardées avec le modèle (pour l'application)
with tempfile.TemporaryDirectory() as d:
    m = MLP([2, 3, 1], seed=1)
    path = m.save(os.path.join(d, "avec_meta"), meta={"classes": ["a", "b"], "taille": 16})
    m2 = MLP.load(path)
    path2 = m.save(os.path.join(d, "sans_meta"))
    m3 = MLP.load(path2)
check("save/load métadonnées", m2.meta == {"classes": ["a", "b"], "taille": 16} and m3.meta == {})

# 7. Protection contre les erreurs de dimension
try:
    MLP([2, 3, 1]).predict(np.zeros((4, 5)))
    check("erreur de dimension détectée", False)
except ValueError:
    check("erreur de dimension détectée", True)

# 8. Reproductibilité : même graine -> même résultat
a = MLP([2, 3, 1], seed=7); a.fit(XOR_X, XOR_Y, 0.1, 500)
b = MLP([2, 3, 1], seed=7); b.fit(XOR_X, XOR_Y, 0.1, 500)
check("reproductibilité (même graine)", np.array_equal(a.get_weights(), b.get_weights()))

print("\nTOUT EST OK" if ok else "\nIL Y A DES ECHECS")
