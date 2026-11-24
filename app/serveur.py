# Fichier de Nadine Zerguine : serveur de l'API (Flask) qui héberge les modèles pré-entraînés
# et qui envoie la page web index.html (le client).
# Le MLP utilise mlp_wrapper.py et images_utils.py (même prétraitement qu'à l'entraînement).
# Les fonctions du modèle linéaire (Assia), SVM et RBF (Aya) reprennent le code de l'ancien application.py de l'équipe.
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory

BASE = Path(__file__).resolve().parent
RACINE = BASE.parent
MODELES = RACINE / "models"

sys.path.insert(0, str(RACINE / "python"))
import images_utils as iu
import mlp_wrapper as mw
from mlp_wrapper import MLP

# ---------- La bibliothèque C++ (on prend la première qui existe) ----------
CHEMINS = [BASE / "libPA_V2.dll",
           RACINE / "build" / "libPA_V2.dylib",
           RACINE / "build" / "libPA_V2.so",
           RACINE / "build" / "libPA_V2.dll"]
chemin_lib = None
for c in CHEMINS:
    if c.exists():
        chemin_lib = c
        break
if chemin_lib is None:
    raise FileNotFoundError("libPA_V2 introuvable : compile le projet avec CMake (voir LISEZMOI.md)")

mw.load_library(str(chemin_lib))
lib = ctypes.CDLL(str(chemin_lib))
double_ptr = ctypes.POINTER(ctypes.c_double)

# ---------- MLP : chargé une seule fois au démarrage du serveur ----------
modele_mlp = MLP.load(MODELES / "MLP" / "mlp_tanh.npz")


def predire_mlp(image):
    taille = modele_mlp.meta["taille_image"]
    x = iu.preprocess_image(image, taille)
    scores = modele_mlp.predict(x)[0]
    classes = modele_mlp.meta["classes"]
    return classes[int(np.argmax(scores))], scores, classes


# ---------- Modèle linéaire (repris de application.py) ----------
lib.create_linear_model.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
lib.create_linear_model.restype = ctypes.c_void_p
lib.set_weights.argtypes = [ctypes.c_void_p, double_ptr]
lib.set_weights.restype = None
lib.predict_scores.argtypes = [ctypes.c_void_p, double_ptr, ctypes.c_int32, double_ptr]
lib.predict_scores.restype = None
lib.delete_linear_model.argtypes = [ctypes.c_void_p]
lib.delete_linear_model.restype = None


def predire_lineaire(image):
    with np.load(MODELES / "modele_lineaire" / "modele_lineaire.npz") as data:
        poids = np.ascontiguousarray(data["w"], dtype=np.float64)
        mu = np.asarray(data["mu"], dtype=np.float64)
        sd = np.asarray(data["sd"], dtype=np.float64)
        classes = data["classes"].tolist()
        taille = int(data["size"])

    image = image.convert("RGB").resize((64, 64), Image.Resampling.BOX)
    x = np.asarray(image, dtype=np.float64)
    facteur = 64 // taille
    x = x.reshape(taille, facteur, taille, facteur, 3).mean(axis=(1, 3))
    x = (x / 255.0).reshape(-1)
    x = np.ascontiguousarray((x - mu) / sd, dtype=np.float64)

    n_entrees, n_sorties = len(x), len(classes)
    poids = np.ascontiguousarray(poids.ravel(), dtype=np.float64)
    modele = lib.create_linear_model(n_entrees, n_sorties, 42)
    lib.set_weights(modele, poids.ctypes.data_as(double_ptr))
    scores = np.zeros(n_sorties, dtype=np.float64)
    lib.predict_scores(modele, x.ctypes.data_as(double_ptr), 1, scores.ctypes.data_as(double_ptr))
    lib.delete_linear_model(modele)
    return classes[int(np.argmax(scores))], scores, classes


# ---------- SVM et RBF (fichiers .npz exportés depuis les notebooks, repris de application.py) ----------
deja_charges = {}


def charger_modele(algo):
    if algo in deja_charges:
        return deja_charges[algo]
    nom_fichier = "svm.npz" if algo == "SVM" else "rbf.npz"
    with np.load(MODELES / algo / nom_fichier, allow_pickle=False) as data:
        m = {k: data[k].copy() for k in data.files}
    m["config"] = json.loads(str(m["config"]))
    m["classes"] = m["classes"].tolist()
    m["taille"] = int(m["taille"])
    deja_charges[algo] = m
    return m


def predire_svm_rbf(image, algo):
    m = charger_modele(algo)
    x = iu.preprocess_image(image, m["taille"])[0]
    gamma = float(m["config"]["gamma"])
    scores = []
    for c in range(len(m["classes"])):
        if algo == "SVM":
            sv = m[f"sv_{c}"]
            coeffs = m[f"alphasY_{c}"]
            biais = float(m[f"bias_{c}"])
            noyau = int(m["config"]["kernel"])
            if noyau == 0:      # linéaire
                kv = sv @ x
            elif noyau == 1:    # polynomial
                kv = np.power(1.0 + sv @ x, int(m["config"]["degree"]))
            else:               # RBF
                kv = np.exp(-gamma * np.sum((sv - x) ** 2, axis=1))
            score = biais + np.dot(coeffs, kv)
        else:
            centres = m[f"centers_{c}"]
            poids = m[f"weights_{c}"]
            score = np.dot(poids, np.exp(-gamma * np.sum((centres - x) ** 2, axis=1)))
        scores.append(float(score))
    return m["classes"][int(np.argmax(scores))], np.array(scores), m["classes"]


# ---------- L'API ----------
app = Flask(__name__)


@app.route("/")
def accueil():
    # la page web (le client) est servie par le serveur
    return send_from_directory(BASE, "index.html")


@app.route("/modeles")
def liste_modeles():
    return jsonify({"modeles": ["MLP", "Linear Model", "SVM", "RBF"]})


@app.route("/predict", methods=["POST"])
def predict():
    nom = request.form.get("modele", "MLP")
    fichier = request.files.get("image")
    if fichier is None:
        return jsonify({"erreur": "aucune image reçue"}), 400

    try:
        image = Image.open(fichier.stream)
        if nom == "MLP":
            classe, scores, classes = predire_mlp(image)
        elif nom == "Linear Model":
            classe, scores, classes = predire_lineaire(image)
        elif nom in ("SVM", "RBF"):
            classe, scores, classes = predire_svm_rbf(image, nom)
        else:
            return jsonify({"erreur": "modèle inconnu : " + nom}), 400
    except Exception as e:
        return jsonify({"erreur": str(e)}), 500

    return jsonify({"modele": nom, "classe": classe, "classes": classes,
                    "scores": [float(s) for s in scores]})


if __name__ == "__main__":
    app.run(port=5000)
