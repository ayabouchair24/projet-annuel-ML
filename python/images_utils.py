# Fichier de Nadine Zerguine : chargement et prétraitement des images (utilisé par tout le groupe)
import os

import numpy as np
from PIL import Image

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SEUIL_DOUBLON = 200.0


def lister_classes(dossier_split):
    classes = []
    for d in sorted(os.listdir(dossier_split)):
        chemin = os.path.join(dossier_split, d)
        if os.path.isdir(chemin) and any(f.lower().endswith(EXTENSIONS) for f in os.listdir(chemin)):
            classes.append(d)
    return classes


def _redimensionner(im, taille):
    im.draft("RGB", (max(taille, 256), max(taille, 256)))
    im = im.convert("RGB")
    return np.asarray(im.resize((taille, taille), Image.BOX), dtype=np.uint8)


def _charger_dossiers(dossiers, taille):
    X, y, chemins, tailles = [], [], [], []
    for c, d in enumerate(dossiers):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Dossier de classe introuvable : {d}")
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(EXTENSIONS):
                continue
            p = os.path.join(d, f)
            im = Image.open(p)
            tailles.append(im.size)
            X.append(_redimensionner(im, taille))
            y.append(c)
            chemins.append(p)
    return np.array(X), np.array(y), chemins, np.array(tailles)


def organisation(dataset_dir):
    if not os.path.isdir(dataset_dir):
        return None
    if os.path.isdir(os.path.join(dataset_dir, "train")) and os.path.isdir(os.path.join(dataset_dir, "test")):
        return "split/classe"
    for d in sorted(os.listdir(dataset_dir)):
        if os.path.isdir(os.path.join(dataset_dir, d, "train")):
            return "classe/split"
    return None


def dossier_images(dataset_dir, split, classe):
    if organisation(dataset_dir) == "split/classe":
        return os.path.join(dataset_dir, split, classe)
    return os.path.join(dataset_dir, classe, split)


def _contient_images(d):
    return os.path.isdir(d) and any(f.lower().endswith(EXTENSIONS) for f in os.listdir(d))


def lister_classes_dataset(dataset_dir):
    org = organisation(dataset_dir)
    if org is None:
        raise FileNotFoundError(f"Organisation du dataset non reconnue dans {dataset_dir} "
                                "(attendu : train/<classe>/ et test/<classe>/, ou <classe>/train/ et <classe>/test/)")
    if org == "split/classe":
        return lister_classes(os.path.join(dataset_dir, "train"))
    return [d for d in sorted(os.listdir(dataset_dir)) if _contient_images(os.path.join(dataset_dir, d, "train"))]


def charger_dataset(dataset_dir, split, classes, taille):
    return _charger_dossiers([dossier_images(dataset_dir, split, nom) for nom in classes], taille)


def vers_vecteurs(X_uint8):
    X = np.asarray(X_uint8, dtype=np.float64).reshape(len(X_uint8), -1) / 255.0
    return X * 2.0 - 1.0


def etiquettes_pm1(y, nb_classes):
    Y = -np.ones((len(y), nb_classes))
    Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return Y


def preprocess_image(image, taille):
    im = image if isinstance(image, Image.Image) else Image.open(image)
    return vers_vecteurs(_redimensionner(im, taille)[None, ...])


def distances(A, B):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.sqrt(np.maximum(d2, 0.0))


def distances_miroirs(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    VA = A.reshape(len(A), -1)
    variantes = [B, B[:, :, ::-1, :], B[:, ::-1, :, :], B[:, ::-1, ::-1, :]]
    D = None
    for v in variantes:
        d = distances(VA, v.reshape(len(v), -1))
        D = d if D is None else np.minimum(D, d)
    return D


def indices_uniques(X, seuil=SEUIL_DOUBLON, miroirs=False):
    X = np.asarray(X)
    D = distances_miroirs(X, X) if miroirs else distances(X.reshape(len(X), -1), X.reshape(len(X), -1))
    garde = []
    for i in range(len(X)):
        if not garde or np.all(D[i, garde] >= seuil):
            garde.append(i)
    return np.array(garde, dtype=int)


def nettoyer_train_test(Xtr, Xte, seuil=SEUIL_DOUBLON, miroirs=True):
    idx_tr = indices_uniques(Xtr, seuil, miroirs)
    idx_te = indices_uniques(Xte, seuil, miroirs)
    if miroirs:
        d_te_tr = distances_miroirs(np.asarray(Xte)[idx_te], np.asarray(Xtr)[idx_tr])
    else:
        d_te_tr = distances(np.asarray(Xte)[idx_te].reshape(len(idx_te), -1), np.asarray(Xtr)[idx_tr].reshape(len(idx_tr), -1))
    absent_du_train = d_te_tr.min(axis=1) >= seuil
    return idx_tr, idx_te[absent_du_train]


def separer_validation(y, fraction=0.2, graine=0):
    rng = np.random.default_rng(graine)
    y = np.asarray(y)
    fit, val = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = max(1, int(round(fraction * len(idx))))
        val += list(idx[:n_val])
        fit += list(idx[n_val:])
    return np.sort(np.array(fit)), np.sort(np.array(val))
