"""
images_utils.py : chargement, contrôle qualité et prétraitement des images du dataset.

Le MÊME prétraitement doit être utilisé à l'entraînement et dans l'application (API) :
    preprocess_image(...)  ->  vecteur d'entrée du MLP

Deux organisations de dataset sont reconnues automatiquement :
    « split / classe » :   dataset/train/<classe>/*.jpg     dataset/test/<classe>/*.jpg
    « classe / split » :   dataset/<classe>/train/*.jpg     dataset/<classe>/test/*.jpg
"""
import os

import numpy as np
from PIL import Image

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SEUIL_DOUBLON = 200.0  # distance (images 32x32x3, valeurs 0..255) en dessous de laquelle 2 images sont « les mêmes » (copie, recompression JPEG, miroir).
#                       Mesuré : copies et miroirs recompressés < ~100 ; deux images réellement différentes > ~800. Le seuil doit tomber dans ce vide
#                       (à vérifier sur l'histogramme du notebook, section 8.2, pour chaque nouveau dataset).


# ----------------------------------------------------------------------------------- chargement
def lister_classes(dossier_split):
    """
    Noms des classes = sous-dossiers CONTENANT AU MOINS UNE IMAGE, triés (l'ordre définit l'indice de classe).
    Un dossier vide (classe pas encore prête, ou seulement un fichier « a_supprimer ») est ignoré.
    """
    classes = []
    for d in sorted(os.listdir(dossier_split)):
        chemin = os.path.join(dossier_split, d)
        if os.path.isdir(chemin) and any(f.lower().endswith(EXTENSIONS) for f in os.listdir(chemin)):
            classes.append(d)
    return classes


def _redimensionner(im, taille):
    im.draft("RGB", (max(taille, 256), max(taille, 256)))  # décodage JPEG plus rapide sur les très grandes images
    im = im.convert("RGB")
    return np.asarray(im.resize((taille, taille), Image.BOX), dtype=np.uint8)


def _charger_dossiers(dossiers, taille):
    """Charge les images de chaque dossier (un dossier = une classe, dans l'ordre) et les redimensionne."""
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


def charger_split(dossier_split, classes, taille):
    """
    Charge toutes les images d'un split organisé « split / classe » (dossier_split/<classe>/*.jpg) et les redimensionne en
    taille x taille (RGB). Retourne : X (n, taille, taille, 3) uint8, y (n,) indices de classe, chemins (liste), tailles_origine (n, 2).
    """
    return _charger_dossiers([os.path.join(dossier_split, nom) for nom in classes], taille)


# ----------------------------------------------------------------------------------- organisation du dataset (2 formats)
def organisation(dataset_dir):
    """
    Détecte comment le dataset est rangé :
      "split/classe" : dataset_dir/train/<classe>/   et   dataset_dir/test/<classe>/
      "classe/split" : dataset_dir/<classe>/train/   et   dataset_dir/<classe>/test/
      None           : ni l'un ni l'autre.
    """
    if not os.path.isdir(dataset_dir):
        return None
    if os.path.isdir(os.path.join(dataset_dir, "train")) and os.path.isdir(os.path.join(dataset_dir, "test")):
        return "split/classe"
    for d in sorted(os.listdir(dataset_dir)):
        if os.path.isdir(os.path.join(dataset_dir, d, "train")):
            return "classe/split"
    return None


def dossier_images(dataset_dir, split, classe):
    """Chemin du dossier d'images d'une classe pour un split ("train" ou "test"), quelle que soit l'organisation."""
    if organisation(dataset_dir) == "split/classe":
        return os.path.join(dataset_dir, split, classe)
    return os.path.join(dataset_dir, classe, split)


def _contient_images(d):
    return os.path.isdir(d) and any(f.lower().endswith(EXTENSIONS) for f in os.listdir(d))


def lister_classes_dataset(dataset_dir):
    """Classes du dataset (triées) = celles qui ont au moins une image dans leur dossier train. Fonctionne pour les deux organisations."""
    org = organisation(dataset_dir)
    if org is None:
        raise FileNotFoundError(f"Organisation du dataset non reconnue dans {dataset_dir} "
                                "(attendu : train/<classe>/ et test/<classe>/, ou <classe>/train/ et <classe>/test/)")
    if org == "split/classe":
        return lister_classes(os.path.join(dataset_dir, "train"))
    return [d for d in sorted(os.listdir(dataset_dir)) if _contient_images(os.path.join(dataset_dir, d, "train"))]


def charger_dataset(dataset_dir, split, classes, taille):
    """Comme charger_split, mais à partir de la racine du dataset et du nom du split ("train" ou "test"), pour les deux organisations."""
    return _charger_dossiers([dossier_images(dataset_dir, split, nom) for nom in classes], taille)


# ----------------------------------------------------------------------------------- prétraitement
def vers_vecteurs(X_uint8):
    """(n, t, t, 3) uint8 -> (n, t*t*3) réels dans [-1, 1] (les pixels 0..255 sont recentrés)."""
    X = np.asarray(X_uint8, dtype=np.float64).reshape(len(X_uint8), -1) / 255.0
    return X * 2.0 - 1.0


def etiquettes_pm1(y, nb_classes):
    """Indices de classe -> matrice (n, nb_classes) en {-1, 1} : 1 pour la bonne classe, -1 ailleurs."""
    Y = -np.ones((len(y), nb_classes))
    Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return Y


def preprocess_image(image, taille):
    """
    Prétraitement d'UNE image (chemin, objet PIL ou fichier ouvert) pour l'application / l'API.
    Retourne un tableau (1, taille*taille*3), exactement comme à l'entraînement.
    """
    im = image if isinstance(image, Image.Image) else Image.open(image)
    return vers_vecteurs(_redimensionner(im, taille)[None, ...])


# ----------------------------------------------------------------------------------- qualité du dataset
def distances(A, B):
    """Distances euclidiennes entre toutes les paires de lignes de A et de B."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.sqrt(np.maximum(d2, 0.0))


def distances_miroirs(A, B):
    """
    Comme distances(A, B) mais une image de B compte aussi si elle apparaît RETOURNÉE (miroir horizontal, vertical ou les deux).
    A et B : tableaux d'images (n, t, t, 3). Retourne, pour chaque paire, la plus petite des 4 distances.
    Sert à repérer les doublons « cachés » créés par augmentation de données (images retournées).
    """
    A = np.asarray(A)
    B = np.asarray(B)
    VA = A.reshape(len(A), -1)
    variantes = [B, B[:, :, ::-1, :], B[:, ::-1, :, :], B[:, ::-1, ::-1, :]]   # original, miroir horizontal, vertical, les deux
    D = None
    for v in variantes:
        d = distances(VA, v.reshape(len(v), -1))
        D = d if D is None else np.minimum(D, d)
    return D


def indices_uniques(X, seuil=SEUIL_DOUBLON, miroirs=False):
    """
    Indices à conserver pour ne garder qu'un exemplaire de chaque groupe d'images identiques.
    miroirs=True : une image retournée compte comme un doublon (X doit alors être de forme (n, t, t, 3)).
    """
    X = np.asarray(X)
    D = distances_miroirs(X, X) if miroirs else distances(X.reshape(len(X), -1), X.reshape(len(X), -1))
    garde = []
    for i in range(len(X)):
        if not garde or np.all(D[i, garde] >= seuil):
            garde.append(i)
    return np.array(garde, dtype=int)


def nettoyer_train_test(Xtr, Xte, seuil=SEUIL_DOUBLON, miroirs=True):
    """
    Retourne (idx_train, idx_test) :
      - train : une seule copie de chaque image ;
      - test  : une seule copie de chaque image ET aucune image déjà présente dans le train (sinon le test est faussé).
    Xtr, Xte : images redimensionnées (n, t, t, 3), de taille fixe (par ex. 32x32).
    miroirs=True : une image retournée (miroir) compte comme la même image.
    """
    idx_tr = indices_uniques(Xtr, seuil, miroirs)
    idx_te = indices_uniques(Xte, seuil, miroirs)
    if miroirs:
        d_te_tr = distances_miroirs(np.asarray(Xte)[idx_te], np.asarray(Xtr)[idx_tr])
    else:
        d_te_tr = distances(np.asarray(Xte)[idx_te].reshape(len(idx_te), -1), np.asarray(Xtr)[idx_tr].reshape(len(idx_tr), -1))
    absent_du_train = d_te_tr.min(axis=1) >= seuil
    return idx_tr, idx_te[absent_du_train]


def separer_validation(y, fraction=0.2, graine=0):
    """
    Sépare des indices en (apprentissage, validation) en gardant les proportions de chaque classe.
    La validation sert à RÉGLER les hyperparamètres ; le jeu de test n'est utilisé qu'à la fin.
    """
    rng = np.random.default_rng(graine)
    y = np.asarray(y)
    fit, val = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = max(1, int(round(fraction * len(idx))))
        val += list(idx[:n_val])
        fit += list(idx[n_val:])
    return np.sort(np.array(fit)), np.sort(np.array(val))


def equilibrer_par_repetition(X, y):
    """
    Rééquilibre les classes en répétant les images des classes minoritaires (jusqu'à ~ la taille de la plus grande classe).
    À n'appliquer qu'au jeu d'APPRENTISSAGE. Retourne (X_equilibre, y_equilibre).
    """
    y = np.asarray(y)
    effectifs = np.bincount(y)
    cible = effectifs.max()
    indices = []
    for c in range(len(effectifs)):
        ic = np.where(y == c)[0]
        indices += list(np.tile(ic, max(1, int(round(cible / len(ic))))))
    indices = np.array(indices)
    return X[indices], y[indices]


def coins_noirs(X, seuil=6, min_coins=4):
    """
    Repère les images dont les coins sont quasi noirs. Typique d'une augmentation par ROTATION avec remplissage noir
    (une rotation noircit les 4 coins). X : (n, t, t, 3) uint8. Retourne un tableau de booléens (n,).
    Critère : au moins min_coins coins (pixel de coin) dont les 3 canaux sont <= seuil. Mesuré sur des photos de fleurs :
    100 % des images tournées de 5° ou plus sont repérées, 0 à 2 % des photos d'origine (fonds naturellement sombres) à tort.
    """
    X = np.asarray(X)
    coins = np.stack([X[:, 0, 0], X[:, 0, -1], X[:, -1, 0], X[:, -1, -1]], axis=1)     # (n, 4, 3)
    return (coins <= seuil).all(axis=2).sum(axis=1) >= min_coins


# ----------------------------------------------------------------------------------- copies tournées / retournées (augmentation)
ANGLES_TOL = (-20, -15, -10, -5, 0, 5, 10, 15, 20)


def _reduire(X, t_cmp):
    """(n, t, t, 3) -> (n, t_cmp, t_cmp, 3) en moyennant des blocs (t doit être un multiple de t_cmp)."""
    n, t = X.shape[0], X.shape[1]
    f = t // t_cmp
    return X.reshape(n, t_cmp, f, t_cmp, f, 3).mean(axis=(2, 4))


def _transformations(X, angles, miroirs):
    """Toutes les versions de X (n, t, t, 3) uint8 : chaque miroir (aucun / horizontal / vertical / les deux) x chaque rotation (fond noir)."""
    bases = [X, X[:, :, ::-1], X[:, ::-1, :], X[:, ::-1, ::-1]] if miroirs else [X]
    for B in bases:
        for a in angles:
            if a == 0:
                yield B
            else:
                yield np.array([np.asarray(Image.fromarray(np.ascontiguousarray(x)).rotate(a, resample=Image.BILINEAR, fillcolor=(0, 0, 0)))
                                for x in B])


def distances_rms_tolerantes(A, B, angles=ANGLES_TOL, miroirs=True, t_cmp=16, rayon=0.45):
    """
    Matrice (len(A), len(B)) de distances « RMS » (écart quadratique moyen par valeur de pixel, échelle 0..255) entre chaque image de A et
    chaque image de B, en autorisant que l'image de A ait été TOURNÉE (angles) et/ou RETOURNÉE (miroirs) : on garde la plus petite distance.
    Sert à repérer les copies créées par augmentation. La comparaison se fait sur un disque central (rayon = fraction de l'image) pour
    ignorer les coins noirs laissés par les rotations. A et B : (n, t, t, 3) uint8, t multiple de t_cmp (par ex. 32 et 16).
    """
    A = np.asarray(A)
    B = np.asarray(B)
    yy, xx = np.mgrid[:t_cmp, :t_cmp]
    c = (t_cmp - 1) / 2.0
    masque = np.repeat(((yy - c) ** 2 + (xx - c) ** 2 <= (rayon * t_cmp) ** 2)[..., None], 3, axis=2).reshape(-1)
    nvals = float(masque.sum())

    def vec(X):
        return _reduire(X, t_cmp).reshape(len(X), -1).astype(np.float32)[:, masque]

    VB = vec(B)
    nb = (VB ** 2).sum(axis=1)
    D = None
    for T in _transformations(A, angles, miroirs):
        VA = vec(T)
        d2 = (VA ** 2).sum(axis=1)[:, None] + nb[None, :] - 2.0 * VA @ VB.T
        d = np.sqrt(np.maximum(d2, 0.0) / nvals)
        D = d if D is None else np.minimum(D, d)
    return D


def plus_proches_tolerants(Xte, yte, Xtr, ytr, **kw):
    """
    Pour chaque image de test : distance RMS tolérante à l'image de train la plus proche DE LA MÊME CLASSE (d_meme) et d'une AUTRE CLASSE (d_autre).
    d_autre sert de référence : une copie ne peut pas exister entre classes différentes, donc d_autre montre à quoi ressemblent des images
    simplement « semblables » mais différentes, et aide à fixer le seuil de copie.
    """
    yte, ytr = np.asarray(yte), np.asarray(ytr)
    d_meme, d_autre = np.zeros(len(Xte)), np.zeros(len(Xte))
    for c in np.unique(yte):
        it, jt = np.where(yte == c)[0], np.where(ytr == c)[0]
        jo = np.where(ytr != c)[0]
        d_meme[it] = distances_rms_tolerantes(np.asarray(Xte)[it], np.asarray(Xtr)[jt], **kw).min(axis=1)
        d_autre[it] = distances_rms_tolerantes(np.asarray(Xte)[it], np.asarray(Xtr)[jo], **kw).min(axis=1)
    return d_meme, d_autre


def groupes_de_copies(X, y, seuil, **kw):
    """
    Numéro de groupe de chaque image : deux images d'une même classe sont dans le même groupe si l'une est une copie (rotation / miroir)
    de l'autre (distance RMS tolérante < seuil), directement ou de proche en proche.
    """
    X, y = np.asarray(X), np.asarray(y)
    parent = np.arange(len(X))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        D = distances_rms_tolerantes(X[idx], X[idx], **kw)
        D = np.minimum(D, D.T)
        ii, jj = np.where(np.triu(D < seuil, k=1))
        for a, b in zip(ii, jj):
            parent[find(idx[a])] = find(idx[b])
    return np.array([find(i) for i in range(len(X))])


def separer_validation_groupes(y, groupes, fraction=0.2, graine=0):
    """
    Comme separer_validation, mais un GROUPE de copies entier va soit dans l'apprentissage, soit dans la validation : la validation ne contient
    ainsi aucune copie d'une image de l'apprentissage (sinon elle est trop optimiste).
    """
    rng = np.random.default_rng(graine)
    y, groupes = np.asarray(y), np.asarray(groupes)
    fit, val = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        gs = rng.permutation(np.unique(groupes[idx]))
        cible, n_val = fraction * len(idx), 0
        for g in gs:
            membres = idx[groupes[idx] == g]
            if n_val < cible:
                val += list(membres)
                n_val += len(membres)
            else:
                fit += list(membres)
    return np.sort(np.array(fit, dtype=int)), np.sort(np.array(val, dtype=int))


def seuil_copie_auto(d_meme, d_autre, largeur=3.0, min_pic=8):
    """
    Seuil de copie choisi dans la VALLÉE de l'histogramme des distances « vers la même classe » : entre la bosse des vraies copies (distances
    faibles) et la masse des images simplement différentes. Démarche : on mesure, pour chaque distance t, le nombre d'images dans [t - largeur,
    t + largeur] ; on repère le pic (la bosse des copies, dans la partie basse) ; à droite du pic, on prend l'endroit le moins peuplé (le milieu s'il y a un vide).
    On ne dépasse pas 0,9 x le 1er centile des distances vers d'AUTRES classes (une copie n'existe jamais entre classes : au-delà, on prendrait
    des images différentes pour des copies). Sans bosse nette (pic < min_pic images), on retient 0,5 x ce 1er centile, par prudence.
    """
    d_meme = np.asarray(d_meme)
    p1 = float(np.percentile(d_autre, 1))
    candidats = np.arange(2.0, 0.9 * p1, 0.25)
    densite = np.array([np.sum((d_meme > t - largeur) & (d_meme < t + largeur)) for t in candidats])
    # le pic des copies se cherche dans la partie BASSE (jusqu'à 0,6 x p1) : sinon, sans copies, la montée de la masse d'images différentes
    # près de la limite serait prise pour une bosse de copies
    bas = np.where(candidats <= 0.6 * p1)[0]
    if len(bas) == 0:
        return 0.5 * p1
    i_pic = int(bas[np.argmax(densite[bas])])
    if densite[i_pic] < min_pic:
        return 0.5 * p1
    droite, d_droite = candidats[i_pic:], densite[i_pic:]
    return float(np.median(droite[d_droite == d_droite.min()]))
