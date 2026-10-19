"""
Tests de images_utils.py sur des images SYNTHÉTIQUES (aucun dataset nécessaire).
Usage : python test_images_utils.py
"""
import os
import tempfile

import numpy as np
from PIL import Image

import images_utils as iu

ok = True


def check(nom, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'OK' if cond else 'ECHEC'}] {nom} {detail}")


rng = np.random.default_rng(0)
with tempfile.TemporaryDirectory() as d:
    # dataset jouet : 2 classes ; la classe "b" contient des doublons ; une image du test est aussi dans le train
    imgs = {}
    for cls, n in (("a", 6), ("b", 4)):
        os.makedirs(f"{d}/train/{cls}"); os.makedirs(f"{d}/test/{cls}")
        for k in range(n):
            imgs[(cls, k)] = rng.integers(0, 256, (50 + 7 * k, 40, 3), dtype=np.uint8)
            Image.fromarray(imgs[(cls, k)]).save(f"{d}/train/{cls}/{k}.jpg", quality=100)
    Image.fromarray(imgs[("b", 0)]).save(f"{d}/train/b/copie.jpg", quality=100)     # doublon dans le train
    Image.fromarray(imgs[("a", 1)]).save(f"{d}/test/a/vue.jpg", quality=100)         # image du test déjà dans le train
    # image « de type photo » (lisse, asymétrique) dans le train ; sa version RETOURNÉE est dans le test (augmentation cachée).
    # (une image de bruit ne convient pas : la compression JPEG détruit son contenu et sa version retournée ne lui ressemble plus)
    lisse = np.asarray(Image.fromarray(rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)).resize((64, 64), Image.BICUBIC))
    Image.fromarray(lisse).save(f"{d}/train/a/lisse.jpg", quality=95)
    Image.fromarray(lisse[:, ::-1, :]).save(f"{d}/test/b/retournee.jpg", quality=95)
    Image.fromarray(rng.integers(0, 256, (60, 60, 3), dtype=np.uint8)).save(f"{d}/test/a/neuve.jpg")
    Image.fromarray(rng.integers(0, 256, (60, 60, 3), dtype=np.uint8)).save(f"{d}/test/b/neuve.jpg")
    open(f"{d}/train/a/a_supprimer", "w").close()                                    # fichier parasite à ignorer
    os.makedirs(f"{d}/train/vide"); open(f"{d}/train/vide/a_supprimer", "w").close()  # classe pas encore prête : dossier sans image

    classes = iu.lister_classes(f"{d}/train")
    Xtr, ytr, chem, tailles = iu.charger_split(f"{d}/train", classes, 16)
    Xte, yte, _, _ = iu.charger_split(f"{d}/test", classes, 16)
    check("classes triées, dossier vide ignoré", classes == ["a", "b"], str(classes))
    check("chargement : forme et fichier parasite ignoré", Xtr.shape == (12, 16, 16, 3) and len(ytr) == 12, f"{Xtr.shape}")
    check("tailles d'origine lues", tuple(tailles[0]) == (40, 50), str(tuple(tailles[0])))

    V = iu.vers_vecteurs(Xtr)
    check("vecteurs : forme et plage [-1, 1]", V.shape == (12, 16 * 16 * 3) and V.min() >= -1 and V.max() <= 1)
    Y = iu.etiquettes_pm1(ytr, 2)
    check("étiquettes {-1, 1}", set(np.unique(Y)) == {-1.0, 1.0} and np.all(Y.sum(axis=1) == 0))

    # le prétraitement serveur doit donner EXACTEMENT les mêmes valeurs que l'entraînement
    x = iu.preprocess_image(chem[3], 16)
    check("preprocess_image == prétraitement d'entraînement", x.shape == (1, 768) and np.array_equal(x[0], V[3]))

    idx_tr, idx_te = iu.nettoyer_train_test(Xtr, Xte)
    check("doublon du train retiré", len(idx_tr) == 11, f"{len(idx_tr)} images conservées sur 12")
    check("images du test déjà vues (identique OU retournée) retirées", len(idx_te) == 2, f"{len(idx_te)} images de test conservées sur 4")
    _, idx_te_sans_miroir = iu.nettoyer_train_test(Xtr, Xte, miroirs=False)
    check("sans l'option miroirs, l'image retournée n'est PAS détectée", len(idx_te_sans_miroir) == 3, f"{len(idx_te_sans_miroir)} conservées")

    # --- deuxième organisation : dataset/<classe>/train et dataset/<classe>/test ---
    import shutil
    d2 = os.path.join(d, "classe_d_abord")
    for split in ("train", "test"):
        for cls in ("a", "b", "vide"):
            if os.path.isdir(f"{d}/{split}/{cls}"):
                shutil.copytree(f"{d}/{split}/{cls}", f"{d2}/{cls}/{split}")
    check("organisation « classe/split » reconnue", iu.organisation(d2) == "classe/split", str(iu.organisation(d2)))
    cl2 = iu.lister_classes_dataset(d2)
    check("classes (classe/split), dossier vide ignoré", cl2 == ["a", "b"], str(cl2))
    X2, y2, _, _ = iu.charger_dataset(d2, "train", cl2, 16)
    X2t, y2t, _, _ = iu.charger_dataset(d2, "test", cl2, 16)
    check("mêmes images dans les deux organisations", np.array_equal(X2, Xtr) and np.array_equal(y2, ytr)
          and np.array_equal(X2t, Xte) and np.array_equal(y2t, yte), f"train {X2.shape}, test {X2t.shape}")
    d1 = os.path.join(d, "split_d_abord"); os.makedirs(d1)
    for split in ("train", "test"):
        shutil.copytree(f"{d}/{split}", f"{d1}/{split}")
    check("organisation « split/classe » reconnue", iu.organisation(d1) == "split/classe", str(iu.organisation(d1)))
    X1, y1, _, _ = iu.charger_dataset(d1, "train", iu.lister_classes_dataset(d1), 16)
    check("charger_dataset == charger_split (split/classe)", np.array_equal(X1, Xtr))
    try:
        iu.lister_classes_dataset(os.path.join(d, "n_existe_pas"))
        check("dataset inexistant -> erreur claire", False)
    except FileNotFoundError:
        check("dataset inexistant -> erreur claire", True)

    # --- coins noirs (rotation avec remplissage noir) ---
    clair = np.full((64, 64, 3), 140, dtype=np.uint8) + (np.arange(64)[None, :, None] // 4).astype(np.uint8)
    tournee = np.asarray(Image.fromarray(clair).rotate(10, fillcolor=(0, 0, 0)))
    sombre = clair.copy(); sombre[:20, :20] = 0                                    # un seul coin noir (photo sombre)
    lot = np.array([clair, tournee, sombre])
    cn = iu.coins_noirs(lot)
    check("coins noirs : image tournée repérée, image normale et un seul coin sombre non", list(cn) == [False, True, False], str(list(cn)))

    Xe, ye = iu.equilibrer_par_repetition(Xtr[idx_tr], ytr[idx_tr])
    n = np.bincount(ye)
    check("rééquilibrage", n.min() >= 0.6 * n.max(), f"effectifs {n.tolist()} (avant : {np.bincount(ytr[idx_tr]).tolist()})")

    fit, val = iu.separer_validation(ytr[idx_tr], 0.3, 0)
    check("validation : disjointe, toutes classes présentes",
          not set(fit) & set(val) and set(ytr[idx_tr][val]) == {0, 1} and len(fit) + len(val) == len(idx_tr))

# ---------------------------------------------------------------------------------------------- copies tournées / retournées
rng2 = np.random.default_rng(5)
def photo(seed):
    r = np.random.default_rng(seed)
    return np.asarray(Image.fromarray(r.integers(0, 256, (6, 6, 3), dtype=np.uint8)).resize((32, 32), Image.BICUBIC))
def copie(img, angle, miroir=False):
    im = Image.fromarray(img).resize((256, 256), Image.BICUBIC).rotate(angle, fillcolor=(0, 0, 0))
    a = np.asarray(im)[:, ::-1] if miroir else np.asarray(im)
    return np.asarray(Image.fromarray(np.ascontiguousarray(a)).resize((32, 32), Image.BOX))
B = [photo(s) for s in range(6)]
train = np.array([copie(B[0], 10, True), copie(B[1], -7), B[2], B[3], B[4]])
ytrain = np.zeros(5, dtype=int)
test = np.array([B[0], B[1], photo(99)])
ytest = np.zeros(3, dtype=int)
# classe 1 (références « autres classes ») : images quelconques
train = np.concatenate([train, np.array([photo(200 + k) for k in range(4)])]); ytrain = np.concatenate([ytrain, np.ones(4, dtype=int)])
d_meme, d_autre = iu.plus_proches_tolerants(test, ytest, train, ytrain)
check("copie tournée+miroir et copie tournée repérées (distance faible)", d_meme[0] < 15 and d_meme[1] < 15, f"d = {d_meme[0]:.1f}, {d_meme[1]:.1f}")
check("image différente non repérée (distance nettement plus grande)", d_meme[2] > 2 * max(d_meme[0], d_meme[1]) and d_meme[2] > 25, f"d = {d_meme[2]:.1f}")
Xg = np.array([B[0], copie(B[0], 12), B[1], copie(B[1], -8, True), B[2]])
g = iu.groupes_de_copies(Xg, np.zeros(5, dtype=int), seuil=15)
check("groupes de copies (0,1) et (2,3), image 4 seule", g[0] == g[1] and g[2] == g[3] and len({g[0], g[2], g[4]}) == 3, str(list(g)))
yv = np.zeros(40, dtype=int); gv = np.repeat(np.arange(10), 4)           # 10 groupes de 4 copies
f, v = iu.separer_validation_groupes(yv, gv, 0.25, 0)
check("validation par groupes : aucun groupe coupé en deux", not set(gv[f]) & set(gv[v]) and len(f) + len(v) == 40 and len(v) >= 10, f"fit {len(f)} / val {len(v)}")

# seuil automatique : dans la vallée entre la bosse des copies et la masse des images différentes
rng3 = np.random.default_rng(3)
def tirer(rng, loc, scale, minimum, n):
    v = rng.normal(loc, scale, 5 * n)
    return v[v > minimum][:n]
copies = rng3.uniform(2, 20, 80)                                    # bosse de copies
masse = tirer(rng3, 45, 7, 31, 600)                                 # images simplement différentes
d_m = np.concatenate([copies, [23.0, 25.0], masse])
d_a = tirer(rng3, 48, 8, 27, len(d_m))
sc = iu.seuil_copie_auto(d_m, d_a)
check("seuil automatique dans la vallée (copies dessous, images différentes dessus)", bool((copies < sc).all() and (masse > sc).all()), f"seuil = {sc:.1f}")
sc0 = iu.seuil_copie_auto(masse, d_a[:len(masse)])
check("sans bosse de copies : seuil prudent (0,5 x 1er centile)", abs(sc0 - 0.5 * np.percentile(d_a[:len(masse)], 1)) < 1e-9, f"seuil = {sc0:.1f}")

print("\nTOUT EST OK" if ok else "\nIL Y A DES ECHECS")
