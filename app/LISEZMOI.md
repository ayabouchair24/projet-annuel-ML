# Dossier `app` : application client / serveur

Application de classification de fleurs (jasmin, jonquille, tournesol), sans Streamlit.

```
 index.html (le client, dans le navigateur) --- image + nom du modèle ---> serveur.py (l'API, Flask) ---> MLP / Linéaire / RBF
                                            <--- classe + scores --------                              (bibliothèque C++ + fichiers .npz)
```

## Fichiers
- `serveur.py` : l'API. Il charge la bibliothèque C++ et les modèles, envoie la page web et répond aux prédictions.
  - `GET /` : la page web `index.html`
  - `GET /modeles` : la liste des modèles
  - `POST /predict` : reçoit `modele` (texte) et `image` (fichier), renvoie `classe`, `classes` et `scores` en JSON
- `index.html` : le client. Une page simple : on choisit un modèle, on choisit une photo, on clique sur Classifier.

## Lancer (depuis le dossier PA)
1. Compiler la bibliothèque (une seule fois) :
   `cmake -S . -B build` puis `cmake --build build`
2. Installer les paquets : `pip install flask pillow numpy`
3. Lancer le serveur : `python app/serveur.py`
4. Ouvrir http://127.0.0.1:5000 dans le navigateur

## Tester l'API sans la page
`curl -F modele=MLP -F image=@une_fleur.jpg http://127.0.0.1:5000/predict`

## Remarques
- Les scores sont les sorties brutes du modèle (tanh pour le MLP), pas des probabilités.
- L'ancien `application/application.py` (Streamlit) est remplacé par ce dossier.
