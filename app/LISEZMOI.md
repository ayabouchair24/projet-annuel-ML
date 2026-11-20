# Dossier `app` : application client / serveur (API)

Partie de Nadine Zerguine. Les modèles ne sont plus dans l'interface : ils sont hébergés par un **serveur** (API) et le **client** les appelle.

```
 client.py (Streamlit)  --- image + nom du modèle --->  serveur.py (Flask)  ---> MLP / Linéaire / SVM / RBF
        (interface)     <--- classe + scores ---------        (API)              (bibliothèque C++ + fichiers .npz)
```

## Fichiers
- `serveur.py` : l'API. Au démarrage il charge la bibliothèque C++ et le MLP (`models/MLP/mlp_tanh.npz`).
  - `GET /modeles` : liste des modèles disponibles.
  - `POST /predict` : reçoit `modele` (texte) et `image` (fichier), renvoie `classe`, `classes`, `scores` en JSON.
- `client.py` : l'interface Streamlit. Elle envoie l'image au serveur avec `requests` et affiche la réponse.

## Lancer (depuis le dossier du projet)
1. Compiler la bibliothèque avec CMake (elle sort dans `build/`), ou la copier ici sous le nom `libPA_V2.dll`.
2. `pip install flask requests streamlit pillow numpy`
3. Terminal 1 : `python app/serveur.py`  (le serveur écoute sur http://127.0.0.1:5000)
4. Terminal 2 : `streamlit run app/client.py`

## Tester le serveur sans le client
`curl -F modele=MLP -F image=@une_fleur.jpg http://127.0.0.1:5000/predict`

## Remarque
`application/application.py` (version tout-en-un de l'équipe) n'est pas modifié.
Les scores affichés sont les sorties brutes du modèle (tanh pour le MLP), pas des probabilités.
