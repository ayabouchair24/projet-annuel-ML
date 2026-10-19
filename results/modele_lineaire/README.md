# Projet fleurs - Section "Modèle linéaire"

```
projet_fleurs/
  cpp/linear_model.cpp        # modèle linéaire en C++ (Rosenblatt + pseudo-inverse + sauvegarde)
  cpp/build.bat | build.sh    # compilation (Windows MinGW / Linux-Mac)
  python/linear_model.py      # wrapper ctypes
  notebooks/01_modele_lineaire.ipynb
```

## Utilisation
1. Compiler : `cd cpp` puis `build.bat` (Windows, g++ MinGW) ou `g++ -O2 -std=c++17 -shared -fPIC -o liblinear_model.so linear_model.cpp` (Linux / WSL).
   La bibliothèque doit être compilée sur le système où tourne Python (une .so Linux ne marche pas sous Windows).
2. Ouvrir le notebook, aller à la section 7.1 et adapter `DATASET_DIR` (un sous-dossier par classe : jasmin, jonquille, tournesol).
3. Exécuter les cellules de la section 7 (les sections 1 à 6 contiennent déjà leurs résultats).

## API C++ exposée (extern "C")
create_linear_model, delete_linear_model, train_rosenblatt, train_pseudo_inverse,
predict_scores, get_weights, set_weights, save_linear_model, load_linear_model.
