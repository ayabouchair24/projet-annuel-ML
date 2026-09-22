import ctypes
import os
import sys

# 1. Localiser le fichier libmlp.dylib
# On remonte d'un dossier (python_bindings/ -> PA/) puis on va dans lib_cpp/build/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Gestion des extensions selon le système d'exploitation (.dylib sur Mac, .so sur Linux, .dll sur Windows)
if sys.platform == "darwin":
    lib_name = "libmlp.dylib"
elif sys.platform.startswith("linux"):
    lib_name = "libmlp.so"
else:
    lib_name = "mlp.dll"

LIB_PATH = os.path.join(PROJECT_ROOT, "lib_cpp", "build", lib_name)

if not os.path.exists(LIB_PATH):
    raise FileNotFoundError(f"Bibliothèque introuvable à l'emplacement : {LIB_PATH}. Avez-vous compilé avec CMake ?")

# 2. Charger la bibliothèque partagée
mlp_lib = ctypes.CDLL(LIB_PATH)

# 3. Définir les types des arguments (argtypes) et de retour (restype) des fonctions C API
# Exemple pour les fonctions déclarées dans mlp_capi.h :

# Exemple : void* create_mlp(int* n_bytes, int n_layers)
if hasattr(mlp_lib, 'create_mlp'):
    mlp_lib.create_mlp.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    mlp_lib.create_mlp.restype = ctypes.c_void_p

# Exemple : void destroy_mlp(void* model)
if hasattr(mlp_lib, 'destroy_mlp'):
    mlp_lib.destroy_mlp.argtypes = [ctypes.c_void_p]
    mlp_lib.destroy_mlp.restype = None

# Exemple : double* predict_mlp(void* model, double* sample, int is_classification)
if hasattr(mlp_lib, 'predict_mlp'):
    mlp_lib.predict_mlp.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    mlp_lib.predict_mlp.restype = ctypes.POINTER(ctypes.c_double)