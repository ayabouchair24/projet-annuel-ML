import ctypes
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement de la DLL
clion_mingw = r"C:\Program Files\JetBrains\CLion 2026.2.2\bin\mingw\bin"
if os.path.exists(clion_mingw):
    os.add_dll_directory(clion_mingw)

lib_name = "libPA.dll"
lib_path = os.path.abspath(lib_name)
os.add_dll_directory(os.path.dirname(lib_path))

cpp_lib = ctypes.CDLL(lib_path)

# Binding des fonctions C++
cpp_lib.create_mlp_model.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
cpp_lib.create_mlp_model.restype = ctypes.c_void_p

cpp_lib.predict_mlp_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_bool]
cpp_lib.predict_mlp_model.restype = ctypes.POINTER(ctypes.c_double)

cpp_lib.train_mlp_model.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_double, ctypes.c_int32, ctypes.c_bool
]
cpp_lib.train_mlp_model.restype = None

# Binding optionnel pour la loss
if hasattr(cpp_lib, 'get_mlp_loss'):
    cpp_lib.get_mlp_loss.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool
    ]
    cpp_lib.get_mlp_loss.restype = ctypes.c_double

cpp_lib.destroy_mlp_model.argtypes = [ctypes.c_void_p]
cpp_lib.destroy_mlp_model.restype = None

# 2. Données XOR
X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
Y_xor = np.array([[-1.0], [1.0], [1.0], [-1.0]], dtype=np.float64)

npl = np.array([2, 2, 1], dtype=np.int32)
model = cpp_lib.create_mlp_model(npl.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(npl))

X_ptr = X_xor.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
Y_ptr = Y_xor.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# 3. Entraînement par étapes pour enregistrer la courbe de Loss
print("Entraînement du MLP et enregistrement de la courbe d'apprentissage...")
losses = []
total_epochs = 50000
step = 500

for epoch in range(0, total_epochs, step):
    cpp_lib.train_mlp_model(model, X_ptr, Y_ptr, 4, 2, 1, 0.05, step, True)
    if hasattr(cpp_lib, 'get_mlp_loss'):
        loss = cpp_lib.get_mlp_loss(model, X_ptr, Y_ptr, 4, 2, 1, True)
        losses.append(loss)

# 4. Affichage graphique (2 figures en 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- GRAPHIQUE 1 : Courbe de Loss ---
if losses:
    ax1.plot(range(0, total_epochs, step), losses, color='purple', linewidth=2)
    ax1.set_title("Courbe d'apprentissage (Loss MSE)")
    ax1.set_xlabel("Époques")
    ax1.set_ylabel("Erreur (MSE)")
    ax1.set_yscale('log')
    ax1.grid(True)

# --- GRAPHIQUE 2 : Frontière de Décision (Decision Boundary) ---
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z = []
for pt in grid_points:
    ptr = pt.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    pred = cpp_lib.predict_mlp_model(model, ptr, True)
    Z.append(pred[0])

Z = np.array(Z).reshape(xx.shape)

# Contour de couleur pour la surface de décision
contour = ax2.contourf(xx, yy, Z, levels=50, cmap='coolwarm', alpha=0.8)
fig.colorbar(contour, ax=ax2, label="Prédiction MLP")

# Trace la ligne de décision exacte (où le réseau prédit 0)
ax2.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

# Dessine les 4 points XOR originaux
ax2.scatter([0, 1], [0, 1], color='red', s=150, edgecolors='k', label='Classe -1 (0,0 et 1,1)')
ax2.scatter([0, 1], [1, 0], color='blue', s=150, edgecolors='k', label='Classe +1 (0,1 et 1,0)')

ax2.set_title("Frontière de décision du MLP (XOR)")
ax2.set_xlabel("X1")
ax2.set_ylabel("X2")
ax2.legend(loc='lower right')
ax2.grid(True)

plt.tight_layout()
plt.show(block=True)

# Nettoyage mémoire
cpp_lib.destroy_mlp_model(model)
