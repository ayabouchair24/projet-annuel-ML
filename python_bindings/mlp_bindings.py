"""
Bindings ctypes pour la classe MLP (lib_cpp) — perimetre Membre 2 uniquement.

Expose : la structure dynamique du MLP, la propagation avant (forward) et
les fonctions d'activation (Tanh / Sigmoide) implementees cote C++ dans
mlp.hpp/mlp.cpp + mlp_capi.h/mlp_capi.cpp.

Ce module ne contient volontairement PAS l'entrainement (fit/backward,
partie de Membre 3) ni le Modele Lineaire (ml_capi.h, autre membre) : il
correspond exactement a ce qui est necessaire pour le rendu 2, "ma partie
a moi seulement".

Usage :
    from mlp_bindings import MLP
    m = MLP([2, 4, 1], output_activation="tanh", hidden_activation="tanh")
    print(m.layer_sizes)              # -> [2, 4, 1]
    print(m.forward([0.5, -0.2]))      # -> [valeur dans ]-1, 1[ ]
"""

import ctypes
from pathlib import Path

LIB_PATH = Path(__file__).resolve().parent.parent / "lib_cpp" / "build" / "libml_lib.dll"

_ACTIVATION_CODES = {"tanh": 0, "sigmoid": 1, "linear": 2}


def _load_library(path: Path = LIB_PATH) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))

    lib.mlp_create.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.mlp_create.restype = ctypes.c_void_p

    lib.mlp_destroy.argtypes = [ctypes.c_void_p]

    lib.mlp_n_layers.argtypes = [ctypes.c_void_p]
    lib.mlp_n_layers.restype = ctypes.c_int

    lib.mlp_layer_sizes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

    lib.mlp_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    return lib


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _load_library()
    return _LIB


class MLP:
    def __init__(self, layer_sizes, output_activation="tanh", hidden_activation="tanh"):
        lib = _lib()
        arr_type = ctypes.c_int * len(layer_sizes)
        self._handle = lib.mlp_create(
            arr_type(*layer_sizes),
            len(layer_sizes),
            _ACTIVATION_CODES[output_activation],
            _ACTIVATION_CODES[hidden_activation],
        )

    def __del__(self):
        if getattr(self, "_handle", None):
            _lib().mlp_destroy(self._handle)

    # --- Structure ---------------------------------------------------
    @property
    def layer_sizes(self):
        lib = _lib()
        n = lib.mlp_n_layers(self._handle)
        buf = (ctypes.c_int * n)()
        lib.mlp_layer_sizes(self._handle, buf)
        return list(buf)

    @property
    def n_inputs(self):
        return self.layer_sizes[0]

    @property
    def n_outputs(self):
        return self.layer_sizes[-1]

    # --- Forward pass pur (pas d'entrainement) ------------------------
    def forward(self, x):
        """Propagation avant sans entrainement. Valide la coherence des
        dimensions d'entree/sortie et le comportement des activations."""
        lib = _lib()
        n_in = self.n_inputs
        n_out = self.n_outputs
        if len(x) != n_in:
            raise ValueError(f"Attendu {n_in} features en entree, recu {len(x)}")
        x_arr = (ctypes.c_double * n_in)(*x)
        out_arr = (ctypes.c_double * n_out)()
        lib.mlp_forward(self._handle, x_arr, n_in, out_arr, n_out)
        return list(out_arr)
