#pragma once

#ifdef _WIN32
#define ML_API __declspec(dllexport)
#else
#define ML_API
#endif

// API C dediee au MLP (Membre 2 : structure, forward pass, activations).
// Separee de ml_capi.h (Modele Lineaire, autre membre) pour que le
// perimetre de mon travail soit clairement identifiable (fichier,
// historique git, etc.), tout en etant compilee dans la meme lib
// partagee ml_lib (voir CMakeLists.txt).
extern "C" {

typedef struct MLPHandle MLPHandle;

// hidden_activation : 0 = Tanh, 1 = Sigmoide
// output_activation : 0 = Tanh, 1 = Sigmoide, 2 = Lineaire (regression)
ML_API MLPHandle* mlp_create(const int* layer_sizes, int n_layers,
                              int output_activation, int hidden_activation);
ML_API void mlp_destroy(MLPHandle* handle);

// --- Introspection de structure : validation des dimensions ---
ML_API int mlp_n_layers(MLPHandle* handle);
ML_API void mlp_layer_sizes(MLPHandle* handle, int* out_sizes);

// --- Propagation avant pure (pas d'entrainement) ---
// x : n_features valeurs (= layer_sizes[0]). out : buffer de taille
// layer_sizes[n_layers-1] fourni par l'appelant.
ML_API void mlp_forward(MLPHandle* handle, const double* x, int n_features,
                         double* out, int n_outputs);

}  // extern "C"
