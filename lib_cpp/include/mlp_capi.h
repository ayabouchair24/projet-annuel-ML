#pragma once

#ifdef _WIN32
  #define ML_API __declspec(dllexport)
#else
  #define ML_API __attribute__((visibility("default")))
#endif

extern "C" {

typedef struct MLPHandle MLPHandle;

// --- Création & Destruction ---
ML_API MLPHandle* create_mlp_model(const int* layer_sizes, int n_layers);
ML_API void destroy_mlp_model(MLPHandle* handle);

// --- Prédiction & Entraînement ---
ML_API double* predict_mlp_model(MLPHandle* handle, const double* sample, bool is_classification);

ML_API void train_mlp_model(
    MLPHandle* handle,
    const double* X,
    const double* Y,
    int sample_count,
    int input_dim,
    int output_dim,
    double alpha,
    int epochs,
    bool is_classification
);

ML_API double get_mlp_loss(
    MLPHandle* handle,
    const double* X,
    const double* Y,
    int sample_count,
    int input_dim,
    int output_dim,
    bool is_classification
);

// --- Introspection (conservation de vos fonctions d'origine) ---
ML_API int mlp_n_layers(MLPHandle* handle);
ML_API void mlp_layer_sizes(MLPHandle* handle, int* out_sizes);

}  // extern "C"