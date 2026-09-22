#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#if defined(_WIN32) || defined(__WIN32__)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

// un réseau de neurones multicouche
struct MLP {
    int32_t* npl;         // Tableau contenant le nombre de neurones par couche (ex: [2, 3, 1])
    int32_t num_layers;   // Nombre total de couches
    double*** W;          // Poids : W[l][i][j] (couche l, neurone i de provenance, neurone j d'arrivée)
    double** X;           // Signal de sortie / Activations : X[l][j]
    double** deltas;      // Erreurs de rétropropagation : deltas[l][j]
};

// fonction d'activation Tanh
double tanh_custom(double x) {
    return tanh(x);
}

// dérivée de Tanh ==> f'(x) = 1 - tanh^2(x)
double tanh_derivative(double x) {
    return 1.0 - (x * x);
}

extern "C" {

// 1. CRÉATION DU MODÈLE MLP
DLLEXPORT MLP* create_mlp_model(const int32_t* npl, int32_t npl_length) {
    MLP* model = new MLP();
    model->num_layers = npl_length;

    // Copie de la structure des couches
    model->npl = new int32_t[npl_length];
    for (int l = 0; l < npl_length; ++l) {
        model->npl[l] = npl[l];
    }

    // Allocation des activations X et des deltas
    model->X = new double*[npl_length];
    model->deltas = new double*[npl_length];

    for (int l = 0; l < npl_length; ++l) {
        // +1 pour le terme de Biais à l'index 0 (sauf sur la dernière couche)
        int size = npl[l] + (l < npl_length - 1 ? 1 : 0);
        model->X[l] = new double[size];
        model->deltas[l] = new double[size]();

        // Initialisation du biais à 1.0
        if (l < npl_length - 1) {
            model->X[l][0] = 1.0;
        }
    }

    // Allocation et initialisation aléatoire des poids W [-1, 1]
    model->W = new double**[npl_length];
    for (int l = 1; l < npl_length; ++l) {
        int prev_neurons = npl[l - 1] + 1; // +1 pour le biais
        int curr_neurons = npl[l] + 1;     // +1 pour le biais de la couche suivante

        model->W[l] = new double*[prev_neurons];
        for (int i = 0; i < prev_neurons; ++i) {
            model->W[l][i] = new double[curr_neurons];
            for (int j = 0; j < curr_neurons; ++j) {
                // Initialisation aléatoire entre -1.0 et 1.0
                model->W[l][i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            }
        }
    }

    return model;
}

// 2. FORWARD PASS (Propagations avant)
DLLEXPORT void forward_pass(MLP* model, const double* raw_sample, bool is_classification) {
    // Remplir la couche d'entrée (l = 0)
    for (int j = 1; j <= model->npl[0]; ++j) {
        model->X[0][j] = raw_sample[j - 1];
    }

    // Propagation à travers les couches cachées et de sortie
    for (int l = 1; l < model->num_layers; ++l) {
        for (int j = 1; j <= model->npl[l]; ++j) {
            double sum = 0.0;
            for (int i = 0; i <= model->npl[l - 1]; ++i) { // i = 0 est le biais
                sum += model->W[l][i][j] * model->X[l - 1][i];
            }

            // Si c'est la dernière couche et qu'on fait de la régression pure, pas de Tanh
            if (l == model->num_layers - 1 && !is_classification) {
                model->X[l][j] = sum;
            } else {
                model->X[l][j] = tanh_custom(sum);
            }
        }
    }
}

// PREDICT (Expose le Forward Pass vers Python)
DLLEXPORT double* predict_mlp_model(MLP* model, const double* raw_sample, bool is_classification) {
    forward_pass(model, raw_sample, is_classification);

    // Retourne le pointeur vers les sorties de la dernière couche
    int last_layer = model->num_layers - 1;
    return &model->X[last_layer][1];
}

// 3. BACKPROPAGATION (Entraînement / Rétropropagation de l'erreur)
DLLEXPORT void train_mlp_model(MLP* model, const double* raw_X, const double* raw_Y,
                            int32_t num_samples, int32_t num_features, int32_t target_dim,
                            double alpha, int32_t epochs, bool is_classification) {
    int L = model->num_layers - 1;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int k = 0; k < num_samples; ++k) {
            // Sélection de l'échantillon courant
            const double* sample_X = &raw_X[k * num_features];
            const double* sample_Y = &raw_Y[k * target_dim];

            // 1. Forward pass
            forward_pass(model, sample_X, is_classification);

            // 2. Calcul des deltas sur la couche de SORTIE (l = L)
            for (int j = 1; j <= model->npl[L]; ++j) {
                double y_pred = model->X[L][j];
                double target = sample_Y[j - 1];
                double error = y_pred - target;

                if (!is_classification) {
                    model->deltas[L][j] = error;
                } else {
                    model->deltas[L][j] = error * tanh_derivative(y_pred);
                }
            }

            // 3. Rétropropagation des deltas (couches L-1 vers 1)
            for (int l = L; l >= 2; --l) {
                for (int i = 0; i <= model->npl[l - 1]; ++i) {
                    double sum = 0.0;
                    for (int j = 1; j <= model->npl[l]; ++j) {
                        sum += model->W[l][i][j] * model->deltas[l][j];
                    }
                    model->deltas[l - 1][i] = sum * tanh_derivative(model->X[l - 1][i]);
                }
            }

            // 4. Mise à jour des poids W (Gradient Descent)
            for (int l = 1; l <= L; ++l) {
                for (int i = 0; i <= model->npl[l - 1]; ++i) {
                    for (int j = 1; j <= model->npl[l]; ++j) {
                        model->W[l][i][j] -= alpha * model->X[l - 1][i] * model->deltas[l][j];
                    }
                }
            }
        }
    }
}

// 4. DESTRUCTION DU MODÈLE (Nettoyage de la mémoire)
DLLEXPORT void destroy_mlp_model(MLP* model) {
    if (!model) return;

    for (int l = 0; l < model->num_layers; ++l) {
        delete[] model->X[l];
        delete[] model->deltas[l];

        if (l > 0) {
            int prev_neurons = model->npl[l - 1] + 1;
            for (int i = 0; i < prev_neurons; ++i) {
                delete[] model->W[l][i];
            }
            delete[] model->W[l];
        }
    }

    delete[] model->X;
    delete[] model->deltas;
    delete[] model->W;
    delete[] model->npl;
    delete model;
}
// Fonction pour calculer la perte (Loss / MSE) à un instant T
DLLEXPORT double get_mlp_loss(MLP* model, const double* raw_X, const double* raw_Y,
                             int32_t num_samples, int32_t num_features, int32_t target_dim,
                             bool is_classification) {
    double total_loss = 0.0;
    int L = model->num_layers - 1;

    for (int k = 0; k < num_samples; ++k) {
        const double* sample_X = &raw_X[k * num_features];
        const double* sample_Y = &raw_Y[k * target_dim];

        forward_pass(model, sample_X, is_classification);

        for (int j = 1; j <= model->npl[L]; ++j) {
            double diff = model->X[L][j] - sample_Y[j - 1];
            total_loss += diff * diff;
        }
    }
    return total_loss / (num_samples * target_dim);
}
} // extern "C"