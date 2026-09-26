// Fichier de Nadine
//

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#if defined(_WIN32) || defined(__WIN32__)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

// Un rÃ©seau de neurones multicouche
struct MLP {
    int32_t* nb_neurones; // Tableau contenant le nombre de neurones par couche
    int32_t nb_layers;    // nombre total de couches
    double*** W;          // poids : W[couche][neurone_depart][neurone_arrivee]
    double** X;           // sorties de neurones  : X[l] couche l et X[l][i] neurone i dans la couche
    double** deltas;      // erreurs : deltas[couche][neurone]
};

// DÃ©rivÃ©e de Tanh : f'(x) = 1 - x^2
double tanh_deriv(double x) {
    return 1.0 - (x * x);
}

extern "C" {

// 1. CRÃ‰ATION DU MODÃˆLE MLP
DLLEXPORT MLP* create_mlp_model(const int32_t* nb_neurones, int32_t npl_length) {
    MLP* model = new MLP();
    model->nb_layers = npl_length;

    // Copie de la structure des couches
    model->nb_neurones = new int32_t[npl_length];
    for (int l = 0; l < npl_length; ++l) {
        model->nb_neurones[l] = nb_neurones[l];
    }

    // Allocation des activations X et des deltas
    model->X = new double*[npl_length];
    model->deltas = new double*[npl_length];

    for (int l = 0; l < npl_length; ++l) {
        // +1 pour le Biais Ã  l'index 0 (sauf sur la derniÃ¨re couche)
        int size = nb_neurones[l] + (l < npl_length - 1 ? 1 : 0);
        model->X[l] = new double[size];
        model->deltas[l] = new double[size]();

        // Initialisation du biais Ã  1.0
        if (l < npl_length - 1) {
            model->X[l][0] = 1.0;
        }
    }

    // Allocation et initialisation alÃ©atoire des poids W [-1, 1]
    model->W = new double**[npl_length];
    for (int l = 1; l < npl_length; ++l) {
        int prev_neurons = nb_neurones[l - 1] + 1; // +1 pour le biais
        int curr_neurons = nb_neurones[l] + 1;     // +1 pour le biais suivant

        model->W[l] = new double*[prev_neurons];
        for (int i = 0; i < prev_neurons; ++i) {
            model->W[l][i] = new double[curr_neurons];
            for (int j = 0; j < curr_neurons; ++j) {
                model->W[l][i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            }
        }
    }

    return model;
}

//  FORWARD PASS
DLLEXPORT void forward_pass(MLP* model, const double* raw_sample, bool is_classification) {
    // Remplir la couche d'entrÃ©e (l = 0)
    for (int j = 1; j <= model->nb_neurones[0]; ++j) {
        model->X[0][j] = raw_sample[j - 1];
    }

    // Propagation vers les couches suivantes
    for (int l = 1; l < model->nb_layers; ++l) {
        for (int j = 1; j <= model->nb_neurones[l]; ++j) {
            double sum = 0.0;
            for (int i = 0; i <= model->nb_neurones[l - 1]; ++i) {
                sum += model->W[l][i][j] * model->X[l - 1][i];
            }

            if (l == model->nb_layers - 1 && !is_classification) {
                model->X[l][j] = sum;
            } else {
                model->X[l][j] = tanh(sum); // Utilisation directe de tanh de <math.h>
            }
        }
    }
}

// PREDICT (expose le Forward Pass vers Python)
DLLEXPORT double* predict_mlp_model(MLP* model, const double* raw_sample, bool is_classification) {
    forward_pass(model, raw_sample, is_classification);

    int last_layer = model->nb_layers - 1;
    return &model->X[last_layer][1]; // Retourne l'adresse de la premiÃ¨re vraie sortie
}

// backpropagation
DLLEXPORT void train_mlp_model(MLP* model, const double* raw_X, const double* raw_Y,
                               int32_t num_samples, int32_t num_features, int32_t target_dim,
                               double alpha, int32_t epochs, bool is_classification) {
    int L = model->nb_layers - 1;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int k = 0; k < num_samples; ++k) {
            const double* sample_X = &raw_X[k * num_features];
            const double* sample_Y = &raw_Y[k * target_dim];

            // 1. Forward pass
            forward_pass(model, sample_X, is_classification);

            // 2. Deltas de la couche de sortie
            for (int j = 1; j <= model->nb_neurones[L]; ++j) {
                double y_pred = model->X[L][j];
                double target = sample_Y[j - 1];
                double error = y_pred - target;

                if (!is_classification) {
                    model->deltas[L][j] = error;
                } else {
                    model->deltas[L][j] = error * tanh_deriv(y_pred);
                }
            }

            // rÃ©tropropagation des deltas
            for (int l = L; l >= 2; --l) {
                for (int i = 0; i <= model->nb_neurones[l - 1]; ++i) {
                    double sum = 0.0;
                    for (int j = 1; j <= model->nb_neurones[l]; ++j) {
                        sum += model->W[l][i][j] * model->deltas[l][j];
                    }
                    model->deltas[l - 1][i] = sum * tanh_deriv(model->X[l - 1][i]);
                }
            }

            // 4. Mise Ã  jour des poids W
            for (int l = 1; l <= L; ++l) {
                for (int i = 0; i <= model->nb_neurones[l - 1]; ++i) {
                    for (int j = 1; j <= model->nb_neurones[l]; ++j) {
                        model->W[l][i][j] -= alpha * model->X[l - 1][i] * model->deltas[l][j];
                    }
                }
            }
        }
    }
}

// 4. DESTRUCTION DU MODÃˆLE
DLLEXPORT void destroy_mlp_model(MLP* model) {
    if (!model) return;

    for (int l = 0; l < model->nb_layers; ++l) {
        delete[] model->X[l];
        delete[] model->deltas[l];

        if (l > 0) {
            int prev_neurons = model->nb_neurones[l - 1] + 1;
            for (int i = 0; i < prev_neurons; ++i) {
                delete[] model->W[l][i];
            }
            delete[] model->W[l];
        }
    }

    delete[] model->X;
    delete[] model->deltas;
    delete[] model->W;
    delete[] model->nb_neurones;
    delete model;
}

// 5. CALCUL DE LA PERTE (MSE)
DLLEXPORT double get_mlp_loss(MLP* model, const double* raw_X, const double* raw_Y,
                             int32_t num_samples, int32_t num_features, int32_t target_dim,
                             bool is_classification) {
    double total_loss = 0.0;
    int L = model->nb_layers - 1;

    for (int k = 0; k < num_samples; ++k) {
        const double* sample_X = &raw_X[k * num_features];
        const double* sample_Y = &raw_Y[k * target_dim];

        forward_pass(model, sample_X, is_classification);

        for (int j = 1; j <= model->nb_neurones[L]; ++j) {
            double diff = model->X[L][j] - sample_Y[j - 1];
            total_loss += diff * diff;
        }
    }
    return total_loss / (num_samples * target_dim);
}

} // extern "C"