// Fichier de Nadine (base du MLP : structure, forward pass, rétropropagation)
// Corrections / ajouts (étape 1) : Nadine Zerguine
//   - Correction du dépassement mémoire sur la dernière couche (X et deltas)
//   - Ajout de la fonction d'activation Sigmoïde (en plus de Tanh)
//   - Tirage au hasard d'un exemple à chaque itération (comme dans le cours)
//   - Ajout de set_mlp_seed pour des résultats reproductibles
//   - Tirage aléatoire de l'exemple valable aussi sur Windows (rand() sur 15 bits)
//   - (étape 2) get_mlp_num_weights / get_mlp_weights / set_mlp_weights : sauvegarde et chargement

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#if defined(_WIN32) || defined(__WIN32__)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

// Identifiants des fonctions d'activation
#define ACT_TANH    0
#define ACT_SIGMOID 1

// Un réseau de neurones multicouche
struct MLP {
    int32_t* nb_neurones; // Tableau contenant le nombre de neurones par couche
    int32_t nb_layers;    // nombre total de couches
    int32_t activation;   // ACT_TANH (défaut) ou ACT_SIGMOID
    double*** W;          // poids : W[couche][neurone_depart][neurone_arrivee]
    double** X;           // sorties de neurones : X[l][i] neurone i de la couche l (i = 0 : biais)
    double** deltas;      // erreurs : deltas[couche][neurone]
};

// Fonction d'activation appliquée au signal s d'un neurone
static double activate(double s, int32_t act) {
    if (act == ACT_SIGMOID) return 1.0 / (1.0 + exp(-s));
    return tanh(s);
}

// Dérivée exprimée à partir de la SORTIE a = theta(s) du neurone
//   Tanh     : theta'(s) = 1 - a^2      (formule du cours)
//   Sigmoïde : theta'(s) = a * (1 - a)
static double activate_deriv(double a, int32_t act) {
    if (act == ACT_SIGMOID) return a * (1.0 - a);
    return 1.0 - a * a;
}

// Entier aléatoire dans [0, n[.
// Sous Windows rand() ne renvoie que 15 bits (0..32767) : avec un seul appel on ne pourrait jamais
// tirer un indice au-delà de 32767. On combine donc deux appels pour couvrir de grands datasets.
static int32_t random_index(int32_t n) {
    uint32_t r = ((uint32_t)rand() << 15) ^ (uint32_t)rand();
    return (int32_t)(r % (uint32_t)n);
}

extern "C" {

// 1. CRÉATION DU MODÈLE MLP
DLLEXPORT MLP* create_mlp_model(const int32_t* nb_neurones, int32_t npl_length) {
    MLP* model = new MLP();
    model->nb_layers = npl_length;
    model->activation = ACT_TANH;

    // Copie de la structure des couches
    model->nb_neurones = new int32_t[npl_length];
    for (int l = 0; l < npl_length; ++l) {
        model->nb_neurones[l] = nb_neurones[l];
    }

    // Allocation des sorties X et des deltas
    model->X = new double*[npl_length];
    model->deltas = new double*[npl_length];

    for (int l = 0; l < npl_length; ++l) {
        // CORRECTION : +1 pour la case 0 (biais) sur TOUTES les couches, y compris la dernière.
        // Avant : la dernière couche n'avait pas cette case et on écrivait 1 double hors du tableau.
        int size = nb_neurones[l] + 1;
        model->X[l] = new double[size]();
        model->deltas[l] = new double[size]();

        // Biais à 1.0 (case 0)
        model->X[l][0] = 1.0;
    }

    // Allocation et initialisation aléatoire des poids W dans [-1, 1]
    model->W = new double**[npl_length];
    model->W[0] = nullptr; // pas de poids vers la couche d'entrée
    for (int l = 1; l < npl_length; ++l) {
        int prev_neurons = nb_neurones[l - 1] + 1; // +1 pour le biais
        int curr_neurons = nb_neurones[l] + 1;     // +1 : la colonne 0 est inutilisée

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

// Choix de la fonction d'activation : 0 = Tanh, 1 = Sigmoïde
DLLEXPORT void set_mlp_activation(MLP* model, int32_t activation) {
    model->activation = (activation == ACT_SIGMOID) ? ACT_SIGMOID : ACT_TANH;
}

// Graine du générateur aléatoire (à appeler AVANT create_mlp_model pour des poids reproductibles)
DLLEXPORT void set_mlp_seed(uint32_t seed) {
    srand(seed);
}

// FORWARD PASS : x_j^l = theta( sum_i w_ij^l * x_i^(l-1) )
DLLEXPORT void forward_pass(MLP* model, const double* raw_sample, bool is_classification) {
    // Remplir la couche d'entrée (l = 0)
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
                model->X[l][j] = sum; // régression : sortie linéaire
            } else {
                model->X[l][j] = activate(sum, model->activation);
            }
        }
    }
}

// PREDICT (expose le Forward Pass vers Python)
DLLEXPORT double* predict_mlp_model(MLP* model, const double* raw_sample, bool is_classification) {
    forward_pass(model, raw_sample, is_classification);

    int last_layer = model->nb_layers - 1;
    return &model->X[last_layer][1]; // adresse de la première vraie sortie
}

// RÉTROPROPAGATION DU GRADIENT STOCHASTIQUE
// Chaque itération : on tire UN exemple au hasard (comme dans le cours),
// on fait un forward, on calcule les deltas, on met à jour les poids.
// Une "époque" = num_samples itérations.
DLLEXPORT void train_mlp_model(MLP* model, const double* raw_X, const double* raw_Y,
                               int32_t num_samples, int32_t num_features, int32_t target_dim,
                               double alpha, int32_t epochs, bool is_classification) {
    int L = model->nb_layers - 1;
    int32_t act = model->activation;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int it = 0; it < num_samples; ++it) {
            int k = random_index(num_samples); // exemple tiré au hasard
            const double* sample_X = &raw_X[k * num_features];
            const double* sample_Y = &raw_Y[k * target_dim];

            // 1. Forward pass
            forward_pass(model, sample_X, is_classification);

            // 2. Deltas de la couche de sortie
            //    classification : delta = theta'(sortie) * (x - y)   [Tanh : (1 - x^2)(x - y)]
            //    régression     : delta = x - y
            for (int j = 1; j <= model->nb_neurones[L]; ++j) {
                double y_pred = model->X[L][j];
                double target = sample_Y[j - 1];
                double error = y_pred - target;

                if (!is_classification) {
                    model->deltas[L][j] = error;
                } else {
                    model->deltas[L][j] = activate_deriv(y_pred, act) * error;
                }
            }

            // 3. Rétropropagation des deltas : delta_i^(l-1) = theta'(x_i^(l-1)) * sum_j w_ij^l * delta_j^l
            for (int l = L; l >= 2; --l) {
                for (int i = 0; i <= model->nb_neurones[l - 1]; ++i) {
                    double sum = 0.0;
                    for (int j = 1; j <= model->nb_neurones[l]; ++j) {
                        sum += model->W[l][i][j] * model->deltas[l][j];
                    }
                    model->deltas[l - 1][i] = sum * activate_deriv(model->X[l - 1][i], act);
                }
            }

            // 4. Mise à jour des poids : w_ij^l <- w_ij^l - alpha * x_i^(l-1) * delta_j^l
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

// SAUVEGARDE / CHARGEMENT : accès aux poids
// Les poids sont aplatis dans l'ordre : couche l = 1..L, neurone de départ i = 0..nb[l-1] (0 = biais),
// neurone d'arrivée j = 1..nb[l].

// Nombre total de poids du réseau (biais compris)
DLLEXPORT int32_t get_mlp_num_weights(MLP* model) {
    int32_t total = 0;
    for (int l = 1; l < model->nb_layers; ++l) {
        total += (model->nb_neurones[l - 1] + 1) * model->nb_neurones[l];
    }
    return total;
}

// Copie tous les poids dans le tableau 'out' (de taille get_mlp_num_weights)
DLLEXPORT void get_mlp_weights(MLP* model, double* out) {
    int32_t n = 0;
    for (int l = 1; l < model->nb_layers; ++l) {
        for (int i = 0; i <= model->nb_neurones[l - 1]; ++i) {
            for (int j = 1; j <= model->nb_neurones[l]; ++j) {
                out[n++] = model->W[l][i][j];
            }
        }
    }
}

// Remplace tous les poids par ceux du tableau 'in' (même ordre que get_mlp_weights)
DLLEXPORT void set_mlp_weights(MLP* model, const double* in) {
    int32_t n = 0;
    for (int l = 1; l < model->nb_layers; ++l) {
        for (int i = 0; i <= model->nb_neurones[l - 1]; ++i) {
            for (int j = 1; j <= model->nb_neurones[l]; ++j) {
                model->W[l][i][j] = in[n++];
            }
        }
    }
}

// DESTRUCTION DU MODÈLE
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

// CALCUL DE LA PERTE (MSE)
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
