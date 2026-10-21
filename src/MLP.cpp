// Fichier de Nadine Zerguine : MLP (Perceptron multicouche)

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
using namespace std;

#if defined(_WIN32) || defined(__WIN32__)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

struct MLP {
    vector<int> nb;
    vector<vector<vector<double>>> W;
    vector<vector<double>> X;
    vector<vector<double>> deltas;
};

extern "C" {

DLLEXPORT MLP* create_mlp_model(const int32_t* nb_neurones, int32_t npl_length) {
    MLP* model = new MLP();

    for (int l = 0; l < npl_length; l++) {
        model->nb.push_back(nb_neurones[l]);
        model->X.push_back(vector<double>(nb_neurones[l] + 1, 0.0));
        model->deltas.push_back(vector<double>(nb_neurones[l] + 1, 0.0));
        model->X[l][0] = 1.0;
    }

    model->W.resize(npl_length);
    for (int l = 1; l < npl_length; l++) {
        for (int i = 0; i <= nb_neurones[l - 1]; i++) {
            vector<double> ligne;
            for (int j = 0; j <= nb_neurones[l]; j++) {
                ligne.push_back(((double)rand() / RAND_MAX) * 2.0 - 1.0);
            }
            model->W[l].push_back(ligne);
        }
    }
    return model;
}

DLLEXPORT void set_mlp_seed(uint32_t seed) {
    srand(seed);
}

DLLEXPORT void forward_pass(MLP* model, const double* sample, bool is_classification) {
    int L = model->nb.size() - 1;

    for (int j = 1; j <= model->nb[0]; j++) {
        model->X[0][j] = sample[j - 1];
    }

    for (int l = 1; l <= L; l++) {
        for (int j = 1; j <= model->nb[l]; j++) {
            double somme = 0.0;
            for (int i = 0; i <= model->nb[l - 1]; i++) {
                somme += model->W[l][i][j] * model->X[l - 1][i];
            }
            if (l == L && !is_classification) {
                model->X[l][j] = somme;
            } else {
                model->X[l][j] = tanh(somme);
            }
        }
    }
}

DLLEXPORT double* predict_mlp_model(MLP* model, const double* sample, bool is_classification) {
    forward_pass(model, sample, is_classification);
    int L = model->nb.size() - 1;
    return &model->X[L][1];
}

DLLEXPORT void train_mlp_model(MLP* model, const double* raw_X, const double* raw_Y,
                               int32_t num_samples, int32_t num_features, int32_t target_dim,
                               double alpha, int32_t epochs, bool is_classification) {
    int L = model->nb.size() - 1;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int it = 0; it < num_samples; it++) {
            int k = rand() % num_samples;
            const double* x = &raw_X[k * num_features];
            const double* y = &raw_Y[k * target_dim];

            forward_pass(model, x, is_classification);

            for (int j = 1; j <= model->nb[L]; j++) {
                double erreur = model->X[L][j] - y[j - 1];
                if (is_classification) {
                    model->deltas[L][j] = (1.0 - model->X[L][j] * model->X[L][j]) * erreur;
                } else {
                    model->deltas[L][j] = erreur;
                }
            }

            for (int l = L; l >= 2; l--) {
                for (int i = 0; i <= model->nb[l - 1]; i++) {
                    double somme = 0.0;
                    for (int j = 1; j <= model->nb[l]; j++) {
                        somme += model->W[l][i][j] * model->deltas[l][j];
                    }
                    model->deltas[l - 1][i] = somme * (1.0 - model->X[l - 1][i] * model->X[l - 1][i]);
                }
            }

            for (int l = 1; l <= L; l++) {
                for (int i = 0; i <= model->nb[l - 1]; i++) {
                    for (int j = 1; j <= model->nb[l]; j++) {
                        model->W[l][i][j] -= alpha * model->X[l - 1][i] * model->deltas[l][j];
                    }
                }
            }
        }
    }
}

DLLEXPORT double get_mlp_loss(MLP* model, const double* raw_X, const double* raw_Y,
                              int32_t num_samples, int32_t num_features, int32_t target_dim,
                              bool is_classification) {
    int L = model->nb.size() - 1;
    double total = 0.0;

    for (int k = 0; k < num_samples; k++) {
        forward_pass(model, &raw_X[k * num_features], is_classification);
        for (int j = 1; j <= model->nb[L]; j++) {
            double diff = model->X[L][j] - raw_Y[k * target_dim + j - 1];
            total += diff * diff;
        }
    }
    return total / (num_samples * target_dim);
}

DLLEXPORT int32_t get_mlp_num_weights(MLP* model) {
    int total = 0;
    for (int l = 1; l < (int)model->nb.size(); l++) {
        total += (model->nb[l - 1] + 1) * model->nb[l];
    }
    return total;
}

DLLEXPORT void get_mlp_weights(MLP* model, double* out) {
    int n = 0;
    for (int l = 1; l < (int)model->nb.size(); l++) {
        for (int i = 0; i <= model->nb[l - 1]; i++) {
            for (int j = 1; j <= model->nb[l]; j++) {
                out[n++] = model->W[l][i][j];
            }
        }
    }
}

DLLEXPORT void set_mlp_weights(MLP* model, const double* in) {
    int n = 0;
    for (int l = 1; l < (int)model->nb.size(); l++) {
        for (int i = 0; i <= model->nb[l - 1]; i++) {
            for (int j = 1; j <= model->nb[l]; j++) {
                model->W[l][i][j] = in[n++];
            }
        }
    }
}

DLLEXPORT void destroy_mlp_model(MLP* model) {
    delete model;
}

}
