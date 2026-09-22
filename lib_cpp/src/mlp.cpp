#include "mlp.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

MLP::MLP(const std::vector<int>& layer_sizes, OutputActivation output_activation,
         HiddenActivation hidden_activation)
    : layer_sizes_(layer_sizes),
      output_activation_(output_activation),
      hidden_activation_(hidden_activation) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    weights_.resize(layer_sizes_.size() - 1);
    for (size_t l = 0; l < weights_.size(); ++l) {
        int n_in = layer_sizes_[l] + 1;  // +1 pour le biais
        int n_out = layer_sizes_[l + 1];
        weights_[l].assign(n_in, std::vector<double>(n_out));
        for (int i = 0; i < n_in; ++i) {
            for (int j = 0; j < n_out; ++j) {
                weights_[l][i][j] = dist(rng);
            }
        }
    }
    activations_.resize(layer_sizes_.size());
}

// Fonctions d'activation

double MLP::apply_activation(HiddenActivation act, double z) {
    if (act == HiddenActivation::Sigmoid) {
        return 1.0 / (1.0 + std::exp(-z));
    }
    return std::tanh(z);
}

double MLP::apply_output_activation(OutputActivation act, double z) {
    switch (act) {
        case OutputActivation::Sigmoid:
            return 1.0 / (1.0 + std::exp(-z));
        case OutputActivation::Linear:
            return z;
        case OutputActivation::Tanh:
        default:
            return std::tanh(z);
    }
}

// Derivee exprimee en fonction de la sortie deja activee "a" (forme
// classique pour Tanh et Sigmoide : evite de recalculer z).
double MLP::activation_derivative(HiddenActivation act, double a) {
    if (act == HiddenActivation::Sigmoid) {
        return a * (1.0 - a);
    }
    return 1.0 - a * a;
}

double MLP::output_activation_derivative(OutputActivation act, double a) {
    switch (act) {
        case OutputActivation::Sigmoid:
            return a * (1.0 - a);
        case OutputActivation::Linear:
            return 1.0;
        case OutputActivation::Tanh:
        default:
            return 1.0 - a * a;
    }
}

//  Propagation avant 

std::vector<double> MLP::forward(const std::vector<double>& x) {
    activations_[0] = x;

    for (size_t l = 0; l < weights_.size(); ++l) {
        int n_in = layer_sizes_[l];
        int n_out = layer_sizes_[l + 1];
        bool is_output_layer = (l == weights_.size() - 1);

        std::vector<double> next(n_out, 0.0);
        for (int j = 0; j < n_out; ++j) {
            double z = weights_[l][n_in][j];  // biais
            for (int i = 0; i < n_in; ++i) {
                z += activations_[l][i] * weights_[l][i][j];
            }
            next[j] = is_output_layer ? apply_output_activation(output_activation_, z)
                                       : apply_activation(hidden_activation_, z);
        }
        activations_[l + 1] = next;
    }
    return activations_.back();
}

// --- Retropropagation (Membre 3) ----------------------------------------

void MLP::backward(const std::vector<double>& target, double learning_rate) {
    size_t n_layers = weights_.size();
    std::vector<std::vector<double>> deltas(n_layers + 1);

    // Delta de la couche de sortie.
    size_t last = n_layers;
    int n_out = layer_sizes_[last];
    deltas[last].resize(n_out);
    for (int j = 0; j < n_out; ++j) {
        double a = activations_[last][j];
        double error = a - target[j];
        double deriv = output_activation_derivative(output_activation_, a);
        deltas[last][j] = error * deriv;
    }

    // Retropropagation dans les couches cachees.
    for (size_t l = n_layers; l-- > 1;) {
        int n_curr = layer_sizes_[l];
        deltas[l].assign(n_curr, 0.0);
        for (int i = 0; i < n_curr; ++i) {
            double sum = 0.0;
            for (int j = 0; j < layer_sizes_[l + 1]; ++j) {
                sum += deltas[l + 1][j] * weights_[l][i][j];
            }
            deltas[l][i] = sum * activation_derivative(hidden_activation_, activations_[l][i]);
        }
    }

    // Mise a jour des poids.
    for (size_t l = 0; l < n_layers; ++l) {
        int n_in = layer_sizes_[l];
        int n_out_l = layer_sizes_[l + 1];
        for (int j = 0; j < n_out_l; ++j) {
            for (int i = 0; i < n_in; ++i) {
                weights_[l][i][j] -= learning_rate * deltas[l + 1][j] * activations_[l][i];
            }
            weights_[l][n_in][j] -= learning_rate * deltas[l + 1][j];  // biais
        }
    }
}

std::vector<double> MLP::fit(const std::vector<std::vector<double>>& X,
                              const std::vector<std::vector<double>>& Y,
                              int epochs,
                              double learning_rate) {
    std::vector<double> mse_history;
    mse_history.reserve(epochs);

    std::vector<size_t> order(X.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(42);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        double squared_error_sum = 0.0;

        for (size_t idx : order) {
            auto output = forward(X[idx]);
            for (size_t j = 0; j < output.size(); ++j) {
                double error = output[j] - Y[idx][j];
                squared_error_sum += error * error;
            }
            backward(Y[idx], learning_rate);
        }
        mse_history.push_back(squared_error_sum / static_cast<double>(X.size()));
    }
    return mse_history;
}

std::vector<double> MLP::predict_raw(const std::vector<double>& x) const {
    return const_cast<MLP*>(this)->forward(x);
}

int MLP::predict_binary(const std::vector<double>& x) const {
    return predict_raw(x)[0] >= 0.0 ? 1 : -1;
}

int MLP::predict_multiclass(const std::vector<double>& x) const {
    auto output = predict_raw(x);
    return static_cast<int>(std::max_element(output.begin(), output.end()) - output.begin());
}
