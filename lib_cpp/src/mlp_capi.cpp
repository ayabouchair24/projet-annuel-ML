#include "mlp_capi.h"
#include "mlp.hpp"

#include <vector>
#include <iostream>

struct MLPHandle {
    MLP model;
    std::vector<double> last_prediction;

    MLPHandle(const std::vector<int>& layer_sizes,
              MLP::OutputActivation output_act = MLP::OutputActivation::Tanh,
              MLP::HiddenActivation hidden_act = MLP::HiddenActivation::Tanh)
        : model(layer_sizes, output_act, hidden_act) {}
};

extern "C" {

MLPHandle* create_mlp_model(const int* layer_sizes, int n_layers) {
    if (!layer_sizes || n_layers < 2) return nullptr;
    std::vector<int> sizes(layer_sizes, layer_sizes + n_layers);
    
    // Par défaut : Tanh pour les couches cachées et la sortie
    return new MLPHandle(sizes, MLP::OutputActivation::Tanh, MLP::HiddenActivation::Tanh);
}

void destroy_mlp_model(MLPHandle* handle) {
    delete handle;
}

double* predict_mlp_model(MLPHandle* handle, const double* sample, bool is_classification) {
    if (!handle || !sample) return nullptr;

    int input_dim = handle->model.layer_sizes().front();
    std::vector<double> input(sample, sample + input_dim);
    
    handle->last_prediction = handle->model.predict_raw(input);

    if (is_classification) {
        for (double& val : handle->last_prediction) {
            val = (val >= 0.0) ? 1.0 : -1.0;
        }
    }

    return handle->last_prediction.data();
}

void train_mlp_model(
    MLPHandle* handle,
    const double* X,
    const double* Y,
    int sample_count,
    int input_dim,
    int output_dim,
    double alpha,
    int epochs,
    bool is_classification
) {
    if (!handle || !X || !Y) return;

    // Conversion des pointeurs bruts C en std::vector<std::vector<double>>
    std::vector<std::vector<double>> inputs(sample_count, std::vector<double>(input_dim));
    std::vector<std::vector<double>> targets(sample_count, std::vector<double>(output_dim));

    for (int i = 0; i < sample_count; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            inputs[i][j] = X[i * input_dim + j];
        }
        for (int j = 0; j < output_dim; ++j) {
            targets[i][j] = Y[i * output_dim + j];
        }
    }

    // Appel conforme à la signature de votre mlp.hpp : (X, Y, epochs, learning_rate)
    handle->model.fit(inputs, targets, epochs, alpha);
}

double get_mlp_loss(
    MLPHandle* handle,
    const double* X,
    const double* Y,
    int sample_count,
    int input_dim,
    int output_dim,
    bool is_classification
) {
    if (!handle || !X || !Y) return 0.0;

    double total_loss = 0.0;
    for (int i = 0; i < sample_count; ++i) {
        std::vector<double> input(X + i * input_dim, X + (i + 1) * input_dim);
        auto pred = handle->model.predict_raw(input);

        for (int j = 0; j < output_dim; ++j) {
            double target = Y[i * output_dim + j];
            double diff = target - pred[j];
            total_loss += diff * diff;
        }
    }

    return total_loss / (sample_count * output_dim);
}

int mlp_n_layers(MLPHandle* handle) {
    return handle ? handle->model.n_layers() : 0;
}

void mlp_layer_sizes(MLPHandle* handle, int* out_sizes) {
    if (!handle || !out_sizes) return;
    const auto& sizes = handle->model.layer_sizes();
    for (size_t i = 0; i < sizes.size(); ++i) out_sizes[i] = sizes[i];
}

}  // extern "C"