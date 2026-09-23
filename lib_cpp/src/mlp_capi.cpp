#include "mlp_capi.h"
#include "mlp.hpp"

#include <vector>

struct MLPHandle {
    MLP model;
    std::vector<double> last_prediction;

    explicit MLPHandle(const std::vector<int>& layer_sizes)
        : model(layer_sizes) {}
};

extern "C" {

MLPHandle* create_mlp_model(const int* layer_sizes, int n_layers) {
    if (!layer_sizes || n_layers < 2) {
        return nullptr;
    }

    std::vector<int> sizes(layer_sizes, layer_sizes + n_layers);
    return new MLPHandle(sizes);
}

void destroy_mlp_model(MLPHandle* handle) {
    delete handle;
}

double* predict_mlp_model(
        MLPHandle* handle,
        const double* sample,
        bool /*is_classification*/) {

    if (!handle || !sample) {
        return nullptr;
    }

    const int inputDim = handle->model.n_inputs();
    std::vector<double> input(sample, sample + inputDim);

    // On renvoie les sorties réelles du PMC.
    // La décision de classe est faite par le code appelant :
    // signe pour une sortie binaire, maximum pour plusieurs sorties.
    handle->last_prediction = handle->model.predict_raw(input);
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
        bool /*is_classification*/) {

    if (!handle || !X || !Y || sample_count <= 0 || epochs <= 0) {
        return;
    }

    if (input_dim != handle->model.n_inputs()
        || output_dim != handle->model.n_outputs()) {
        return;
    }

    std::vector<std::vector<double>> inputs(
            sample_count, std::vector<double>(input_dim));
    std::vector<std::vector<double>> targets(
            sample_count, std::vector<double>(output_dim));

    for (int i = 0; i < sample_count; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            inputs[i][j] = X[i * input_dim + j];
        }
        for (int j = 0; j < output_dim; ++j) {
            targets[i][j] = Y[i * output_dim + j];
        }
    }

    handle->model.fit(inputs, targets, epochs, alpha);
}

double get_mlp_loss(
        MLPHandle* handle,
        const double* X,
        const double* Y,
        int sample_count,
        int input_dim,
        int output_dim,
        bool /*is_classification*/) {

    if (!handle || !X || !Y || sample_count <= 0) {
        return 0.0;
    }

    if (input_dim != handle->model.n_inputs()
        || output_dim != handle->model.n_outputs()) {
        return 0.0;
    }

    double totalLoss = 0.0;

    for (int i = 0; i < sample_count; ++i) {
        std::vector<double> input(
                X + i * input_dim,
                X + (i + 1) * input_dim);

        const std::vector<double> prediction =
                handle->model.predict_raw(input);

        for (int j = 0; j < output_dim; ++j) {
            const double error =
                    prediction[j] - Y[i * output_dim + j];
            totalLoss += error * error;
        }
    }

    return totalLoss /
           static_cast<double>(sample_count * output_dim);
}

int mlp_n_layers(MLPHandle* handle) {
    return handle ? handle->model.n_layers() : 0;
}

void mlp_layer_sizes(MLPHandle* handle, int* out_sizes) {
    if (!handle || !out_sizes) {
        return;
    }

    const auto& sizes = handle->model.layer_sizes();
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        out_sizes[i] = sizes[i];
    }
}

} // extern "C"
