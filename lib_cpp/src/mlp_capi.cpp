#include "mlp_capi.h"
#include "mlp.hpp"

#include <vector>

namespace {

MLP::OutputActivation to_output_activation(int code) {
    switch (code) {
        case 1: return MLP::OutputActivation::Sigmoid;
        case 2: return MLP::OutputActivation::Linear;
        default: return MLP::OutputActivation::Tanh;
    }
}

MLP::HiddenActivation to_hidden_activation(int code) {
    return code == 1 ? MLP::HiddenActivation::Sigmoid : MLP::HiddenActivation::Tanh;
}

}  // namespace

struct MLPHandle {
    MLP model;
    MLPHandle(const std::vector<int>& layer_sizes,
              MLP::OutputActivation output_activation,
              MLP::HiddenActivation hidden_activation)
        : model(layer_sizes, output_activation, hidden_activation) {}
};

extern "C" {

MLPHandle* mlp_create(const int* layer_sizes, int n_layers,
                       int output_activation, int hidden_activation) {
    std::vector<int> sizes(layer_sizes, layer_sizes + n_layers);
    return new MLPHandle(sizes, to_output_activation(output_activation),
                          to_hidden_activation(hidden_activation));
}

void mlp_destroy(MLPHandle* handle) {
    delete handle;
}

int mlp_n_layers(MLPHandle* handle) {
    return handle->model.n_layers();
}

void mlp_layer_sizes(MLPHandle* handle, int* out_sizes) {
    const auto& sizes = handle->model.layer_sizes();
    for (size_t i = 0; i < sizes.size(); ++i) out_sizes[i] = sizes[i];
}

void mlp_forward(MLPHandle* handle, const double* x, int n_features,
                  double* out, int n_outputs) {
    std::vector<double> input(x, x + n_features);
    auto output = handle->model.predict_raw(input);
    for (int i = 0; i < n_outputs; ++i) out[i] = output[i];
}

}  // extern "C"
