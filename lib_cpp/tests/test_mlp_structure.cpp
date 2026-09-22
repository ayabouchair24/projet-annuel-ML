
#include "mlp.hpp"

#include <cassert>
#include <iostream>
#include <random>
#include <vector>

namespace {

bool check_structure(const std::vector<int>& layer_sizes) {
    MLP model(layer_sizes);
    bool ok = (model.layer_sizes() == layer_sizes) &&
              (model.n_inputs() == layer_sizes.front()) &&
              (model.n_outputs() == layer_sizes.back());
    std::cout << "[structure] layer_sizes=[";
    for (size_t i = 0; i < layer_sizes.size(); ++i)
        std::cout << layer_sizes[i] << (i + 1 < layer_sizes.size() ? "," : "");
    std::cout << "]  => " << (ok ? "OK" : "KO") << "\n";
    return ok;
}

bool check_forward(const std::vector<int>& layer_sizes,
                    MLP::OutputActivation output_activation,
                    MLP::HiddenActivation hidden_activation,
                    const std::string& activation_name,
                    int n_trials = 30) {
    MLP model(layer_sizes, output_activation, hidden_activation);
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    double lo = (output_activation == MLP::OutputActivation::Sigmoid) ? 0.0 : -1.0;
    double hi = (output_activation == MLP::OutputActivation::Linear) ? 1e18 : 1.0;
    bool has_bounds = (output_activation != MLP::OutputActivation::Linear);

    bool ok = true;
    for (int t = 0; t < n_trials; ++t) {
        std::vector<double> x(layer_sizes.front());
        for (double& v : x) v = dist(rng);

        auto y = model.predict_raw(x);
        if (static_cast<int>(y.size()) != layer_sizes.back()) {
            ok = false;
            break;
        }
        if (has_bounds) {
            for (double v : y) {
                if (!(v > lo && v < hi)) {
                    ok = false;
                    break;
                }
            }
        }
    }
    std::cout << "[forward]   arch=[";
    for (size_t i = 0; i < layer_sizes.size(); ++i)
        std::cout << layer_sizes[i] << (i + 1 < layer_sizes.size() ? "," : "");
    std::cout << "] activation=" << activation_name
               << "  (" << n_trials << " entrees aleatoires) => " << (ok ? "OK" : "KO") << "\n";
    return ok;
}

}  // namespace

int main() {
    bool all_ok = true;

    std::cout << "=== Validation de structure ===\n";
    all_ok &= check_structure({2, 1});
    all_ok &= check_structure({2, 2, 1});
    all_ok &= check_structure({2, 4, 1});
    all_ok &= check_structure({2, 3});
    all_ok &= check_structure({2, 16, 16, 3});
    all_ok &= check_structure({768, 16, 1});  // dimension du vrai dataset (16x16x3 px)

    std::cout << "\n=== Validation du forward pass sur entrees aleatoires ===\n";
    all_ok &= check_forward({2, 4, 1}, MLP::OutputActivation::Tanh, MLP::HiddenActivation::Tanh, "tanh/tanh");
    all_ok &= check_forward({2, 4, 1}, MLP::OutputActivation::Sigmoid, MLP::HiddenActivation::Sigmoid, "sigmoid/sigmoid");
    all_ok &= check_forward({2, 4, 1}, MLP::OutputActivation::Tanh, MLP::HiddenActivation::Sigmoid, "tanh(out)/sigmoid(hidden)");
    all_ok &= check_forward({2, 16, 16, 3}, MLP::OutputActivation::Tanh, MLP::HiddenActivation::Tanh, "tanh/tanh (profond)");
    all_ok &= check_forward({1, 4, 1}, MLP::OutputActivation::Linear, MLP::HiddenActivation::Tanh, "linear(out)/tanh (regression)");
    all_ok &= check_forward({768, 16, 1}, MLP::OutputActivation::Tanh, MLP::HiddenActivation::Tanh, "tanh/tanh (dim reelle dataset)");

    std::cout << "\n" << (all_ok ? "TOUT OK" : "ECHEC - voir KO ci-dessus") << "\n";
    return all_ok ? 0 : 1;
}
