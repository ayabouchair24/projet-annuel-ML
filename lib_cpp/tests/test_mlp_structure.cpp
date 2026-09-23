#include "mlp.hpp"

#include <iostream>
#include <random>
#include <vector>

namespace {

bool check_structure(
        const std::vector<int>& layer_sizes) {

    MLP model(layer_sizes);

    bool ok =
            model.layer_sizes() == layer_sizes &&
            model.n_inputs() == layer_sizes.front() &&
            model.n_outputs() == layer_sizes.back();

    std::cout << "[structure] [";

    for (std::size_t i = 0;
         i < layer_sizes.size();
         ++i) {

        std::cout << layer_sizes[i];

        if (i + 1 < layer_sizes.size()) {
            std::cout << ",";
        }
    }

    std::cout << "] => "
              << (ok ? "OK" : "KO")
              << "\n";

    return ok;
}


bool check_forward(
        const std::vector<int>& layer_sizes) {

    MLP model(layer_sizes);

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> distribution(
            -5.0,
            5.0
    );

    for (int trial = 0;
         trial < 30;
         ++trial) {

        std::vector<double> x(
                layer_sizes.front()
        );

        for (double& value : x) {
            value = distribution(rng);
        }

        std::vector<double> output =
                model.predict_raw(x);

        if (output.size() !=
            static_cast<std::size_t>(
                    layer_sizes.back())) {

            return false;
        }

        for (double value : output) {

            if (value <= -1.0 ||
                value >= 1.0) {

                return false;
            }
        }
    }

    return true;
}

}

int main() {

    bool all_ok = true;

    std::cout
            << "=== Validation de structure ===\n";

    all_ok &= check_structure({2, 1});
    all_ok &= check_structure({2, 2, 1});
    all_ok &= check_structure({2, 4, 1});
    all_ok &= check_structure({11, 16, 8, 3});

    std::cout
            << "\n=== Validation du forward ===\n";

    all_ok &= check_forward({2, 4, 1});
    all_ok &= check_forward({2, 16, 16, 3});
    all_ok &= check_forward({11, 16, 8, 3});

    std::cout << "\n";

    if (all_ok) {
        std::cout << "TOUT OK\n";
        return 0;
    }

    std::cout
            << "ECHEC\n";

    return 1;
}