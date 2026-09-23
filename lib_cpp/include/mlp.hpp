#pragma once

#include <vector>

// Perceptron multi-couches.
// Chaque couche calcule une somme pondérée,
// ajoute un biais puis applique tanh.
// L'apprentissage utilise la rétropropagation
// du gradient.
class MLP {
public:
    explicit MLP(const std::vector<int>& layer_sizes);

    std::vector<double> fit(
            const std::vector<std::vector<double>>& X,
            const std::vector<std::vector<double>>& Y,
            int epochs,
            double learning_rate);

    std::vector<double> predict_raw(
            const std::vector<double>& x) const;

    int predict_binary(
            const std::vector<double>& x) const;

    int predict_multiclass(
            const std::vector<double>& x) const;

    int n_layers() const {
        return static_cast<int>(layer_sizes_.size());
    }

    const std::vector<int>& layer_sizes() const {
        return layer_sizes_;
    }

    int n_inputs() const {
        return layer_sizes_.front();
    }

    int n_outputs() const {
        return layer_sizes_.back();
    }

private:
    std::vector<double> forward(
            const std::vector<double>& x);

    void backward(
            const std::vector<double>& target,
            double learning_rate);

    static double activation(double z);

    static double activation_derivative(double a);

    std::vector<int> layer_sizes_;

    std::vector<std::vector<std::vector<double>>> weights_;

    std::vector<std::vector<double>> biases_;

    std::vector<std::vector<double>> activations_;
};