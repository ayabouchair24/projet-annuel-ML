#include "mlp.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

MLP::MLP(const std::vector<int>& layer_sizes)
        : layer_sizes_(layer_sizes) {

    if (layer_sizes_.size() < 2) {
        throw std::runtime_error(
                "Un PMC doit avoir au moins une couche d'entree et une couche de sortie."
        );
    }

    for (int size : layer_sizes_) {
        if (size <= 0) {
            throw std::runtime_error(
                    "La taille d'une couche doit etre positive."
            );
        }
    }

    // Création des poids et des biais.
    for (std::size_t l = 1; l < layer_sizes_.size(); ++l) {

        const int inputSize = layer_sizes_[l - 1];
        const int outputSize = layer_sizes_[l];

        std::vector<std::vector<double>> weights(
                outputSize,
                std::vector<double>(inputSize)
        );

        std::vector<double> biases(
                outputSize,
                0.0
        );

        for (int i = 0; i < outputSize; ++i) {

            for (int j = 0; j < inputSize; ++j) {

                weights[i][j] =
                        ((double) std::rand() / RAND_MAX) * 2.0 - 1.0;
            }
        }

        weights_.push_back(weights);
        biases_.push_back(biases);
    }
}


// -----------------------------------------------------------------------------
// Fonction d'activation
// -----------------------------------------------------------------------------

double MLP::activation(double z) {
    return std::tanh(z);
}


// Dérivée de tanh.
// Ici a représente déjà tanh(z).
double MLP::activation_derivative(double a) {
    return 1.0 - a * a;
}


// -----------------------------------------------------------------------------
// Propagation avant
// -----------------------------------------------------------------------------

std::vector<double> MLP::forward(
        const std::vector<double>& input) {

    if (input.size() != static_cast<std::size_t>(n_inputs())) {
        throw std::runtime_error(
                "Nombre d'entrees incompatible avec le PMC."
        );
    }

    activations_.clear();
    activations_.push_back(input);

    std::vector<double> current = input;

    for (std::size_t l = 0; l < weights_.size(); ++l) {

        std::vector<double> next(
                weights_[l].size(),
                0.0
        );

        for (std::size_t i = 0; i < weights_[l].size(); ++i) {

            double z = biases_[l][i];

            for (std::size_t j = 0; j < current.size(); ++j) {
                z += weights_[l][i][j] * current[j];
            }

            next[i] = activation(z);
        }

        activations_.push_back(next);
        current = next;
    }

    return current;
}


// -----------------------------------------------------------------------------
// Prédiction
// -----------------------------------------------------------------------------

std::vector<double> MLP::predict_raw(
        const std::vector<double>& input) const {

    if (input.size() != static_cast<std::size_t>(n_inputs())) {
        throw std::runtime_error(
                "Nombre d'entrees incompatible avec le PMC."
        );
    }

    std::vector<double> current = input;

    for (std::size_t l = 0; l < weights_.size(); ++l) {

        std::vector<double> next(
                weights_[l].size(),
                0.0
        );

        for (std::size_t i = 0; i < weights_[l].size(); ++i) {

            double z = biases_[l][i];

            for (std::size_t j = 0; j < current.size(); ++j) {
                z += weights_[l][i][j] * current[j];
            }

            next[i] = activation(z);
        }

        current = next;
    }

    return current;
}


// -----------------------------------------------------------------------------
// Rétropropagation
// -----------------------------------------------------------------------------

void MLP::backward(
        const std::vector<double>& target,
        double learning_rate) {

    if (target.size() != static_cast<std::size_t>(n_outputs())) {
        throw std::runtime_error(
                "Dimension de la cible incompatible avec le PMC."
        );
    }

    const std::size_t numberOfLayers = weights_.size();

    // Une erreur par couche.
    std::vector<std::vector<double>> errors(
            numberOfLayers
    );

    // Erreur de sortie.
    const std::vector<double>& output =
            activations_.back();

    errors.back().resize(output.size());

    for (std::size_t i = 0; i < output.size(); ++i) {

        double difference =
                output[i] - target[i];

        errors.back()[i] =
                difference *
                activation_derivative(output[i]);
    }

    // Propagation de l'erreur vers les couches cachées.
    for (int l = static_cast<int>(numberOfLayers) - 2;
         l >= 0;
         --l) {

        errors[l].resize(weights_[l].size());

        for (std::size_t i = 0;
             i < weights_[l].size();
             ++i) {

            double error = 0.0;

            for (std::size_t k = 0;
                 k < weights_[l + 1].size();
                 ++k) {

                error +=
                        weights_[l + 1][k][i] *
                        errors[l + 1][k];
            }

            errors[l][i] =
                    error *
                    activation_derivative(
                            activations_[l + 1][i]
                    );
        }
    }

    // Mise à jour des poids et des biais.
    for (std::size_t l = 0;
         l < numberOfLayers;
         ++l) {

        const std::vector<double>& previous =
                activations_[l];

        for (std::size_t i = 0;
             i < weights_[l].size();
             ++i) {

            for (std::size_t j = 0;
                 j < weights_[l][i].size();
                 ++j) {

                weights_[l][i][j] -=
                        learning_rate *
                        errors[l][i] *
                        previous[j];
            }

            biases_[l][i] -=
                    learning_rate *
                    errors[l][i];
        }
    }
}


// -----------------------------------------------------------------------------
// Entraînement
// -----------------------------------------------------------------------------

std::vector<double> MLP::fit(
        const std::vector<std::vector<double>>& X,
        const std::vector<std::vector<double>>& Y,
        int epochs,
        double learning_rate) {

    if (X.empty() || Y.empty()) {
        throw std::runtime_error(
                "Le dataset d'entraînement est vide."
        );
    }

    if (X.size() != Y.size()) {
        throw std::runtime_error(
                "X et Y doivent avoir le même nombre d'exemples."
        );
    }

    std::vector<double> losses;

    for (int epoch = 0;
         epoch < epochs;
         ++epoch) {

        double totalLoss = 0.0;

        for (std::size_t i = 0;
             i < X.size();
             ++i) {

            std::vector<double> output =
                    forward(X[i]);

            double sampleLoss = 0.0;

            for (std::size_t j = 0;
                 j < output.size();
                 ++j) {

                double difference =
                        output[j] - Y[i][j];

                sampleLoss +=
                        difference * difference;
            }

            totalLoss += sampleLoss;

            backward(
                    Y[i],
                    learning_rate
            );
        }

        totalLoss /=
                static_cast<double>(
                        X.size()
                );

        losses.push_back(totalLoss);
    }

    return losses;
}


// -----------------------------------------------------------------------------
// Classification binaire
// -----------------------------------------------------------------------------

int MLP::predict_binary(
        const std::vector<double>& x) const {

    std::vector<double> output =
            predict_raw(x);

    if (output.empty()) {
        throw std::runtime_error(
                "Le PMC ne possède pas de sortie."
        );
    }

    return output[0] >= 0.0 ? 1 : -1;
}


// -----------------------------------------------------------------------------
// Classification multi-classe
// -----------------------------------------------------------------------------

int MLP::predict_multiclass(
        const std::vector<double>& x) const {

    std::vector<double> output =
            predict_raw(x);

    if (output.empty()) {
        throw std::runtime_error(
                "Le PMC ne possède pas de sortie."
        );
    }

    int bestClass = 0;

    for (std::size_t i = 1;
         i < output.size();
         ++i) {

        if (output[i] > output[bestClass]) {
            bestClass = static_cast<int>(i);
        }
    }

    return bestClass;
}