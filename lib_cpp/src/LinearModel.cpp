#include "LinearModel.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

void StandardScaler::fit(const std::vector<Sample>& data) {
    if (data.empty()) {
        throw std::runtime_error("Impossible de normaliser un dataset vide.");
    }

    const std::size_t featureCount = data.front().features.size();
    means_.assign(featureCount, 0.0);
    standardDeviations_.assign(featureCount, 0.0);

    for (const Sample& sample : data) {
        if (sample.features.size() != featureCount) {
            throw std::runtime_error("Les samples n'ont pas tous le même nombre de features.");
        }
        for (std::size_t j = 0; j < featureCount; ++j) {
            means_[j] += sample.features[j];
        }
    }

    for (double& mean : means_) {
        mean /= static_cast<double>(data.size());
    }

    for (const Sample& sample : data) {
        for (std::size_t j = 0; j < featureCount; ++j) {
            const double difference = sample.features[j] - means_[j];
            standardDeviations_[j] += difference * difference;
        }
    }

    for (double& deviation : standardDeviations_) {
        deviation = std::sqrt(deviation / static_cast<double>(data.size()));
        if (deviation < 1e-12) {
            deviation = 1.0;
        }
    }
}

void StandardScaler::transform(std::vector<Sample>& data) const {
    if (means_.empty()) {
        throw std::runtime_error("Le scaler doit être entraîné avec fit() avant transform().");
    }

    for (Sample& sample : data) {
        if (sample.features.size() != means_.size()) {
            throw std::runtime_error("Nombre de features incompatible avec le scaler.");
        }
        for (std::size_t j = 0; j < sample.features.size(); ++j) {
            sample.features[j] =
                    (sample.features[j] - means_[j]) / standardDeviations_[j];
        }
    }
}

LinearPerceptron::LinearPerceptron(std::size_t numberOfFeatures)
        : weights_(numberOfFeatures, 0.0), bias_(0.0) {
    // Initialisation conforme au cours : poids nuls ou aléatoires.
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (double& weight : weights_) {
        weight = dist(rng);
    }
    bias_ = dist(rng);
}

double LinearPerceptron::score(const std::vector<double>& features) const {
    if (features.size() != weights_.size()) {
        throw std::runtime_error("Nombre de features incompatible avec le perceptron.");
    }

    double result = bias_;
    for (std::size_t j = 0; j < weights_.size(); ++j) {
        result += weights_[j] * features[j];
    }
    return result;
}

int LinearPerceptron::predict(const std::vector<double>& features) const {
    return score(features) >= 0.0 ? 1 : -1;
}

void LinearPerceptron::train(
        const std::vector<Sample>& data,
        int epochs,
        double learningRate) {

    if (data.empty()) {
        throw std::runtime_error("Dataset d'entraînement vide.");
    }

    std::mt19937 rng(42);
    std::vector<std::size_t> order(data.size());
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);

        for (std::size_t index : order) {
            const Sample& sample = data[index];
            const int prediction = predict(sample.features);

            // Règle de Rosenblatt pour des sorties -1/+1.
            if (prediction != sample.label) {
                for (std::size_t j = 0; j < weights_.size(); ++j) {
                    weights_[j] += learningRate * sample.label * sample.features[j];
                }
                bias_ += learningRate * sample.label;
            }
        }
    }
}

double LinearPerceptron::accuracy(const std::vector<Sample>& data) const {
    if (data.empty()) {
        return 0.0;
    }

    int correct = 0;
    for (const Sample& sample : data) {
        if (predict(sample.features) == sample.label) {
            ++correct;
        }
    }

    return 100.0 * static_cast<double>(correct) / static_cast<double>(data.size());
}

LinearMultiClass::LinearMultiClass(
        std::size_t numberOfFeatures,
        int numberOfClasses) {

    if (numberOfClasses < 2) {
        throw std::runtime_error("Il faut au moins deux classes.");
    }

    models_.reserve(numberOfClasses);
    for (int i = 0; i < numberOfClasses; ++i) {
        models_.emplace_back(numberOfFeatures);
    }
}

void LinearMultiClass::train(
        const std::vector<Sample>& data,
        int epochs,
        double learningRate) {

    if (data.empty()) {
        throw std::runtime_error("Dataset d'entraînement vide.");
    }

    for (std::size_t classIndex = 0; classIndex < models_.size(); ++classIndex) {
        std::vector<Sample> binaryData = data;

        for (Sample& sample : binaryData) {
            sample.label =
                    (sample.label == static_cast<int>(classIndex)) ? 1 : -1;
        }

        models_[classIndex].train(binaryData, epochs, learningRate);
    }
}

int LinearMultiClass::predict(const std::vector<double>& features) const {
    if (models_.empty()) {
        throw std::runtime_error("Aucun modèle linéaire disponible.");
    }

    int bestClass = 0;
    double bestScore = models_[0].score(features);

    for (std::size_t classIndex = 1; classIndex < models_.size(); ++classIndex) {
        const double currentScore = models_[classIndex].score(features);
        if (currentScore > bestScore) {
            bestScore = currentScore;
            bestClass = static_cast<int>(classIndex);
        }
    }

    return bestClass;
}

double LinearMultiClass::accuracy(const std::vector<Sample>& data) const {
    if (data.empty()) {
        return 0.0;
    }

    int correct = 0;
    for (const Sample& sample : data) {
        if (predict(sample.features) == sample.label) {
            ++correct;
        }
    }

    return 100.0 * static_cast<double>(correct) / static_cast<double>(data.size());
}
