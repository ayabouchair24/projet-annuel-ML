//
// Created by assia on 19/09/2026.
//
#include "LinearModel.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

void StandardScaler::fit(const std::vector<Sample>& data) {
    if (data.empty()) {
        throw std::runtime_error("Impossible de normaliser un dataset vide.");
    }

    const std::size_t featureCount = data[0].features.size();

    means_.assign(featureCount, 0.0);
    standardDeviations_.assign(featureCount, 0.0);

    // Calcul des moyennes.
    for (const Sample& sample : data) {
        for (std::size_t j = 0; j < featureCount; ++j) {
            means_[j] += sample.features[j];
        }
    }

    for (double& mean : means_) {
        mean /= static_cast<double>(data.size());
    }

    // Calcul des écarts-types.
    for (const Sample& sample : data) {
        for (std::size_t j = 0; j < featureCount; ++j) {
            double difference = sample.features[j] - means_[j];
            standardDeviations_[j] += difference * difference;
        }
    }

    for (double& deviation : standardDeviations_) {
        deviation = std::sqrt(deviation / static_cast<double>(data.size()));

        // Évite une division par zéro si toutes les valeurs sont identiques.
        deviation = std::max(deviation, 1e-9);
    }
}

void StandardScaler::transform(std::vector<Sample>& data) const {
    for (Sample& sample : data) {
        for (std::size_t j = 0; j < sample.features.size(); ++j) {
            sample.features[j] =
                    (sample.features[j] - means_[j]) / standardDeviations_[j];
        }
    }
}

LinearSoftmax::LinearSoftmax(std::size_t numberOfFeatures, int numberOfClasses)
        : numberOfFeatures_(numberOfFeatures),
          numberOfClasses_(numberOfClasses),
          weights_(numberOfClasses, std::vector<double>(numberOfFeatures, 0.0)),
          biases_(numberOfClasses, 0.0) {
}

std::vector<double> LinearSoftmax::probabilities(
        const std::vector<double>& features) const {

    std::vector<double> scores(numberOfClasses_, 0.0);

    // score = w1*x1 + w2*x2 + ... + b
    for (int currentClass = 0; currentClass < numberOfClasses_; ++currentClass) {
        scores[currentClass] = biases_[currentClass];

        for (std::size_t j = 0; j < numberOfFeatures_; ++j) {
            scores[currentClass] +=
                    weights_[currentClass][j] * features[j];
        }
    }

    // Stabilisation numérique du softmax.
    const double maximumScore =
            *std::max_element(scores.begin(), scores.end());

    double sum = 0.0;

    for (double& score : scores) {
        score = std::exp(score - maximumScore);
        sum += score;
    }

    for (double& score : scores) {
        score /= sum;
    }

    return scores;
}

void LinearSoftmax::train(
        const std::vector<Sample>& data,
        int epochs,
        double learningRate) {

    if (data.empty()) {
        throw std::runtime_error("Dataset d'entrainement vide.");
    }

    std::mt19937 randomGenerator(42);

    std::vector<std::size_t> order(data.size());
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), randomGenerator);

        double totalLoss = 0.0;

        for (std::size_t index : order) {
            const Sample& sample = data[index];

            std::vector<double> prediction =
                    probabilities(sample.features);

            totalLoss -= std::log(
                    std::max(prediction[sample.label], 1e-12)
            );

            // Descente de gradient.
            for (int currentClass = 0;
                 currentClass < numberOfClasses_;
                 ++currentClass) {

                const double expected =
                        currentClass == sample.label ? 1.0 : 0.0;

                const double error =
                        prediction[currentClass] - expected;

                for (std::size_t j = 0;
                     j < numberOfFeatures_;
                     ++j) {

                    weights_[currentClass][j] -=
                            learningRate * error * sample.features[j];
                }

                biases_[currentClass] -= learningRate * error;
            }
        }

        if (epoch == 1 || epoch % 50 == 0 || epoch == epochs) {
            std::cout
                    << "Epoch " << epoch
                    << " | loss moyenne = "
                    << totalLoss / static_cast<double>(data.size())
                    << '\n';
        }
    }
}

int LinearSoftmax::predict(
        const std::vector<double>& features) const {

    const std::vector<double> prediction = probabilities(features);

    return static_cast<int>(
            std::distance(
                    prediction.begin(),
                    std::max_element(prediction.begin(), prediction.end())
            )
    );
}

double LinearSoftmax::accuracy(const std::vector<Sample>& data) const {
    if (data.empty()) {
        return 0.0;
    }

    int correctPredictions = 0;

    for (const Sample& sample : data) {
        if (predict(sample.features) == sample.label) {
            ++correctPredictions;
        }
    }

    return 100.0 * static_cast<double>(correctPredictions)
           / static_cast<double>(data.size());
}