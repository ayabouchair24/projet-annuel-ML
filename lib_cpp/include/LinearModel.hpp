//
// Created by assia on 19/09/2026.
//

#ifndef FLEURS_CLASSIFICATION_LINEARMODEL_HPP
#define FLEURS_CLASSIFICATION_LINEARMODEL_HPP

#pragma once

#include <cstddef>
#include <vector>

// Une image ou un point de test représenté par des nombres
struct Sample {
    std::vector<double> features;
    int label;
};

// Normaliser les valeurs : moyenne 0, écart-type 1
// C'est important car les couleurs et les proportions n'ont pas la même échelle
class StandardScaler {
public:
    void fit(const std::vector<Sample>& data);
    void transform(std::vector<Sample>& data) const;

private:
    std::vector<double> means_;
    std::vector<double> standardDeviations_;
};

// Modèle linéaire multi-classe utilisant softmax
// Aucun modèle de machine learning externe n'est utilisé
class LinearSoftmax {
public:
    LinearSoftmax(std::size_t numberOfFeatures, int numberOfClasses);

    void train(const std::vector<Sample>& data, int epochs, double learningRate);

    int predict(const std::vector<double>& features) const;

    double accuracy(const std::vector<Sample>& data) const;

private:
    std::vector<double> probabilities(const std::vector<double>& features) const;

    std::size_t numberOfFeatures_;
    int numberOfClasses_;

    // weights_[classe][caracteristique]
    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;
};
#endif //FLEURS_CLASSIFICATION_LINEARMODEL_HPP