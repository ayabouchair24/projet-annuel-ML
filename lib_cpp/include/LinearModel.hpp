#pragma once

#include <cstddef>
#include <vector>

// Un exemple est représenté par des caractéristiques numériques et une classe.
struct Sample {
    std::vector<double> features;
    int label;
};

// Apprend moyenne et écart-type sur le train, puis transforme les données.
class StandardScaler {
public:
    void fit(const std::vector<Sample>& data);
    void transform(std::vector<Sample>& data) const;

private:
    std::vector<double> means_;
    std::vector<double> standardDeviations_;
};

// Perceptron binaire vu en cours.
// Les sorties attendues sont -1 ou +1.
class LinearPerceptron {
public:
    explicit LinearPerceptron(std::size_t numberOfFeatures);

    void train(const std::vector<Sample>& data, int epochs, double learningRate);

    int predict(const std::vector<double>& features) const;
    double score(const std::vector<double>& features) const;
    double accuracy(const std::vector<Sample>& data) const;

private:
    std::vector<double> weights_;
    double bias_;
};

// Classification multi-classe avec plusieurs perceptrons binaires.
// Un perceptron est entraîné pour chaque classe (classe courante = +1,
// autres classes = -1), puis la classe dont le score est le plus grand est choisie.
class LinearMultiClass {
public:
    LinearMultiClass(std::size_t numberOfFeatures, int numberOfClasses);

    void train(const std::vector<Sample>& data, int epochs, double learningRate);

    int predict(const std::vector<double>& features) const;
    double accuracy(const std::vector<Sample>& data) const;

private:
    std::vector<LinearPerceptron> models_;
};
