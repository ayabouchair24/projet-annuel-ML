#include "ImageFeatures.hpp"
#include "LinearModel.hpp"
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "mlp.hpp"

namespace fs = std::filesystem;

// Jeu de test facilement séparable par une droite
std::vector<Sample> createLinearTest() {
    return {
            {{-2.0, -1.0}, 0},
            {{-1.0, -2.0}, 0},
            {{-2.0, -3.0}, 0},

            {{ 1.0,  2.0}, 1},
            {{ 2.0,  1.0}, 1},
            {{ 3.0,  2.0}, 1}
    };
}

// XOR pas séparable par une droite
// Avec transformed = true, on ajoute x1*x2
std::vector<Sample> createXorTest(bool transformed) {
    std::vector<Sample> data = {
            {{-1.0, -1.0}, 0},
            {{ 1.0,  1.0}, 0},

            {{-1.0,  1.0}, 1},
            {{ 1.0, -1.0}, 1}
    };

    if (transformed) {
        for (Sample& sample : data) {
            const double x1 = sample.features[0];
            const double x2 = sample.features[1];

            // Transformation non linéaire :
            // phi(x1, x2) = [x1, x2, x1*x2]
            sample.features.push_back(x1 * x2);
        }
    }

    return data;
}

void runToyExperiment(
        const std::string& title,
        std::vector<Sample> data) {

    std::cout << "\n==============================\n";
    std::cout << title << '\n';
    std::cout << "==============================\n";

    StandardScaler scaler;

    scaler.fit(data);
    scaler.transform(data);

    LinearSoftmax model(
            data[0].features.size(),
            2
    );

    model.train(data, 300, 0.08);

    std::cout << std::fixed << std::setprecision(2);

    std::cout
            << "Accuracy : "
            << model.accuracy(data)
            << "%\n";
}

void showClassCounts(
        const std::vector<Sample>& data,
        const std::vector<std::string>& labels,
        const std::string& datasetName) {

    std::vector<int> counts(labels.size(), 0);

    for (const Sample& sample : data) {
        ++counts[sample.label];
    }

    std::cout << "\n" << datasetName << " :\n";

    for (std::size_t i = 0; i < labels.size(); ++i) {
        std::cout
                << " - " << labels[i]
                << " : " << counts[i]
                << " image(s)\n";
    }
}

void runFlowerExperiment(
        const std::string& trainPath,
        const std::string& testPath,
        int epochs) {

    std::cout << "\nChargement des images...\n";

    std::vector<std::string> labels;

    std::vector<Sample> trainData =
            loadDataset(trainPath, labels);

    std::vector<Sample> testData =
            loadDataset(testPath, labels);

    if (labels.size() != 3) {
        std::cerr
                << "\nERREUR : il faut exactement 3 dossiers de classes.\n"
                << "Exemple : jonquille, fleur_2, fleur_3.\n";

        return;
    }

    if (trainData.empty() || testData.empty()) {
        std::cerr
                << "\nERREUR : le dossier train ou test ne contient pas d'images.\n";

        return;
    }

    showClassCounts(trainData, labels, "Images d'entrainement");
    showClassCounts(testData, labels, "Images de test");

    // La normalisation est apprise seulement avec train.
    StandardScaler scaler;

    scaler.fit(trainData);
    scaler.transform(trainData);
    scaler.transform(testData);

    LinearSoftmax model(
            trainData[0].features.size(),
            3
    );

    std::cout << "\nEntrainement du modele...\n";

    model.train(trainData, epochs, 0.03);

    std::cout << std::fixed << std::setprecision(2);

    std::cout
            << "\nAccuracy entrainement : "
            << model.accuracy(trainData)
            << "%\n";

    std::cout
            << "Accuracy test : "
            << model.accuracy(testData)
            << "%\n";
}void runFlowerMLPExperiment(
        const std::string& trainPath,
        const std::string& testPath,
        int epochs) {

    std::cout << "\n====================================\n";
    std::cout << "Classification des fleurs avec MLP\n";
    std::cout << "====================================\n";

    std::vector<std::string> labels;

    std::vector<Sample> trainData =
            loadDataset(trainPath, labels);

    std::vector<Sample> testData =
            loadDataset(testPath, labels);

    if (labels.size() != 3) {
        std::cerr << "\nERREUR : il faut exactement 3 classes.\n";
        return;
    }

    if (trainData.empty() || testData.empty()) {
        std::cerr << "\nERREUR : train ou test vide.\n";
        return;
    }

    showClassCounts(trainData, labels, "Images d'entrainement");
    showClassCounts(testData, labels, "Images de test");

    // Normalisation apprise uniquement sur le train.
    StandardScaler scaler;
    scaler.fit(trainData);
    scaler.transform(trainData);
    scaler.transform(testData);

    // Conversion vers le format attendu par le MLP.
    std::vector<std::vector<double>> X_train;
    std::vector<std::vector<double>> Y_train;

    for (const Sample& sample : trainData) {
        X_train.push_back(sample.features);

        std::vector<double> target(3, 0.0);
        target[sample.label] = 1.0;
        Y_train.push_back(target);
    }

    // Architecture : 11 features -> 16 -> 8 -> 3 classes.
    MLP model(
            {static_cast<int>(trainData[0].features.size()), 16, 8, 3},
            MLP::OutputActivation::Tanh,
            MLP::HiddenActivation::Tanh
    );

    std::cout << "\nArchitecture MLP : ";
    for (int size : model.layer_sizes()) {
        std::cout << size << " ";
    }
    std::cout << "\n";

    std::cout << "Entrainement du MLP...\n";

    std::vector<double> losses =
            model.fit(X_train, Y_train, epochs, 0.01);

        // Sauvegarde de la courbe de Loss pour le rapport.
std::ofstream lossFile("../rapport/mlp_loss.csv");

if (lossFile.is_open()) {
    lossFile << "epoch,loss\n";

    for (std::size_t i = 0; i < losses.size(); ++i) {
        lossFile << (i + 1) << "," << losses[i] << "\n";
    }

    lossFile.close();

    std::cout << "Courbe de Loss sauvegardee dans ../rapport/mlp_loss.csv\n";
} else {
    std::cerr << "Attention : impossible de sauvegarder mlp_loss.csv\n";
}

    // Accuracy train.
    int correctTrain = 0;

    for (const Sample& sample : trainData) {
        int prediction =
                model.predict_multiclass(sample.features);

        if (prediction == sample.label) {
            ++correctTrain;
        }
    }

    // Accuracy test.
    int correctTest = 0;

    for (const Sample& sample : testData) {
        int prediction =
                model.predict_multiclass(sample.features);

        if (prediction == sample.label) {
            ++correctTest;
        }
    }

    double trainAccuracy =
            100.0 * correctTrain / trainData.size();

    double testAccuracy =
            100.0 * correctTest / testData.size();

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "\nAccuracy entrainement MLP : "
              << trainAccuracy << "%\n";

    std::cout << "Accuracy test MLP : "
              << testAccuracy << "%\n";

    if (!losses.empty()) {
        std::cout << "Loss initiale : "
                  << losses.front() << "\n";

        std::cout << "Loss finale : "
                  << losses.back() << "\n";
    }
}

void showUsage() {
    std::cout
            << "Commandes disponibles :\n\n"
            << "4. Classification des fleurs avec MLP :\n"
            << "   --images-mlp data/train data/test 500\n"
            << "1. Test lineaire :\n"
            << "   --toy-linear\n\n"
            << "2. Test XOR :\n"
            << "   --toy-xor\n\n"
            << "3. Classification des fleurs :\n"
            << "   --images data/train data/test 500\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            showUsage();
            return 0;
        }

        const std::string command = argv[1];

        if (command == "--toy-linear") {
            runToyExperiment(
                    "Cas lineairement separable",
                    createLinearTest()
            );
        }
        else if (command == "--toy-xor") {
            runToyExperiment(
                    "XOR : cas KO sans transformation non lineaire",
                    createXorTest(false)
            );

            runToyExperiment(
                    "XOR : cas OK avec phi(x) = [x1, x2, x1*x2]",
                    createXorTest(true)
            );
        }
        else if (command == "--images") {
                
            const std::string trainPath =
                    argc >= 3 ? argv[2] : "data/train";

            const std::string testPath =
                    argc >= 4 ? argv[3] : "data/test";

            const int epochs =
                    argc >= 5 ? std::stoi(argv[4]) : 500;

            runFlowerExperiment(trainPath, testPath, epochs);
        }
        else if (command == "--images-mlp") {
    const std::string trainPath =
            argc >= 3 ? argv[2] : "data/train";

    const std::string testPath =
            argc >= 4 ? argv[3] : "data/test";

    const int epochs =
            argc >= 5 ? std::stoi(argv[4]) : 500;

    runFlowerMLPExperiment(trainPath, testPath, epochs);
}
        else {
            showUsage();
        }
    }
    catch (const std::exception& error) {
        std::cerr << "\nErreur : " << error.what() << '\n';
        return 1;
    }

    return 0;
}