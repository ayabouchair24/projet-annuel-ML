#include "ImageFeatures.hpp"
#include "LinearModel.hpp"
#include "mlp.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// Cas de tests
// -----------------------------------------------------------------------------

std::vector<Sample> createLinearTest() {
    // Deux classes séparables par une droite.
    return {
            {{1.0, 1.0},  1},
            {{1.0, 2.0},  1},
            {{2.0, 1.0},  1},
            {{-1.0, -1.0}, -1},
            {{-1.0, -2.0}, -1},
            {{-2.0, -1.0}, -1}
    };
}

std::vector<Sample> createXorTest(bool transformed) {
    // XOR : impossible à séparer par une seule droite dans l'espace initial.
    std::vector<Sample> data = {
            {{-1.0, -1.0}, -1},
            {{ 1.0,  1.0}, -1},
            {{-1.0,  1.0},  1},
            {{ 1.0, -1.0},  1}
    };

    if (transformed) {
        for (Sample& sample : data) {
            const double x1 = sample.features[0];
            const double x2 = sample.features[1];

            // Transformation vue en cours : on ajoute une combinaison
            // non linéaire des entrées pour rendre le problème séparable.
            sample.features.push_back(x1 * x2);
        }
    }

    return data;
}

void runLinearToyExperiment(
        const std::string& title,
        std::vector<Sample> data) {

    std::cout << "\n====================================\n";
    std::cout << title << '\n';
    std::cout << "====================================\n";

    StandardScaler scaler;
    scaler.fit(data);
    scaler.transform(data);

    LinearPerceptron model(data.front().features.size());
    model.train(data, 100, 0.1);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Accuracy : " << model.accuracy(data) << "%\n";
}

void runMlpXorExperiment() {
    std::cout << "\n====================================\n";
    std::cout << "XOR avec PMC\n";
    std::cout << "====================================\n";

    const std::vector<std::vector<double>> X = {
            {-1.0, -1.0},
            { 1.0,  1.0},
            {-1.0,  1.0},
            { 1.0, -1.0}
    };

    // Sortie unique dans [-1, 1], conforme à l'utilisation de tanh vue en cours.
    const std::vector<std::vector<double>> Y = {
            {-1.0},
            {-1.0},
            { 1.0},
            { 1.0}
    };

   MLP model({2, 2, 1});

    const int epochs = 5000;
    const double learningRate = 0.05;
    const std::vector<double> losses = model.fit(X, Y, epochs, learningRate);

    int correct = 0;
    for (std::size_t i = 0; i < X.size(); ++i) {
        if (model.predict_binary(X[i]) == static_cast<int>(Y[i][0])) {
            ++correct;
        }
    }

    std::cout << "Architecture : 2 2 1\n";
    std::cout << "Accuracy : "
              << 100.0 * correct / static_cast<double>(X.size())
              << "%\n";

    if (!losses.empty()) {
        std::cout << "Loss initiale : " << losses.front() << '\n';
        std::cout << "Loss finale   : " << losses.back() << '\n';
    }
}

// -----------------------------------------------------------------------------
// Dataset fleurs
// -----------------------------------------------------------------------------

void showClassCounts(
        const std::vector<Sample>& data,
        const std::vector<std::string>& labels,
        const std::string& datasetName) {

    std::vector<int> counts(labels.size(), 0);
    for (const Sample& sample : data) {
        if (sample.label >= 0 && sample.label < static_cast<int>(counts.size())) {
            ++counts[sample.label];
        }
    }

    std::cout << "\n" << datasetName << " :\n";
    for (std::size_t i = 0; i < labels.size(); ++i) {
        std::cout << " - " << labels[i]
                  << " : " << counts[i] << " image(s)\n";
    }
}

void loadFlowerData(
        const std::string& trainPath,
        const std::string& testPath,
        std::vector<Sample>& trainData,
        std::vector<Sample>& testData,
        std::vector<std::string>& labels) {

    trainData = loadDataset(trainPath, labels);
    testData = loadDataset(testPath, labels);

    if (labels.size() != 3) {
        throw std::runtime_error("Le dataset doit contenir exactement 3 classes.");
    }
    if (trainData.empty() || testData.empty()) {
        throw std::runtime_error("Le dossier train ou test est vide.");
    }
}

void runFlowerLinearExperiment(
        const std::string& trainPath,
        const std::string& testPath,
        int epochs) {

    std::cout << "\n====================================\n";
    std::cout << "Classification des fleurs avec modèle linéaire\n";
    std::cout << "====================================\n";

    std::vector<std::string> labels;
    std::vector<Sample> trainData;
    std::vector<Sample> testData;
    loadFlowerData(trainPath, testPath, trainData, testData, labels);

    showClassCounts(trainData, labels, "Images d'entraînement");
    showClassCounts(testData, labels, "Images de test");

    StandardScaler scaler;
    scaler.fit(trainData);
    scaler.transform(trainData);
    scaler.transform(testData);

    LinearMultiClass model(trainData.front().features.size(), 3);
    model.train(trainData, epochs, 0.001);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nAccuracy entraînement : "
              << model.accuracy(trainData) << "%\n";
    std::cout << "Accuracy test : "
              << model.accuracy(testData) << "%\n";
}

fs::path reportFolder() {
    if (fs::exists("rapport")) {
        return "rapport";
    }
    return "../rapport";
}

void runFlowerMlpExperiment(
        const std::string& trainPath,
        const std::string& testPath,
        int epochs) {

    std::cout << "\n====================================\n";
    std::cout << "Classification des fleurs avec PMC\n";
    std::cout << "====================================\n";

    std::vector<std::string> labels;
    std::vector<Sample> trainData;
    std::vector<Sample> testData;
    loadFlowerData(trainPath, testPath, trainData, testData, labels);

    showClassCounts(trainData, labels, "Images d'entraînement");
    showClassCounts(testData, labels, "Images de test");

    // Le scaler est appris uniquement sur le train.
    StandardScaler scaler;
    scaler.fit(trainData);
    scaler.transform(trainData);
    scaler.transform(testData);

    std::vector<std::vector<double>> XTrain;
    std::vector<std::vector<double>> YTrain;

    for (const Sample& sample : trainData) {
        XTrain.push_back(sample.features);

        // Une sortie par classe, avec 1 pour la classe attendue et 0 ailleurs.
        // Le PMC utilise tanh et apprend ici par erreur quadratique.
        std::vector<double> target(3, 0.0);
        target[sample.label] = 1.0;
        YTrain.push_back(target);
    }

    MLP model({
        static_cast<int>(trainData.front().features.size()),
        16,
        8,
        3
});

    std::cout << "\nArchitecture PMC : ";
    for (int size : model.layer_sizes()) {
        std::cout << size << ' ';
    }
    std::cout << "\n";

    const std::vector<double> losses =
            model.fit(XTrain, YTrain, epochs, 0.01);

    int correctTrain = 0;
    for (const Sample& sample : trainData) {
        if (model.predict_multiclass(sample.features) == sample.label) {
            ++correctTrain;
        }
    }

    int correctTest = 0;
    for (const Sample& sample : testData) {
        if (model.predict_multiclass(sample.features) == sample.label) {
            ++correctTest;
        }
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Accuracy entraînement PMC : "
              << 100.0 * correctTrain / trainData.size() << "%\n";
    std::cout << "Accuracy test PMC : "
              << 100.0 * correctTest / testData.size() << "%\n";

    if (!losses.empty()) {
        std::cout << "Loss initiale : " << losses.front() << '\n';
        std::cout << "Loss finale : " << losses.back() << '\n';

        const fs::path output = reportFolder() / "mlp_loss.csv";
        std::ofstream file(output);
        if (file) {
            file << "epoch,loss\n";
            for (std::size_t i = 0; i < losses.size(); ++i) {
                file << (i + 1) << ',' << losses[i] << '\n';
            }
            std::cout << "Loss sauvegardée dans " << output << '\n';
        }
    }
}

void showUsage() {
    std::cout
            << "Commandes disponibles :\n\n"
            << "1. Cas linéaire :\n"
            << "   --toy-linear\n\n"
            << "2. XOR avec modèle linéaire + transformation :\n"
            << "   --toy-xor\n\n"
            << "3. XOR avec PMC :\n"
            << "   --toy-mlp-xor\n\n"
            << "4. Fleurs avec modèle linéaire :\n"
            << "   --images data/train data/test 100\n\n"
            << "5. Fleurs avec PMC :\n"
            << "   --images-mlp data/train data/test 500\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            showUsage();
            return 0;
        }

        const std::string command = argv[1];

        if (command == "--toy-linear") {
            runLinearToyExperiment(
                    "Cas linéairement séparable",
                    createLinearTest()
            );
        }
        else if (command == "--toy-xor") {
            runLinearToyExperiment(
                    "XOR sans transformation : KO attendu",
                    createXorTest(false)
            );

            runLinearToyExperiment(
                    "XOR après transformation x1*x2 : OK attendu",
                    createXorTest(true)
            );
        }
        else if (command == "--toy-mlp-xor") {
            runMlpXorExperiment();
        }
        else if (command == "--images" || command == "--images-mlp") {
            const std::string trainPath =
                    argc >= 3 ? argv[2] : "data/train";
            const std::string testPath =
                    argc >= 4 ? argv[3] : "data/test";
            const int epochs =
                    argc >= 5 ? std::stoi(argv[4]) :
                    (command == "--images" ? 100 : 500);

            if (command == "--images") {
                runFlowerLinearExperiment(trainPath, testPath, epochs);
            } else {
                runFlowerMlpExperiment(trainPath, testPath, epochs);
            }
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
