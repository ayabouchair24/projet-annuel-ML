//
// Created by assia on 19/09/2026.
//
#include "ImageFeatures.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <iostream>

namespace fs = std::filesystem;

bool isImageFile(const fs::path& filePath) {
    std::string extension = filePath.extension().string();

    std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            }
    );

    return extension == ".jpg"
           || extension == ".jpeg"
           || extension == ".png";
}

std::vector<double> extractFeatures(const fs::path& imagePath) {
    cv::Mat image = cv::imread(imagePath.string());

    if (image.empty()) {
        std::cerr << "Image impossible a lire : "
                  << imagePath.string() << '\n';

        return {};
    }

    // Toutes les images ont la même taille
    cv::resize(image, image, cv::Size(128, 128));

    cv::Mat hsv;
    cv::Mat gray;

    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Scalar meanBgr;
    cv::Scalar standardDeviationBgr;

    cv::Scalar meanHsv;
    cv::Scalar standardDeviationHsv;

    cv::meanStdDev(image, meanBgr, standardDeviationBgr);
    cv::meanStdDev(hsv, meanHsv, standardDeviationHsv);

    // Détection approximative du jaune
    cv::Mat yellowMask;

    cv::inRange(
            hsv,
            cv::Scalar(18, 70, 70),
            cv::Scalar(42, 255, 255),
            yellowMask
    );

    // Détection des contours
    cv::Mat edges;
    cv::Canny(gray, edges, 80, 160);

    const double numberOfPixels =
            static_cast<double>(image.rows * image.cols);

    const double yellowRatio =
            static_cast<double>(cv::countNonZero(yellowMask))
            / numberOfPixels;

    const double edgeRatio =
            static_cast<double>(cv::countNonZero(edges))
            / numberOfPixels;

    // Onze caractéristiques explicables
    return {
            meanBgr[0],
            meanBgr[1],
            meanBgr[2],

            standardDeviationBgr[0],
            standardDeviationBgr[1],
            standardDeviationBgr[2],

            meanHsv[0],
            meanHsv[1],
            meanHsv[2],

            yellowRatio,
            edgeRatio
    };
}

std::vector<Sample> loadDataset(
        const fs::path& rootFolder,
        std::vector<std::string>& labels) {

    std::vector<Sample> dataset;

    if (!fs::exists(rootFolder)) {
        std::cerr << "Dossier introuvable : "
                  << rootFolder.string() << '\n';

        return dataset;
    }

    std::vector<fs::path> classFolders;

    for (const fs::directory_entry& entry :
         fs::directory_iterator(rootFolder)) {

        if (entry.is_directory()) {
            classFolders.push_back(entry.path());
        }
    }

    std::sort(
            classFolders.begin(),
            classFolders.end(),
            [](const fs::path& first, const fs::path& second) {
                return first.filename().string()
                       < second.filename().string();
            }
    );

    // Pendant le chargement de train, on crée les labels.
    if (labels.empty()) {
        for (const fs::path& classFolder : classFolders) {
            labels.push_back(classFolder.filename().string());
        }
    }

    for (const fs::path& classFolder : classFolders) {
        const std::string className =
                classFolder.filename().string();

        auto labelPosition =
                std::find(labels.begin(), labels.end(), className);

        if (labelPosition == labels.end()) {
            std::cerr << "Classe inconnue ignoree : "
                      << className << '\n';

            continue;
        }

        const int label = static_cast<int>(
                std::distance(labels.begin(), labelPosition)
        );

        for (const fs::directory_entry& file :
             fs::recursive_directory_iterator(classFolder)) {

            if (!file.is_regular_file() || !isImageFile(file.path())) {
                continue;
            }

            std::vector<double> features =
                    extractFeatures(file.path());

            if (!features.empty()) {
                dataset.push_back({features, label});
            }
        }
    }

    return dataset;
}