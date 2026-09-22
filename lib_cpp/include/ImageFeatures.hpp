//
// Created by assia on 19/09/2026.
//

#ifndef FLEURS_CLASSIFICATION_IMAGEFEATURES_H
#define FLEURS_CLASSIFICATION_IMAGEFEATURES_H

#pragma once

#include "LinearModel.hpp"

#include <filesystem>
#include <string>
#include <vector>

// Transforme une image en une liste de caractéristiques numériques
std::vector<double> extractFeatures(
        const std::filesystem::path& imagePath
);

// Charge les images présentes dans data/train ou data/test
std::vector<Sample> loadDataset(
        const std::filesystem::path& rootFolder,
        std::vector<std::string>& labels
);
#endif //FLEURS_CLASSIFICATION_IMAGEFEATURES_H