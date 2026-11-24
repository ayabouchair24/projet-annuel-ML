cmake_minimum_required(VERSION 3.15...3.30)
project(PA_V2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Your library configuration
add_library(PA_V2 SHARED
        src/RBF.cpp
        src/MLP.cpp
        src/modele_lineaire.cpp
)
