cmake_minimum_required(VERSION 4.3)
project(PA_V2)

set(CMAKE_CXX_STANDARD 20)

add_executable(PA_V2
        src/modele_lineaire.cpp
        src/MLP.cpp
        src/RBF.cpp
        src/SVM.cpp
)
