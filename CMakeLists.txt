cmake_minimum_required(VERSION 4.3)
project(PA_V2)

set(CMAKE_CXX_STANDARD 20)

add_executable(PA_V2
        src/modele_lineaire.cpp
        src/MLP.cpp
        src/RBF.cpp
        src/SVM.cpp
)

# Bibliothèque partagée du MLP (utilisée par python/mlp_wrapper.py via ctypes)
# Produit libmlp.so (Linux), libmlp.dylib (macOS) ou mlp.dll (Windows) dans le dossier python/
add_library(mlp SHARED src/MLP.cpp)
set_target_properties(mlp PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
