cmake_minimum_required(VERSION 3.15)
project(PA_V2)

set(CMAKE_CXX_STANDARD 20)

add_executable(PA_V2
        src/modele_lineaire.cpp
        src/MLP.cpp
        src/RBF.cpp
        src/SVM.cpp
)
# Bibliothèque partagée du modèle linéaire (utilisée par python/linear_model.py via ctypes)
add_library(linear_model SHARED src/modele_lineaire.cpp)
set_target_properties(linear_model PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python)

if(WIN32)
    # produit linear_model.dll (sans préfixe "lib"), le nom attendu par le wrapper
    set_target_properties(linear_model PROPERTIES PREFIX "")
endif()
if(MINGW)
    target_link_options(linear_model PRIVATE -static)
endif()