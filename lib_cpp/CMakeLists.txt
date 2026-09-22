cmake_minimum_required(VERSION 3.14)
project(lib_cpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Recommandé pour macOS avec Apple Silicon pour localiser Brew
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

# Recherche du paquet OpenCV
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

set(LIB_SOURCES
    src/ImageFeatures.cpp
    src/LinearModel.cpp
    src/mlp.cpp
    src/mlp_capi.cpp
)

# 1. Bibliothèque partagée
add_library(mlp_shared SHARED ${LIB_SOURCES})
set_target_properties(mlp_shared PROPERTIES OUTPUT_NAME "mlp")
target_link_libraries(mlp_shared PRIVATE ${OpenCV_LIBS})

# 2. Exécutable main
add_executable(main_app src/main.cpp)
target_link_libraries(main_app PRIVATE mlp_shared ${OpenCV_LIBS})

# 3. Exécutable de test
add_executable(test_mlp_structure tests/test_mlp_structure.cpp)
target_link_libraries(test_mlp_structure PRIVATE mlp_shared ${OpenCV_LIBS})