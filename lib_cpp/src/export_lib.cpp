#include "LinearModel.hpp"
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

extern "C" {

    EXPORT_API LinearSoftmax* LinearModel_create(int num_features, int num_classes) {
        return new LinearSoftmax(static_cast<std::size_t>(num_features), num_classes);
    }

    EXPORT_API void LinearModel_free(LinearSoftmax* model) {
        delete model;
    }

    EXPORT_API void LinearModel_train(LinearSoftmax* model, const double* X, const int* y, int n_samples, int n_features, int epochs, double lr) {
        std::vector<Sample> data(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            data[i].features.assign(X + i * n_features, X + (i + 1) * n_features);
            data[i].label = y[i];
        }
        model->train(data, epochs, lr);
    }

    EXPORT_API int LinearModel_predict(LinearSoftmax* model, const double* features, int n_features) {
        std::vector<double> feat(features, features + n_features);
        return model->predict(feat);
    }

}