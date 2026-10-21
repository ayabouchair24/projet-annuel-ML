// fichier d'aya
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

#if defined(_WIN32) || defined(__WIN32__)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

struct RBFModel {
    int32_t dimensions;
    int32_t centersCount;
    double gamma;
    std::vector<double> centers;
    std::vector<double> weights;
};

static double gaussian(const double* x, const double* center,
                       int32_t d, double gamma) {
    double distance = 0.0;

    for (int32_t j = 0; j < d; ++j) {
        double diff = x[j] - center[j];
        distance += diff * diff;
    }

    return std::exp(-gamma * distance);
}

static bool solveSystem(std::vector<double> A,
                        std::vector<double> b,
                        int n,
                        std::vector<double>& solution) {
    for (int i = 0; i < n; ++i) {
        int pivot = i;

        for (int j = i + 1; j < n; ++j) {
            if (std::abs(A[j*n+i]) > std::abs(A[pivot*n+i]))
                pivot = j;
        }

        if (std::abs(A[pivot*n+i]) < 1e-12)
            return false;

        for (int j = 0; j < n; ++j)
            std::swap(A[i*n+j], A[pivot*n+j]);

        std::swap(b[i], b[pivot]);

        for (int j = i + 1; j < n; ++j) {
            double factor = A[j*n+i] / A[i*n+i];

            for (int k = i; k < n; ++k)
                A[j*n+k] -= factor * A[i*n+k];

            b[j] -= factor * b[i];
        }
    }

    solution.assign(n, 0.0);

    for (int i = n - 1; i >= 0; --i) {
        double value = b[i];

        for (int j = i + 1; j < n; ++j)
            value -= A[i*n+j] * solution[j];

        solution[i] = value / A[i*n+i];
    }

    return true;
}

extern "C" {

DLLEXPORT RBFModel* create_rbf_model(int32_t dimensions,
                                     int32_t K,
                                     double gamma) {
    if (dimensions < 1 || K < 1 ||
        !std::isfinite(gamma) || gamma <= 0.0)
        return nullptr;

    return new RBFModel{dimensions, K, gamma, {}, {}};
}

DLLEXPORT int32_t train_rbf_model(RBFModel* model,
                                  const double* X,
                                  const double* Y,
                                  int32_t n,
                                  int32_t d,
                                  double lambda,
                                  int32_t maxIterations) {
    if (!model || !X || !Y || n < 1 ||
        d != model->dimensions ||
        model->centersCount > n ||
        !std::isfinite(lambda) || lambda < 0.0 ||
        maxIterations < 1)
        return -1;

    int K = model->centersCount;

    model->centers.assign(K*d, 0.0);
    model->weights.clear();

    if (K == n) {
        std::copy(X, X+n*d, model->centers.begin());
    } else {
        std::copy(X, X+d, model->centers.begin());

        for (int k = 1; k < K; ++k) {
            int bestIndex = 0;
            double bestDistance = -1.0;

            for (int i = 0; i < n; ++i) {
                double nearest = std::numeric_limits<double>::infinity();

                for (int c = 0; c < k; ++c) {
                    double distance = 0.0;

                    for (int j = 0; j < d; ++j) {
                        double diff = X[i*d+j] -
                                      model->centers[c*d+j];
                        distance += diff * diff;
                    }

                    nearest = std::min(nearest, distance);
                }

                if (nearest > bestDistance) {
                    bestDistance = nearest;
                    bestIndex = i;
                }
            }

            std::copy(X+bestIndex*d, X+(bestIndex+1)*d,
                      model->centers.begin()+k*d);
        }

        std::vector<int> assignments(n, -1);

        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            bool changed = false;
            std::vector<double> sums(K*d, 0.0);
            std::vector<int> counts(K, 0);

            for (int i = 0; i < n; ++i) {
                int nearest = 0;
                double bestDistance = std::numeric_limits<double>::infinity();

                for (int c = 0; c < K; ++c) {
                    double distance = 0.0;

                    for (int j = 0; j < d; ++j) {
                        double diff = X[i*d+j] -
                                      model->centers[c*d+j];
                        distance += diff * diff;
                    }

                    if (distance < bestDistance) {
                        bestDistance = distance;
                        nearest = c;
                    }
                }

                if (assignments[i] != nearest)
                    changed = true;

                assignments[i] = nearest;
                counts[nearest]++;

                for (int j = 0; j < d; ++j)
                    sums[nearest*d+j] += X[i*d+j];
            }

            for (int c = 0; c < K; ++c) {
                if (counts[c] > 0) {
                    for (int j = 0; j < d; ++j)
                        model->centers[c*d+j] =
                            sums[c*d+j] / counts[c];
                }
            }

            if (!changed)
                break;
        }
    }

    std::vector<double> A;
    std::vector<double> b;

    if (K == n) {
        A.assign(n*n, 0.0);
        b.assign(Y, Y+n);

        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < K; ++k) {
                A[i*K+k] = gaussian(
                    X+i*d,
                    model->centers.data()+k*d,
                    d, model->gamma
                );
            }
        }

        for (int k = 0; k < K; ++k)
            A[k*K+k] += lambda;
    } else {
        A.assign(K*K, 0.0);
        b.assign(K, 0.0);

        std::vector<double> phi(K);

        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < K; ++k) {
                phi[k] = gaussian(
                    X+i*d,
                    model->centers.data()+k*d,
                    d, model->gamma
                );
            }

            for (int a = 0; a < K; ++a) {
                b[a] += phi[a] * Y[i];

                for (int c = 0; c < K; ++c)
                    A[a*K+c] += phi[a] * phi[c];
            }
        }

        for (int k = 0; k < K; ++k)
            A[k*K+k] += lambda;
    }

    if (!solveSystem(A, b, K, model->weights)) {
        model->weights.clear();
        return -2;
    }

    return 0;
}

DLLEXPORT double decision_rbf(RBFModel* model,
                              const double* input) {
    if (!model || !input ||
        (int)model->weights.size() != model->centersCount)
        return std::numeric_limits<double>::quiet_NaN();

    double result = 0.0;

    for (int k = 0; k < model->centersCount; ++k) {
        result += model->weights[k] * gaussian(
            input,
            model->centers.data()+k*model->dimensions,
            model->dimensions,
            model->gamma
        );
    }

    return result;
}

DLLEXPORT double predict_rbf(RBFModel* model,
                             const double* input) {
    double result = decision_rbf(model, input);

    if (!std::isfinite(result))
        return result;

    return result > 0.0 ? 1.0 : -1.0;
}

DLLEXPORT double predict_rbf_regression(RBFModel* model,
                                        const double* input) {
    return decision_rbf(model, input);
}

DLLEXPORT int32_t get_num_rbf_centers(RBFModel* model) {
    return model ? model->centersCount : 0;
}

DLLEXPORT void get_rbf_centers(RBFModel* model,
                               double* out) {
    if (!model || !out)
        return;

    std::copy(model->centers.begin(),
              model->centers.end(), out);
}

DLLEXPORT void get_rbf_weights(RBFModel* model,
                               double* out) {
    if (!model || !out)
        return;

    std::copy(model->weights.begin(),
              model->weights.end(), out);
}

DLLEXPORT double get_rbf_bias(RBFModel* model) {
    return 0.0;
}

DLLEXPORT void destroy_rbf_model(RBFModel* model) {
    delete model;
}

}
