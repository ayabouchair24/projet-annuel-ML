// Fichier d'Aya
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "osqp.h"

#if defined(_WIN32) || defined(__WIN32__)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

// Noyaux du cours : lineaire (produit scalaire), polynomial (1 + Xn^T Xm)^Q, radial (RBF)
enum KernelType {
    LINEAR = 0,
    POLYNOMIAL = 1,
    RBF = 2
};

struct SVMModel {
    int32_t numDimensions;
    KernelType kernelType;
    double gamma;              // RBF : exp(-gamma * ||Xn - Xm||^2), gamma = 1 donne le noyau du cours
    int32_t degree;            // Q dans le cours (polynomial)
    int32_t numSupportVectors;
    double* supportVectors;    // numSupportVectors x numDimensions, ligne par ligne
    double* alphasY;           // alpha_n * Y_n pour chaque vecteur support
    double W0;                 // biais w0 du cours
};

static double scalarProduct(const double* Xn, const double* Xm, int32_t numDimensions) {
    double result = 0.0;
    for (int32_t d = 0; d < numDimensions; ++d) result += Xn[d] * Xm[d];
    return result;
}

static double kernel(const double* Xn, const double* Xm, const SVMModel* model) {
    if (model->kernelType == LINEAR) {
        return scalarProduct(Xn, Xm, model->numDimensions);
    }
    if (model->kernelType == POLYNOMIAL) {
        return std::pow(1.0 + scalarProduct(Xn, Xm, model->numDimensions), model->degree);
    }
    double squaredDistance = 0.0;
    for (int32_t d = 0; d < model->numDimensions; ++d) {
        double difference = Xn[d] - Xm[d];
        squaredDistance += difference * difference;
    }
    return std::exp(-model->gamma * squaredDistance);
}

static void freeSupportVectors(SVMModel* model) {
    delete[] model->supportVectors;
    delete[] model->alphasY;
    model->supportVectors = nullptr;
    model->alphasY = nullptr;
    model->numSupportVectors = 0;
}

extern "C" {

DLLEXPORT SVMModel* create_svm_model(int32_t numDimensions, double gamma, int32_t degree, int32_t kernelType) {
    SVMModel* model = new SVMModel();
    model->numDimensions = numDimensions;
    model->gamma = gamma;
    model->degree = degree;
    model->kernelType = static_cast<KernelType>(kernelType);
    model->numSupportVectors = 0;
    model->supportVectors = nullptr;
    model->alphasY = nullptr;
    model->W0 = 0.0;
    return model;
}


DLLEXPORT int32_t train_svm_model(SVMModel* model, const double* X, const double* Y,
                                  int32_t numSamples, int32_t numDimensions, double C) {
    freeSupportVectors(model);
    model->numDimensions = numDimensions;
    const int N = numSamples;


    std::vector<c_float> BigMatrixValues;
    std::vector<c_int> BigMatrixRows;
    std::vector<c_int> BigMatrixColumnStarts(N + 1, 0);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i <= j; ++i) {
            BigMatrixValues.push_back(Y[i] * Y[j] * kernel(&X[i * numDimensions], &X[j * numDimensions], model));
            BigMatrixRows.push_back(i);
        }
        BigMatrixColumnStarts[j + 1] = (c_int)BigMatrixValues.size();
    }

    // ETAPE 2 : minimiser 1/2 alphas^T BigMatrix alphas + (-1,...,-1).alphas
    //           sous Y^T alphas = 0 et alphas >= 0 (et <= C en marge souple)
    std::vector<c_float> minusOnes(N, -1.0);

    // Contraintes : ligne 0 = Y^T alphas, lignes 1..N = identite
    std::vector<c_float> constraintValues;
    std::vector<c_int> constraintRows;
    std::vector<c_int> constraintColumnStarts(N + 1, 0);
    for (int j = 0; j < N; ++j) {
        constraintValues.push_back(Y[j]);
        constraintRows.push_back(0);
        constraintValues.push_back(1.0);
        constraintRows.push_back(1 + j);
        constraintColumnStarts[j + 1] = (c_int)constraintValues.size();
    }
    std::vector<c_float> lowerBounds(N + 1, 0.0);
    std::vector<c_float> upperBounds(N + 1, 0.0);
    const bool softMargin = (C > 0.0);
    for (int j = 1; j <= N; ++j) upperBounds[j] = softMargin ? C : OSQP_INFTY;

    csc P;
    P.nzmax = (c_int)BigMatrixValues.size(); P.m = N; P.n = N;
    P.x = BigMatrixValues.data(); P.i = BigMatrixRows.data(); P.p = BigMatrixColumnStarts.data(); P.nz = -1;
    csc A;
    A.nzmax = (c_int)constraintValues.size(); A.m = N + 1; A.n = N;
    A.x = constraintValues.data(); A.i = constraintRows.data(); A.p = constraintColumnStarts.data(); A.nz = -1;

    OSQPData data;
    data.n = N;
    data.m = N + 1;
    data.P = &P;
    data.q = minusOnes.data();
    data.A = &A;
    data.l = lowerBounds.data();
    data.u = upperBounds.data();

    OSQPSettings settings;
    osqp_set_default_settings(&settings);
    settings.verbose = 0;
    settings.polish = 1;
    settings.eps_abs = 1e-6;
    settings.eps_rel = 1e-6;
    settings.max_iter = 20000;

    OSQPWorkspace* work = nullptr;
    c_int exitFlag = osqp_setup(&work, &data, &settings);
    if (exitFlag != 0) return (int32_t)exitFlag;
    osqp_solve(work);
    const int32_t status = (int32_t)work->info->status_val;
    std::vector<double> alphas(work->solution->x, work->solution->x + N);
    osqp_cleanup(work);
    const int32_t OSQP_MAX_ITER = -2; // OSQP_MAX_ITER_REACHED : la solution courante reste utilisable
    if (status != OSQP_SOLVED && status != OSQP_SOLVED_INACCURATE && status != OSQP_MAX_ITER) return status;

    // ETAPE 3 : garder les vecteurs supports (alpha > 0.000001, comme svPos dans le notebook du cours)
    // OSQP est iteratif : les alphas "nuls" valent ~1e-7 et non exactement 0.
    // Seuil du notebook (1e-6), rendu relatif a la plus grande valeur d'alpha.
    double maxAlpha = 0.0;
    for (int i = 0; i < N; ++i) if (alphas[i] > maxAlpha) maxAlpha = alphas[i];
    const double supportVectorThreshold = std::max(1e-6, 1e-4 * maxAlpha);
    std::vector<int> supportVectorIndices;
    for (int i = 0; i < N; ++i) {
        if (alphas[i] > supportVectorThreshold) supportVectorIndices.push_back(i);
    }
    model->numSupportVectors = (int32_t)supportVectorIndices.size();
    model->supportVectors = new double[model->numSupportVectors * numDimensions];
    model->alphasY = new double[model->numSupportVectors];
    for (int s = 0; s < model->numSupportVectors; ++s) {
        int i = supportVectorIndices[s];
        for (int d = 0; d < numDimensions; ++d) {
            model->supportVectors[s * numDimensions + d] = X[i * numDimensions + d];
        }
        model->alphasY[s] = alphas[i] * Y[i];
    }

    // ETAPE 4 : W0 = 1 / Y_svPos - sum_k alphas_k * Y_k * K(X_k, X_svPos)
    // En marge souple, il faut un vecteur support strictement sur la marge (alpha < C)
    int svPos = -1;
    for (int i : supportVectorIndices) {
        if (!softMargin || alphas[i] < C - supportVectorThreshold) { svPos = i; break; }
    }
    if (svPos < 0 && !supportVectorIndices.empty()) svPos = supportVectorIndices[0];
    if (svPos >= 0) {
        double sum = 0.0;
        for (int s = 0; s < model->numSupportVectors; ++s) {
            sum += model->alphasY[s] * kernel(&model->supportVectors[s * numDimensions],
                                              &X[svPos * numDimensions], model);
        }
        model->W0 = 1.0 / Y[svPos] - sum;
    } else {
        model->W0 = 0.0;
    }
    return (status == OSQP_SOLVED) ? 0 : 1; // 0 = converge, 1 = solution approchee
}

DLLEXPORT double decision_svm(SVMModel* model, const double* input) {
    double result = model->W0;
    for (int s = 0; s < model->numSupportVectors; ++s) {
        result += model->alphasY[s] * kernel(&model->supportVectors[s * model->numDimensions], input, model);
    }
    return result;
}

DLLEXPORT double predict_svm(SVMModel* model, const double* input) {
    return decision_svm(model, input) > 0.0 ? 1.0 : -1.0;
}

DLLEXPORT int32_t get_num_support_vectors(SVMModel* model) {
    return model->numSupportVectors;
}

DLLEXPORT void get_support_vectors(SVMModel* model, double* out) {
    for (int k = 0;
DLLEXPORT void destroy_svm_model(SVMModel* model) {
    if (!model) return;
    freeSupportVectors(model);
    delete model;
}

struct SVRModel {
    SVMModel base;
};
