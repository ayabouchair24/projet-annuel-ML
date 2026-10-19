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
    for (int k = 0; k < model->numSupportVectors * model->numDimensions; ++k) out[k] = model->supportVectors[k];
}

DLLEXPORT void get_alphas_y(SVMModel* model, double* out) {
    for (int s = 0; s < model->numSupportVectors; ++s) out[s] = model->alphasY[s];
}

DLLEXPORT double get_W0(SVMModel* model) {
    return model->W0;
}

DLLEXPORT void destroy_svm_model(SVMModel* model) {
    if (!model) return;
    freeSupportVectors(model);
    delete model;
}

struct SVRModel {
    SVMModel base;
};

DLLEXPORT SVRModel* create_svr_model(int32_t dimensions, double gamma, int32_t degree, int32_t kernelType) {
    SVRModel* m = new SVRModel{};
    m->base.numDimensions = dimensions;
    m->base.kernelType = static_cast<KernelType>(kernelType);
    m->base.gamma = gamma;
    m->base.degree = degree;
    m->base.numSupportVectors = 0;
    m->base.supportVectors = nullptr;
    m->base.alphasY = nullptr;
    m->base.W0 = 0.0;
    return m;
}

DLLEXPORT int32_t train_svr_model(SVRModel* model, const double* X, const double* Y,
                                  int32_t numSamples, int32_t numDimensions,
                                  double C, double epsilon) {
    if (!model || !X || !Y || numSamples < 2 || numDimensions < 1 ||
        !(C > 0.0) || !(epsilon >= 0.0)) return -100;
    SVMModel* b = &model->base;
    freeSupportVectors(b);
    b->W0 = 0.0;
    b->numDimensions = numDimensions;
    const int n = numSamples, vars = 2 * n;
    // z = [alpha_0..alpha_(n-1), alpha*_0..alpha*_(n-1)]
    // beta_i = z_i - z_(n+i)
    std::vector<c_float> pv;
    std::vector<c_int> pi, pp(vars + 1, 0);
    for (int j = 0; j < vars; ++j) {
        int jj = j % n;
        double sj = j < n ? 1.0 : -1.0;
        for (int i = 0; i <= j; ++i) {
            int ii = i % n;
            double si = i < n ? 1.0 : -1.0;
            pv.push_back(si * sj * kernel(X + ii * numDimensions, X + jj * numDimensions, b));
            pi.push_back(i);
        }
        pp[j + 1] = (c_int)pv.size();
    }
    std::vector<c_float> q(vars);
    for (int i = 0; i < n; ++i) {
        q[i] = epsilon - Y[i];
        q[n + i] = epsilon + Y[i];
    }
    std::vector<c_float> av;
    std::vector<c_int> ai, ap(vars + 1, 0);
    for (int j = 0; j < vars; ++j) {
        av.push_back(j < n ? 1.0 : -1.0);
        ai.push_back(0);
        av.push_back(1.0);
        ai.push_back(j + 1);
        ap[j + 1] = (c_int)av.size();
    }
    std::vector<c_float> lower(vars + 1, 0.0), upper(vars + 1, C);
    upper[0] = 0.0;
    csc P{};
    P.nzmax = (c_int)pv.size(); P.m = vars; P.n = vars;
    P.x = pv.data(); P.i = pi.data(); P.p = pp.data(); P.nz = -1;
    csc A{};
    A.nzmax = (c_int)av.size(); A.m = vars + 1; A.n = vars;
    A.x = av.data(); A.i = ai.data(); A.p = ap.data(); A.nz = -1;
    OSQPData data{};
    data.n = vars; data.m = vars + 1; data.P = &P;
    data.q = q.data(); data.A = &A;
    data.l = lower.data(); data.u = upper.data();
    OSQPSettings settings;
    osqp_set_default_settings(&settings);
    settings.verbose = 0;
    settings.polish = 1;
    settings.eps_abs = 1e-6;
    settings.eps_rel = 1e-6;
    settings.max_iter = 20000;
    OSQPWorkspace* work = nullptr;
    c_int flag = osqp_setup(&work, &data, &settings);
    if (flag != 0) { if (work) osqp_cleanup(work); return (int32_t)flag; }
    osqp_solve(work);
    int32_t status = (int32_t)work->info->status_val;
    if ((status != OSQP_SOLVED && status != OSQP_SOLVED_INACCURATE) ||
        !work->solution || !work->solution->x) {
        osqp_cleanup(work);
        return status;
    }
    std::vector<double> alpha(n), alphaStar(n), beta(n);
    for (int i = 0; i < n; ++i) {
        alpha[i] = std::max(0.0, (double)work->solution->x[i]);
        alphaStar[i] = std::max(0.0, (double)work->solution->x[n + i]);
        beta[i] = alpha[i] - alphaStar[i];
    }
    osqp_cleanup(work);
    double maxBeta = 0.0;
    for (double v : beta) maxBeta = std::max(maxBeta, std::abs(v));
    double threshold = std::max(1e-7, 1e-5 * maxBeta);
    std::vector<int> indices;
    for (int i = 0; i < n; ++i)
        if (std::abs(beta[i]) > threshold) indices.push_back(i);
    b->numSupportVectors = (int32_t)indices.size();
    b->supportVectors = new double[indices.size() * numDimensions];
    b->alphasY = new double[indices.size()];
    for (int s = 0; s < (int)indices.size(); ++s) {
        int i = indices[s];
        std::copy(X + i * numDimensions, X + (i + 1) * numDimensions,
                  b->supportVectors + s * numDimensions);
        b->alphasY[s] = beta[i];
    }

    double biasSum = 0.0;
    int biasCount = 0;
    const double tol = std::max(1e-6, 1e-4 * C);
    for (int i = 0; i < n; ++i) {
        double f = 0.0;
        for (int s = 0; s < b->numSupportVectors; ++s)
            f += b->alphasY[s] * kernel(b->supportVectors + s * numDimensions,
                                        X + i * numDimensions, b);
        if (alpha[i] > tol && alpha[i] < C - tol) {
            biasSum += Y[i] - epsilon - f;
            ++biasCount;
        }
        if (alphaStar[i] > tol && alphaStar[i] < C - tol) {
            biasSum += Y[i] + epsilon - f;
            ++biasCount;
        }
    }
    if (biasCount) b->W0 = biasSum / biasCount;
    else {

        double lo = -OSQP_INFTY, hi = OSQP_INFTY;
        for (int i = 0; i < n; ++i) {
            double f = 0.0;
            for (int s = 0; s < b->numSupportVectors; ++s)
                f += b->alphasY[s] * kernel(b->supportVectors + s * numDimensions,
                                            X + i * numDimensions, b);
            double u = Y[i] - epsilon - f;
            double v = Y[i] + epsilon - f;
            if (alpha[i] <= tol) lo = std::max(lo, u);
            if (alpha[i] >= C - tol) hi = std::min(hi, u);
            if (alphaStar[i] <= tol) hi = std::min(hi, v);
            if (alphaStar[i] >= C - tol) lo = std::max(lo, v);
        }
        if (lo > -OSQP_INFTY / 2 && hi < OSQP_INFTY / 2)
            b->W0 = 0.5 * (lo + hi);
        else if (lo > -OSQP_INFTY / 2) b->W0 = lo;
        else if (hi < OSQP_INFTY / 2) b->W0 = hi;
        else b->W0 = 0.0;
    }
    return status == OSQP_SOLVED ? 0 : 1;
}

DLLEXPORT double predict_svr(SVRModel* model, const double* input) {
    return model && input ? decision_svm(&model->base, input) : 0.0;
}

DLLEXPORT int32_t get_num_support_vectors_svr(SVRModel* model) {
    return model ? model->base.numSupportVectors : 0;
}

DLLEXPORT void destroy_svr_model(SVRModel* model) {
    if (!model) return;
    freeSupportVectors(&model->base);
    delete model;
}


}