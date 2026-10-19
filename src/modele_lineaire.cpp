// ============================================================================
// ASSIA
// linear_model.cpp  -  Modèle linéaire (bibliothèque dynamique pour ctypes)
// Projet annuel IABD - classification d'images de fleurs
//
// Contenu :
//   - Classification : règle de Rosenblatt (avec pas d'apprentissage et élan)
//   - Régression     : pseudo-inverse (moindres carrés, avec petite régularisation)
//   - Sauvegarde / chargement du modèle
//
// Convention :
//   - Un modèle a n_in entrées et n_out sorties (n_out = nb de classes en
//     classification "one-hot" avec cibles -1 / +1).
//   - Les poids sont stockés dans un tableau de taille n_out * (n_in + 1) :
//     pour la sortie k, w[k*(n_in+1) + 0] est le biais, puis les n_in poids.
//   - Les matrices sont passées "aplaties" ligne par ligne (row-major) :
//     X[i*n_in + j] = composante j de l'exemple i
//     Y[i*n_out + k] = cible k de l'exemple i
//
// Compilation :
//   Linux/Mac : g++ -O2 -std=c++17 -shared -fPIC -o liblinear_model.so linear_model.cpp
//   Windows   : g++ -O2 -std=c++17 -shared -o linear_model.dll linear_model.cpp
//   (ou Visual Studio : projet "DLL", fichier unique)
// ============================================================================

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#include <algorithm>

#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct LinearModel {
    int n_in;
    int n_out;
    std::vector<double> w;  // n_out * (n_in + 1)

    LinearModel(int n_in_, int n_out_, unsigned seed)
        : n_in(n_in_), n_out(n_out_), w((size_t)n_out_ * (n_in_ + 1)) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        for (auto& v : w) v = dist(gen);
    }

    // Score brut (avant le signe) de la sortie k pour l'exemple x
    double score(const double* x, int k) const {
        const double* wk = &w[(size_t)k * (n_in + 1)];
        double s = wk[0];
        for (int j = 0; j < n_in; ++j) s += wk[j + 1] * x[j];
        return s;
    }
};

// ---------------------------------------------------------------------------
// Création / destruction
// ---------------------------------------------------------------------------
EXPORT void* create_linear_model(int n_in, int n_out, int seed) {
    if (n_in <= 0 || n_out <= 0) return nullptr;
    return new LinearModel(n_in, n_out, (unsigned)seed);
}

EXPORT void delete_linear_model(void* model) {
    delete static_cast<LinearModel*>(model);
}

EXPORT int get_n_in(void* model) { return static_cast<LinearModel*>(model)->n_in; }
EXPORT int get_n_out(void* model) { return static_cast<LinearModel*>(model)->n_out; }

// ---------------------------------------------------------------------------
// Accès aux poids (copie vers / depuis un tableau fourni par Python)
// ---------------------------------------------------------------------------
EXPORT void get_weights(void* model, double* out) {
    auto* m = static_cast<LinearModel*>(model);
    std::copy(m->w.begin(), m->w.end(), out);
}

EXPORT void set_weights(void* model, const double* in) {
    auto* m = static_cast<LinearModel*>(model);
    std::copy(in, in + m->w.size(), m->w.begin());
}

// ---------------------------------------------------------------------------
// Prédiction : scores bruts. Python applique argmax / signe.
// ---------------------------------------------------------------------------
EXPORT void predict_scores(void* model, const double* X, int n_samples, double* out) {
    auto* m = static_cast<LinearModel*>(model);
    for (int i = 0; i < n_samples; ++i)
        for (int k = 0; k < m->n_out; ++k)
            out[(size_t)i * m->n_out + k] = m->score(X + (size_t)i * m->n_in, k);
}

// ---------------------------------------------------------------------------
// Classification : règle de Rosenblatt
//   Pour chaque exemple et chaque sortie k :
//       y_hat = signe(w_k . x)        (+1 si score >= 0, sinon -1)
//       delta = lr * (y - y_hat) * x  (+ momentum * delta précédent)
//       w_k  += delta
//   Les cibles Y valent -1 ou +1.
//   history (taille epochs, peut être nullptr) reçoit, pour chaque epoch, la
//   proportion d'exemples pour lesquels au moins une sortie était fausse.
//   Retourne le nombre d'epochs réellement effectuées (arrêt si erreur = 0).
// ---------------------------------------------------------------------------
EXPORT int train_rosenblatt(void* model, const double* X, const double* Y,
                            int n_samples, double lr, double momentum,
                            int epochs, int seed, double* history) {
    auto* m = static_cast<LinearModel*>(model);
    const int d = m->n_in + 1;
    std::vector<int> order(n_samples);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 gen((unsigned)seed);
    std::vector<double> prev_delta(m->w.size(), 0.0);

    int done = 0;
    for (int e = 0; e < epochs; ++e) {
        std::shuffle(order.begin(), order.end(), gen);
        int wrong_samples = 0;
        for (int idx = 0; idx < n_samples; ++idx) {
            const int i = order[idx];
            const double* x = X + (size_t)i * m->n_in;
            bool any_wrong = false;
            for (int k = 0; k < m->n_out; ++k) {
                const double y = Y[(size_t)i * m->n_out + k];
                const double y_hat = (m->score(x, k) >= 0.0) ? 1.0 : -1.0;
                double* wk = &m->w[(size_t)k * d];
                double* pk = &prev_delta[(size_t)k * d];
                const double err = y - y_hat;  // 0, +2 ou -2
                if (err != 0.0) any_wrong = true;
                // biais
                double dlt = lr * err + momentum * pk[0];
                wk[0] += dlt; pk[0] = dlt;
                for (int j = 0; j < m->n_in; ++j) {
                    dlt = lr * err * x[j] + momentum * pk[j + 1];
                    wk[j + 1] += dlt; pk[j + 1] = dlt;
                }
            }
            if (any_wrong) ++wrong_samples;
        }
        if (history) history[e] = (double)wrong_samples / n_samples;
        ++done;
        if (wrong_samples == 0) break;  // convergence : tout est bien classé
    }
    return done;
}

// ---------------------------------------------------------------------------
// Régression (ou classification par moindres carrés) : pseudo-inverse
//   W = (X^T X + ridge * I)^-1 X^T Y      où X contient une colonne de 1 (biais)
//   La régularisation ridge (très petite par défaut, ex. 1e-8) évite les
//   matrices singulières. Résolution par élimination de Gauss-Jordan avec
//   pivot partiel.
//   Retourne 0 si OK, -1 si la matrice est singulière.
// ---------------------------------------------------------------------------
EXPORT int train_pseudo_inverse(void* model, const double* X, const double* Y,
                                int n_samples, double ridge) {
    auto* m = static_cast<LinearModel*>(model);
    const int d = m->n_in + 1;
    const int K = m->n_out;

    // A = X^T X + ridge*I  (d x d) ; B = X^T Y  (d x K)
    std::vector<double> A((size_t)d * d, 0.0), B((size_t)d * K, 0.0);
    std::vector<double> xi(d);
    for (int i = 0; i < n_samples; ++i) {
        xi[0] = 1.0;
        for (int j = 0; j < m->n_in; ++j) xi[j + 1] = X[(size_t)i * m->n_in + j];
        for (int a = 0; a < d; ++a) {
            const double va = xi[a];
            for (int b = a; b < d; ++b) A[(size_t)a * d + b] += va * xi[b];
            for (int k = 0; k < K; ++k) B[(size_t)a * K + k] += va * Y[(size_t)i * K + k];
        }
    }
    for (int a = 0; a < d; ++a) {
        for (int b = 0; b < a; ++b) A[(size_t)a * d + b] = A[(size_t)b * d + a];  // symétrie
        A[(size_t)a * d + a] += ridge;
    }

    // Gauss-Jordan sur [A | B]
    for (int col = 0; col < d; ++col) {
        int piv = col;
        double best = std::fabs(A[(size_t)col * d + col]);
        for (int r = col + 1; r < d; ++r) {
            double v = std::fabs(A[(size_t)r * d + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-14) return -1;  // singulière
        if (piv != col) {
            for (int c = 0; c < d; ++c) std::swap(A[(size_t)piv * d + c], A[(size_t)col * d + c]);
            for (int c = 0; c < K; ++c) std::swap(B[(size_t)piv * K + c], B[(size_t)col * K + c]);
        }
        const double inv = 1.0 / A[(size_t)col * d + col];
        for (int c = 0; c < d; ++c) A[(size_t)col * d + c] *= inv;
        for (int c = 0; c < K; ++c) B[(size_t)col * K + c] *= inv;
        for (int r = 0; r < d; ++r) {
            if (r == col) continue;
            const double f = A[(size_t)r * d + col];
            if (f == 0.0) continue;
            for (int c = col; c < d; ++c) A[(size_t)r * d + c] -= f * A[(size_t)col * d + c];
            for (int c = 0; c < K; ++c) B[(size_t)r * K + c] -= f * B[(size_t)col * K + c];
        }
    }
    // B contient maintenant W (d x K) ; on range dans m->w (K x d)
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < d; ++j)
            m->w[(size_t)k * d + j] = B[(size_t)j * K + k];
    return 0;
}

// ---------------------------------------------------------------------------
// Sauvegarde / chargement (fichier binaire : n_in, n_out, poids)
// ---------------------------------------------------------------------------
EXPORT int save_linear_model(void* model, const char* path) {
    auto* m = static_cast<LinearModel*>(model);
    FILE* f = std::fopen(path, "wb");
    if (!f) return -1;
    std::fwrite(&m->n_in, sizeof(int), 1, f);
    std::fwrite(&m->n_out, sizeof(int), 1, f);
    std::fwrite(m->w.data(), sizeof(double), m->w.size(), f);
    std::fclose(f);
    return 0;
}

EXPORT void* load_linear_model(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return nullptr;
    int n_in = 0, n_out = 0;
    if (std::fread(&n_in, sizeof(int), 1, f) != 1 || std::fread(&n_out, sizeof(int), 1, f) != 1 ||
        n_in <= 0 || n_out <= 0) {
        std::fclose(f);
        return nullptr;
    }
    auto* m = new LinearModel(n_in, n_out, 0);
    size_t read = std::fread(m->w.data(), sizeof(double), m->w.size(), f);
    std::fclose(f);
    if (read != m->w.size()) { delete m; return nullptr; }
    return m;
}
