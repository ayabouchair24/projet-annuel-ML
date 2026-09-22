#pragma once

#include <vector>

// Perceptron Multi-Couches generique : nombre de couches et de neurones par
// couche configurables. Activation des couches cachees configurable
// (Tanh ou Sigmoide), activation de sortie configurable (Tanh/Sigmoide pour
// la classification, Lineaire pour la regression). Entrainement par
// retropropagation du gradient, mise a jour des poids en ligne (un sample a
// la fois, ordre mélangé a chaque epoch).
//
// Repartition des taches (Membre 2) :
//   - Structure dynamique (constructeur, layer_sizes_, weights_)
//   - Propagation avant (forward)
//   - Fonctions d'activation (Tanh, Sigmoide) : apply_activation/derivative
//   - Accesseurs de structure (n_layers/layer_sizes) pour validation des
//     dimensions et des sorties sur entrees aleatoires (cf. notebook de
//     tests) sans avoir besoin d'entrainer le modele.
// La retropropagation (backward) et la boucle d'entrainement (fit) sont la
// partie de Membre 3.
class MLP {
public:
    enum class HiddenActivation { Tanh, Sigmoid };
    enum class OutputActivation { Tanh, Sigmoid, Linear };

    // layer_sizes = {n_entrees, n_neurones_couche_1, ..., n_sorties}
    explicit MLP(const std::vector<int>& layer_sizes,
                 OutputActivation output_activation = OutputActivation::Tanh,
                 HiddenActivation hidden_activation = HiddenActivation::Tanh);

    // Y : une cible par sample, meme dimension que la couche de sortie.
    // Renvoie l'historique de la MSE moyenne par epoch (courbe de convergence).
    std::vector<double> fit(const std::vector<std::vector<double>>& X,
                             const std::vector<std::vector<double>>& Y,
                             int epochs,
                             double learning_rate);

    std::vector<double> predict_raw(const std::vector<double>& x) const;

    // Sortie unique : classe = signe de la sortie.
    int predict_binary(const std::vector<double>& x) const;

    // Sorties multiples (one-hot) : classe = index du score le plus haut.
    int predict_multiclass(const std::vector<double>& x) const;

    // --- Introspection de structure (Membre 2) ---
    // Utile pour valider les dimensions depuis Python/le notebook sans
    // dupliquer la configuration cote appelant.
    int n_layers() const { return static_cast<int>(layer_sizes_.size()); }
    const std::vector<int>& layer_sizes() const { return layer_sizes_; }
    int n_inputs() const { return layer_sizes_.front(); }
    int n_outputs() const { return layer_sizes_.back(); }

private:
    std::vector<double> forward(const std::vector<double>& x);
    void backward(const std::vector<double>& target, double learning_rate);

    static double apply_activation(HiddenActivation act, double z);
    static double apply_output_activation(OutputActivation act, double z);
    static double activation_derivative(HiddenActivation act, double a);
    static double output_activation_derivative(OutputActivation act, double a);

    std::vector<int> layer_sizes_;
    OutputActivation output_activation_;
    HiddenActivation hidden_activation_;

    // weights_[l][i][j] : poids de l'entree i (i == layer_sizes_[l] => biais)
    // de la couche l vers le neurone j de la couche l+1.
    std::vector<std::vector<std::vector<double>>> weights_;

    // activations_[l] : sorties de la couche l (activations_[0] = entree).
    std::vector<std::vector<double>> activations_;
};
