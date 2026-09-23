import matplotlib.pyplot as plt

from mlp_bindings import MLP

# XOR : impossible à résoudre avec une seule séparation linéaire.
X = [
    [-1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
]

Y = [
    [-1.0],
    [-1.0],
    [1.0],
    [1.0],
]

model = MLP([2, 2, 1])

print("Architecture :", model.layer_sizes)
print("Loss avant entraînement :", model.loss(X, Y))

losses = []

# L'API C expose l'entraînement C++ par blocs d'epochs.
for _ in range(50):
    model.train(X, Y, epochs=100, learning_rate=0.05)
    losses.append(model.loss(X, Y))

print("Loss après entraînement :", losses[-1])
print("Accuracy XOR :", model.accuracy_binary(X, Y) * 100, "%")

for x, y in zip(X, Y):
    raw = model.predict(x)[0]
    predicted = 1 if raw >= 0 else -1
    print(f"{x} -> sortie={raw:.4f}, attendu={int(y[0])}, prédit={predicted}")

plt.plot(range(1, len(losses) + 1), losses)
plt.xlabel("Blocs de 100 epochs")
plt.ylabel("MSE")
plt.title("Apprentissage du PMC sur XOR")
plt.grid(True)
plt.tight_layout()
plt.show()
