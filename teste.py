import tensorflow as tf
import numpy as np

# Dados do problema
indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
pesos = [40, 48, 13, 24, 10, 48, 1, 13, 23, 30, 5, 35, 13, 6, 33, 6, 46, 10, 48, 46, 21, 29, 44, 6, 47, 22, 13, 21, 24, 37, 29, 43, 21, 14, 37, 40, 43, 27, 48, 18, 12, 17, 34, 12, 47, 18, 1, 7, 38, 32, 43, 1, 40, 39, 39, 48, 20, 5, 16, 28, 4, 26, 23, 18, 11, 33, 41, 25, 3, 1, 1, 19, 37, 9, 40, 37, 4, 6, 4, 37, 11, 43, 27, 38, 38, 7, 18, 38, 3, 45, 29, 18, 41, 24, 32, 48, 22, 36, 27, 7]
estoques = [6, 2, 6, 2, 5, 5, 9, 8, 9, 9, 1, 2, 3, 8, 1, 2, 1, 3, 5, 7, 3, 7, 2, 5, 5, 2, 1, 6, 7, 2, 7, 6, 3, 5, 5, 7, 7, 6, 9, 4, 1, 4, 2, 8, 9, 5, 6, 2, 9, 4, 9, 7, 5, 1, 6, 9, 9, 3, 3, 4, 4, 8, 4, 5, 5, 6, 7, 3, 2, 9, 2, 7, 3, 8, 1, 6, 1, 1, 5, 2, 3, 1, 3, 1, 9, 1, 7, 2, 1, 4, 2, 8, 9, 3, 4, 5, 6, 4, 9, 1]
valores = [395, 268, 204, 289, 303, 232, 245, 270, 217, 293, 413, 467, 335, 423, 373, 234, 459, 487, 264, 285, 469, 444, 389, 423, 305, 452, 107, 116, 130, 368, 160, 344, 452, 439, 125, 327, 145, 305, 231, 136, 447, 336, 140, 252, 168, 221, 159, 201, 272, 110, 165, 397, 350, 261, 346, 277, 475, 389, 340, 145, 486, 236, 459, 251, 473, 126, 433, 245, 438, 427, 283, 423, 439, 407, 212, 436, 228, 283, 165, 118, 183, 127, 276, 101, 370, 190, 405, 432, 203, 272, 222, 313, 106, 289, 478, 424, 119, 185, 154, 232]
capacidade_maxima = 500.0

# Normalização dos dados
pesos_norm = np.array(pesos) / max(pesos)
estoques_norm = np.array(estoques) / max(estoques)
valores_norm = np.array(valores) / max(valores)

# Criando o modelo de rede neural simples
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=len(indices)*3, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilando o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Criando os dados de entrada e saída
X = np.array(list(zip(pesos_norm, estoques_norm, valores_norm)))
y = np.array([1] * len(indices))  # Todos os itens são inicialmente incluídos

# Treinando o modelo
model.fit(X, y, epochs=100, verbose=1)

# Obtendo as previsões
predictions = model.predict(X)

# Arredondando as previsões para obter 0 ou 1
binary_predictions = np.round(predictions)

# Exibindo as previsões
print(binary_predictions)
