
# Instale o TensorFlow (caso necessário)
# !pip install tensorflow

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Criar DataFrame com base nos dados apresentados
data = {
    "Estrutura": ["4BBBBB", "4BBBBB", "4BBB", "4BB", "4B", "4AA", "4A", "3DD", "3D", "3C", "3BB", "3B", "3A", "2B", "2A", "1D", "1C", "1B", "1A", "1A"],
    "Distância": [102, 97.5, 92.5, 87, 84, 79.5, 75, 68.5, 61, 58, 53, 51, 32.5, 26, 19.5, 40, 35, 45, 25, 2.5],
    "Espessura": [3.5, 5.5, 5.5, 3, 4.5, 4.5, 6.5, 7.5, 8, 8, 8, 5, 6.5, 6.5, 6.5, 5.5, 5.5, 4, 4.5, 2.5],
    "Coletas_2010": [15, 22.5, 22.5, 10, 11, 50, 100, 52, 52, 51, 120, 100, 57, 46, 40, 40, 71, 45, 56, 6],
    "MOA.T": [327, np.nan, np.nan, np.nan, 64, np.nan, 74, 286, 286, 258, np.nan, 286, 338, 258, 286, 286, 331, 266, 251, 190]
}
df = pd.DataFrame(data)

# Codificar variáveis categóricas (se houver)
encoder = OneHotEncoder(sparse=False)
estrutura_encoded = encoder.fit_transform(df[['Estrutura']])

# Combinar as colunas codificadas com as demais features
X = np.hstack([estrutura_encoded, df[['Distância', 'Espessura', 'Coletas_2010']].values])

# Selecionar a coluna-alvo com valores ausentes
y = df['MOA.T'].values

# Dividir os dados em treino e teste, considerando apenas linhas com valores conhecidos
mask = ~np.isnan(y)
X_train, X_test, y_train, y_test = train_test_split(X[mask], y[mask], test_size=0.2, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar o modelo de Rede Neural
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Saída única para regressão
])

# Compilar o modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Treinar o modelo
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=8)

# Prever os valores ausentes
X_full = scaler.transform(X)
y_pred = model.predict(X_full)

# Preencher os valores ausentes
df['MOA.T'] = np.where(np.isnan(df['MOA.T']), y_pred.flatten(), df['MOA.T'])

# Mostrar o DataFrame atualizado
print(df)
