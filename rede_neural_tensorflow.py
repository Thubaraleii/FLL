
# Instale o TensorFlow (caso necessário)
# !pip install tensorflow

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Simulação de um DataFrame
# Substitua este exemplo pelos seus dados reais
data = {
    "Estrutura": ["A", "B", "C", "D", "E"],
    "Distância": [10, 20, 30, 40, 50],
    "Espessura": [5, 10, 15, 20, 25],
    "Coletas_2010": [1, 2, 3, 4, 5],
    "MOA.T": [100, 200, None, 400, None]  # Exemplo de valores ausentes
}
df = pd.DataFrame(data)

# Preparar os dados
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
