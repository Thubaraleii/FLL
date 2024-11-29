
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Simulação de dados
data = {
    "Estrutura": ["A", "B", "C", "D", "E"],
    "Distância": [10, 20, 30, 40, 50],
    "Espessura": [5, 10, 15, 20, 25],
    "Coletas_2010": [1, 2, 3, 4, 5],
    "MOA.T": [100, 200, None, 400, None]  # Exemplo de valores ausentes
}
df = pd.DataFrame(data)

# Codificar a coluna 'Estrutura'
encoder = OneHotEncoder(sparse=False)
estrutura_encoded = encoder.fit_transform(df[['Estrutura']])

# Combinar os dados codificados com as demais colunas de entrada
X = np.hstack([estrutura_encoded, df[['Distância', 'Espessura', 'Coletas_2010']].values])

# Selecionar a coluna-alvo com valores ausentes
y = df['MOA.T'].values

# Filtrar os dados para treino (sem valores ausentes na coluna alvo)
mask = ~np.isnan(y)
X_train, y_train = X[mask], y[mask]

# Treinar o modelo de regressão
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prever os valores ausentes
y_pred = regressor.predict(X)

# Preencher os valores ausentes
df['MOA.T'] = np.where(np.isnan(df['MOA.T']), y_pred, df['MOA.T'])

# Mostrar o DataFrame atualizado
print(df)
