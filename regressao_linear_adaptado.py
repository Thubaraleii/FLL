
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Criar DataFrame com base nos dados apresentados
data = {
    "Estrutura": ["4BBBBB", "4BBBBB", "4BBB", "4BB", "4B", "4AA", "4A", "3DD", "3D", "3C", "3BB", "3B", "3A", "2B", "2A", "1D", "1C", "1B", "1A", "1A"],
    "Distância": [102, 97.5, 92.5, 87, 84, 79.5, 75, 68.5, 61, 58, 53, 51, 32.5, 26, 19.5, 40, 35, 45, 25, 2.5],
    "Espessura": [3.5, 5.5, 5.5, 3, 4.5, 4.5, 6.5, 7.5, 8, 8, 8, 5, 6.5, 6.5, 6.5, 5.5, 5.5, 4, 4.5, 2.5],
    "Coletas_2010": [15, 22.5, 22.5, 10, 11, 50, 100, 52, 52, 51, 120, 100, 57, 46, 40, 40, 71, 45, 56, 6],
    "MOA.T": [327, np.nan, np.nan, np.nan, 64, np.nan, 74, 286, 286, 258, np.nan, 286, 338, 258, 286, 286, 331, 266, 251, 190]
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
