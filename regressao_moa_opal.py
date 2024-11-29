
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Criar DataFrame com base nos dados apresentados
data = {
    "Subníveis": ["4BBBBB", "4BBBBB", "4BBB", "4BB", "4B", "4AA", "4A", "3DD", "3D", "3C", "3BB", "3B", "3A", "2B", "2A", "1D", "1C", "1B", "1A", "1A"],
    "Distância": [102, 97.5, 92.5, 87, 84, 79.5, 75, 68.5, 61, 53, 45, 38, 32.5, 26, 19.5, 40, 35, 4, 2.5, 2.5],
    "Espessura": [3.5, 5.5, 5.5, 3, 4.5, 4.5, 6.5, 7.5, 8, 8, 7, 5.5, 6.5, 6.5, 6.5, 5.5, 5.5, 2.5, 2.5, 2.5],
    "Fósseis": [25, 30, 35, 20, 35, 92, 71, 85, 92, 87, 254, 255, 180, 115, 110, 105, 72, 18, 14, 6],
    "MOA": [29, 27, 10, 54, 70, 64, 71, 74, 74, None, None, None, 89, None, None, None, 72, None, None, 6],
    "OPAL": [4, 10, 0, 7, 3, 1, 2, 5, 2, None, None, None, 1, None, None, None, 2, None, None, None]
}
df = pd.DataFrame(data)

# Separar os dados em features (X) e targets (y)
features = ["Distância", "Espessura", "Fósseis"]
X = df[features]

# Preencher valores ausentes nas features, se necessário
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Regressão para a coluna 'MOA'
y_moa = df["MOA"].values
mask_moa = ~np.isnan(y_moa)
regressor_moa = LinearRegression()
regressor_moa.fit(X[mask_moa], y_moa[mask_moa])
df["MOA"] = np.where(np.isnan(df["MOA"]), regressor_moa.predict(X), df["MOA"])

# Regressão para a coluna 'OPAL'
y_opal = df["OPAL"].values
mask_opal = ~np.isnan(y_opal)
regressor_opal = LinearRegression()
regressor_opal.fit(X[mask_opal], y_opal[mask_opal])
df["OPAL"] = np.where(np.isnan(df["OPAL"]), regressor_opal.predict(X), df["OPAL"])

# Mostrar o DataFrame atualizado
print(df)
