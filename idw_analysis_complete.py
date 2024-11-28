
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Função para interpolação IDW
def idw_interpolation(df, target_col, distance_col, power=2):
    """
    Realiza interpolação IDW para preencher valores ausentes.
    """
    for i in df[df[target_col].isna()].index:  # Índices com valores ausentes
        espessura_alvo = df.loc[i, distance_col]  # Valor de referência (DISTÂNCIA)
        
        # Selecionar linhas com valores conhecidos
        known_values = df[df[target_col].notna()]
        
        # Distâncias e pesos
        distances = np.abs(known_values[distance_col] - espessura_alvo)
        weights = 1 / (distances ** power)  # Pesos inversamente proporcionais à distância^power
        
        # Normalizar pesos para evitar dominância
        weights /= weights.sum()

        # Verificar e evitar divisão por zero
        if np.isinf(weights).any():
            interpolated_value = known_values.loc[weights == np.inf, target_col].values[0]
        else:
            # Cálculo da interpolação IDW
            interpolated_value = np.sum(weights * known_values[target_col])
        
        # Preencher a célula ausente
        df.loc[i, target_col] = interpolated_value
    
    return df

# Simulação dos dados com base na última tabela fornecida
data = {
    'SUBNIVEIS': ['7E', '7D', '7C', '7B', '7A', '6B', '6A', '5C', '5B', '5A', '4B', '4A', '3A', '2B', '2A', '1D', '1C', '1B', '1A'],
    'DISTÂNCIA': [102, 97.5, 92.5, 87, 84, 79.5, 75, 68.5, 61, 53, 45, 38, 32.5, 26, 19.5, 14, 8, 4, 2.5],
    'TS': [2.1789, 2.1702, 2.1412, 2.0346, 1.9412, 2.9442, 1.49, 1.96, 2.73, np.nan, 4.54, 1.71, 1.24, 1.72, 1.35, 3.67, 2.62, np.nan, 2.62],
    'TOC': [11.337, 11.353, 11.4, 11.536, 11.532, 11.932, 11.93, 12.11, 11.7, np.nan, 12.01, 13.05, 8.79, 7.42, np.nan, 12.41, 7.42, 5.12, 12.97],
    'TN': [0.4433, 0.4432, 0.443, 0.4417, 0.4447, 0.4447, 0.43, 0.46, 0.44, np.nan, 0.43, 0.54, 0.54, 0.29, np.nan, 0.22, 0.54, np.nan, 0.54],
    'Fe2O3': [6.5703, 6.5778, 6.6081, 6.6876, 6.8138, 6.8188, 7.06, np.nan, 6.66, 6.41, np.nan, 5.23, 7.41, 5.99, 6.89, 5.78, 5.25, 5.52, 4.94],
    'U/Th': [1.8388, 1.8427, 1.8546, 1.8954, 1.9724, 1.974, 2.06, np.nan, 1.98, 1.82, np.nan, 1.81, 1.66, 1.35, 1.28, 1.11, 1.17, 1.51, 1.48],
    'Al2O3': [14.099, 14.097, 14.101, 14.108, 14.133, 14.143, 14.13, np.nan, 14.31, 13.69, np.nan, 14.1, 13.62, 15.01, 14.65, 13.95, 14.89, 13.38, 13.12],
    'TiO2': [0.582, 0.5799, 0.5809, 0.5784, 0.5701, 0.5701, 0.57, np.nan, 0.55, 0.59, np.nan, 0.61, 0.6, 0.61, 0.61, 0.62, 0.64, 0.61, 0.63]
}

df = pd.DataFrame(data)

# Aplicar IDW para todas as colunas com valores ausentes
columns_to_interpolate = ['TS', 'TOC', 'TN', 'Fe2O3', 'U/Th', 'Al2O3', 'TiO2']
for col in columns_to_interpolate:
    df = idw_interpolation(df, target_col=col, distance_col='DISTÂNCIA')

# Converter os resultados para Excel
file_path = 'tabela_refinada_idw_final.xlsx'
df.to_excel(file_path, index=False)

# Geração de gráficos para cada variável
for col in columns_to_interpolate:
    plt.figure(figsize=(8, 6))
    plt.plot(df['DISTÂNCIA'], df[col], marker='o', label=col)
    plt.xlabel('DISTÂNCIA', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.title(f'Gráfico de {col} vs DISTÂNCIA', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
