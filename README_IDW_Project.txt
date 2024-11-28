
# Processamento e Análise de Dados com IDW

## Visão Geral
Este projeto realizou a interpolação de valores ausentes em um conjunto de dados geológicos utilizando o método **IDW (Inverse Distance Weighting)**. Os dados foram processados e gráficos foram gerados para visualizar as tendências entre a distância e os teores analisados.

## Dados Utilizados
A tabela inicial contém informações de:
- SUBNÍVEIS
- DISTÂNCIA
- TS, TOC, TN, Fe2O3, U/Th, Al2O3, TiO2

Os valores ausentes nas colunas foram preenchidos utilizando o método IDW com base na coluna de **DISTÂNCIA**.

## Passos Realizados
1. **Preparação dos Dados**:
   - Importação da tabela inicial.
   - Identificação de valores ausentes nas colunas.
2. **Interpolação com IDW**:
   - Implementação do método IDW para preencher os valores ausentes.
   - Cálculo dos pesos com base na distância e aplicação para interpolação.
3. **Exportação dos Resultados**:
   - Tabela completa exportada para o formato Excel.
4. **Geração de Gráficos**:
   - Gráficos gerados para as colunas `TS`, `TOC`, `TN`, `Fe2O3`, `U/Th`, `Al2O3`, `TiO2` em função da distância.

## Resultados
Os resultados estão disponíveis nos seguintes arquivos:
- **Tabela Final**: `tabela_refinada_idw.xlsx`
  - Tabela preenchida com os valores interpolados.
- **Gráficos**:
  - Gráficos de cada variável em função da distância.

## Como Utilizar
1. Faça o download do arquivo `tabela_refinada_idw.xlsx` para obter os dados processados.
2. Consulte os gráficos gerados para análise visual.

## Ferramentas Utilizadas
- **Python**: Para processamento e análise.
- **Bibliotecas**:
  - `pandas` e `numpy`: Manipulação de dados.
  - `matplotlib`: Geração de gráficos.

## Contato
Caso tenha dúvidas ou precise de suporte, entre em contato.
