# Regressão Linear em Duas Dimensões

A regressão linear é modelada pela equação:

\[
y = ax + b
\]

Onde:
- \( y \) é o alvo (variável dependente)
- \( x \) é a única feature (variável independente)
- \( a, b \) são os coeficientes do modelo

Os coeficientes \( a \) e \( b \) são determinados minimizando uma função de erro.

## Distância e Resíduos

A diferença entre os valores preditos e os valores reais são chamados de **resíduos**.

## Método dos Mínimos Quadrados Ordinários (OLS)

O erro total é minimizado usando a soma dos quadrados dos resíduos:

\[
RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

Onde:
- \( y_i \) é o valor real
- \( \hat{y}_i \) é o valor predito

## Regressão Linear com Múltiplas Dimensões

Para múltiplas variáveis independentes, o modelo se generaliza para:

\[
y = a_1x_1 + a_2x_2 + \dots + a_nx_n + b
\]

## Coeficiente de Determinação (R-Squared)

O coeficiente de determinação \( R^2 \) mede o quanto da variabilidade dos dados é explicada pelo modelo:

\[
R^2 \in [0,1]
\]

Um valor próximo de 1 indica um bom ajuste do modelo.

## Métricas de Erro

### Erro Quadrático Médio (MSE)

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

### Raiz do Erro Quadrático Médio (RMSE)

\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]

O RMSE é mais interpretável pois está na mesma unidade dos valores preditos.

### Desequilibrio de classes -> Quando uma classe ocorre muito mais que a outra e a ocorrência de transações fraudulentas passa despercebida, falso positivos e falso negativos

\[
    Precision = \frac{truePositives}{truePositivies+falsePositives}
\]

\[
    recall = \frac{truePositives}{truePositivies+falseNegativies}
\]

\[
    f1Score: 2* \frac{precision*recall}{precision+recall}
\]

