# Aprendizado de Máquina

## O que é?

Aprendizado de máquina é a **capacidade de uma máquina aprender padrões através de dados fornecidos por humanos**.  
É a ciência (e a arte) de programar computadores para que eles possam **aprender com os dados**.

## Por que usar?

Aplicar técnicas de aprendizado de máquina para analisar grandes quantidades de dados pode ajudar na **descoberta de padrões não explícitos** — esse processo é conhecido como **mineração de dados**.

Também é útil para:
- Resolver **problemas complexos demais** para algoritmos tradicionais ou análise humana direta.
- Automatizar a **tomada de decisões** com base em dados históricos.
- Permitir que sistemas se tornem **adaptáveis**.

## Vantagens

- Resolver problemas complexos
- Adaptabilidade
- Descoberta de padrões

---

## Tipos de Aprendizado de Máquina

### 1. Aprendizado Supervisionado
- **Dados rotulados (com labels)**.
- O objetivo é prever um valor ou categoria com base em entradas conhecidas.

#### Exemplos:
- **Classificação**: poucas saídas possíveis (valores discretos).
- **Regressão**: muitas saídas possíveis (valores contínuos).

> **Nota:** A *regressão logística* retorna a probabilidade de pertencer a uma determinada classe (apesar do nome, é usada para classificação).

#### Técnicas comuns:
- K-vizinhos mais próximos (KNN)
- Regressão linear
- Regressão logística
- Máquinas de vetores de suporte (SVM)
- Árvores de decisão e florestas aleatórias (Random Forests)
- Redes neurais

---

### 2. Aprendizado Não Supervisionado
- **Sem dados rotulados**.
- O objetivo é encontrar padrões, grupos ou estruturas nos dados.

#### Técnicas:
- **Clusterização**:
  - K-Means (K-médias)
  - DBSCAN (baseado em densidade)
  - Análise de cluster hierárquica (HCA)

- **Detecção de anomalias**:
  - One-Class SVM
  - Floresta de isolamento (Isolation Forest)

- **Redução de dimensionalidade / visualização**:
  - PCA (Análise de Componentes Principais)
  - Kernel PCA
  - LLE (Locally Linear Embedding)
  - t-SNE (t-distributed Stochastic Neighbor Embedding)

- **Regras de associação**:
  - Apriori
  - Eclat

---

### 3. Aprendizado Semi-Supervisionado
- Parte dos dados está rotulada.
- Útil quando o rotulamento completo é caro ou difícil.

---

## Aprendizado Offline vs Online

### Aprendizado Offline (Batch)
- Treinamento com todos os dados disponíveis de uma vez.
- Mais rápido e preciso na inferência.
- Treinamento mais demorado e menos flexível.
- Requer retrain completo para incorporar novos dados.

### Aprendizado Online
- Aprendizado contínuo, com dados chegando aos poucos.
- Mais flexível e adaptável.
- Mais lento e menos preciso inicialmente.
- Atualizações rápidas e frequentes.

---

## Aprendizado Baseado em Modelo vs Aprendizado Baseado em Instância

| Aprendizado Baseado em Modelo              | Aprendizado Baseado em Instância                        |
|-------------------------------------------|----------------------------------------------------------|
| Cria modelos com base em exemplos          | Usa medidas de similaridade com outras instâncias       |
| Utiliza **função de utilidade** para medir o quão bom é o modelo | Aprende os exemplos e generaliza a partir deles         |
| Utiliza **função de custo** para medir erros | Não constrói um modelo explícito                        |

---

## Principais Desafios do Aprendizado de Máquina

- **Overfitting (Sobreajuste)**  
  - Modelo se ajusta demais aos dados de treino  
  - Incapaz de generalizar para novos dados  
  - Possível solução: *downsampling*, regularização, mais dados diversos

- **Underfitting (Subajuste)**  
  - Modelo muito simples ou com poucos dados  
  - Incapaz de aprender padrões nem nos dados de treino  
  - Possível solução: *oversampling*, modelo mais complexo, mais dados

- **Dados de baixa qualidade**  
  - Dados com ruídos, inconsistentes ou incompletos

- **Características irrelevantes**  
  - Variáveis que não contribuem para o aprendizado podem atrapalhar

---

## Ajuste de Hiperparâmetros

- **Hiperparâmetros** são parâmetros definidos antes do treinamento (ex: número de vizinhos no KNN, taxa de aprendizado em redes neurais).
- O ajuste correto é essencial para alcançar boa performance.

### Técnicas de Validação

- **Validação Holdout**  
  - Divisão simples dos dados, geralmente 80% para treino e 20% para validação.

- **Validação Cruzada (Cross-Validation)**  
  - Ex: **k-fold**: os dados são divididos em *k* partes e o treinamento é repetido *k* vezes alternando os conjuntos de treino e validação.

---

## Considerações Finais

- **Não existe bala de prata**:  
  Nenhum modelo ou técnica é universalmente superior para todos os problemas.

- A escolha do algoritmo, dos dados e dos parâmetros deve ser feita com base em testes, validações e conhecimento do domínio.

---
