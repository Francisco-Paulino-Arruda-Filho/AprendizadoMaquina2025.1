

import random
import numpy as np

# Exemplo com 4 instalações
fluxo = np.array([
    [0, 90, 10, 23, 43, 0, 0, 0, 0, 0, 0, 0],
    [90, 0, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0],
    [10, 0, 0, 0, 0, 0, 26, 16, 0, 0, 0, 0],
    [23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 88, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 16, 0, 0, 0, 0, 0, 0, 96, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 29, 0],
    [0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0, 37],
    [0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0],
])

distancia = np.array([
   [0, 36, 54, 26, 59, 72, 9, 34, 79, 17, 46, 95],
    [36, 0, 73, 35, 90, 58, 30, 78, 35, 44, 79, 36],
    [54, 73, 0, 21, 10, 97, 58, 66, 69, 61, 54, 63],
    [26, 35, 21, 0, 93, 12, 46, 40, 37, 48, 68, 85],
    [59, 90, 10, 93, 0, 64, 5, 29, 76, 16, 5, 76],
    [72, 58, 97, 12, 64, 0, 96, 55, 38, 54, 0, 34],
    [9, 30, 58, 46, 5, 96, 0, 83, 35, 11, 56, 37],
    [34, 78, 66, 40, 29, 55, 83, 0, 44, 12, 15, 80],
    [79, 35, 69, 37, 76, 38, 35, 44, 0, 64, 39, 33],
    [17, 44, 61, 48, 16, 54, 11, 12, 64, 0, 70, 86],
    [46, 79, 54, 68, 5, 0, 56, 15, 39, 70, 0, 18],
    [95, 36, 63, 85, 76, 34, 37, 80, 33, 86, 18, 0]
])


def calcular_custo(permutacao, fluxo, distancia):
    n = len(permutacao)
    custo = 0
    for i in range(n):
        for j in range(n):
            custo += fluxo[permutacao[i]][permutacao[j]] * distancia[i][j]
    return custo

def vizinhanca_swap(permutacao):
    vizinhos = []
    n = len(permutacao)
    for i in range(n):
        for j in range(i + 1, n):
            nova = permutacao.copy()
            nova[i], nova[j] = nova[j], nova[i]
            vizinhos.append(nova)
    return vizinhos

def busca_local(fluxo, distancia, max_iter=1000):
    n = len(fluxo)
    atual = list(range(n))
    random.shuffle(atual)
    custo_atual = calcular_custo(atual, fluxo, distancia)

    for _ in range(max_iter):
        melhor_vizinho = atual
        melhor_custo = custo_atual
        for vizinho in vizinhanca_swap(atual):
            custo_vizinho = calcular_custo(vizinho, fluxo, distancia)
            if custo_vizinho < melhor_custo:
                melhor_custo = custo_vizinho
                melhor_vizinho = vizinho
        if melhor_custo < custo_atual:
            atual = melhor_vizinho
            custo_atual = melhor_custo
        else:
            break  # Nenhuma melhora -> ótimo local
    return atual, custo_atual

solucao, custo = busca_local(fluxo, distancia)
print(f"Melhor solução: {solucao}")
print(f"Custo: {custo}")
