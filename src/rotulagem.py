# Componente algorítmico: rotulagem de componentes conexos implementada manualmente
# Algoritmo de duas passagens com tabela de equivalências (union-find)
# Conectividade 4 e conectividade 8 ficam em funções separadas para poderem ser comparadas

import numpy as np

# ---------------------------------------------------------------------------
# Funções compartilhadas pelas duas conectividades
# ---------------------------------------------------------------------------

def encontrar_raiz(equivalencias, rotulo):
    # Sobe pela cadeia de equivalências até achar a raiz (o rótulo que aponta para si mesmo)
    # Ex: equivalencias = [0, 1, 1, 2] -> raiz de 3: 3 -> 2 -> 1 -> 1 (raiz = 1)

    raiz = rotulo
    while equivalencias[raiz] != raiz:
        raiz = equivalencias[raiz]

    # Compressão de caminho: faz cada rótulo do caminho apontar direto para a raiz
    # Assim a próxima busca pelo mesmo rótulo termina em um único passo
    while equivalencias[rotulo] != raiz:
        proximo = equivalencias[rotulo]
        equivalencias[rotulo] = raiz
        rotulo = proximo
    
    return raiz

def unir(equivalencias, rotulo_a, rotulo_b):
    # Registra que dois rótulos pertencem ao mesmo objeto
    raiz_a = encontrar_raiz(equivalencias, rotulo_a)
    raiz_b = encontrar_raiz(equivalencias, rotulo_b)

    if raiz_a == raiz_b:
        return

    # A raiz menor vira a raiz do grupo; assim a raiz é sempre o menor rótulo do objeto
    if raiz_a < raiz_b:
        equivalencias[raiz_b] = raiz_a
    else:
        equivalencias[raiz_a] = raiz_b

def rotular_pixel(rotulos, equivalencias, y, x, vizinhos):
    # Decide o rótulo do pixel (y, x) a partir dos rótulos dos vizinhos já visitados
    if not vizinhos:
        # Nenhum vizinho rotulado: começo de um objeto novo
        # O novo rótulo é o próximo índice livre da lista e começa sendo raiz de si mesmo
        novo_rotulo = len(equivalencias)
        equivalencias.append(novo_rotulo)
        rotulos[y, x] = novo_rotulo
        return

    menor_rotulo = min(vizinhos)
    rotulos[y, x] = menor_rotulo

    # Se os vizinhos têm rótulos diferentes, duas partes do mesmo objeto se encontraram
    for rotulo_vizinho in vizinhos:
        if rotulo_vizinho != menor_rotulo:
            unir(equivalencias, menor_rotulo, rotulo_vizinho)

def consolidar_rotulos(rotulos, equivalencias):
    # Segunda passagem: troca cada rótulo provisório pelo número final do seu objeto
    tabela = np.zeros(len(equivalencias), dtype=np.int32)
    quantidade = 0

    # Percorre os rótulos em ordem crescente (o índice 0 é o fundo e fica com 0)
    for rotulo in range(1, len(equivalencias)):
        raiz = encontrar_raiz(equivalencias, rotulo)

        if raiz == rotulo:
            # É raiz: representa um objeto novo, que recebe o próximo número sequencial
            quantidade += 1
            tabela[rotulo] = quantidade

        else:
            # Não é raiz: herda o número da sua raiz
            # A raiz é sempre menor que o rótulo, então seu número já foi definido
            tabela[rotulo] = tabela[raiz]
    # Aplica a tabela na matriz inteira de uma vez (indexação do NumPy)
    #   Cada pixel com valor r passa a valer tabela[r]; o fundo (0) continua 0
    return quantidade, tabela[rotulos]

def preparar_rotulagem(mascara):
    # Estruturas usadas pelas duas conectividades
    binaria = mascara > 0     # True = objeto, False = fundo
    rotulos = np.zeros(mascara.shape, dtype=np.int32)      # 0 = fundo / ainda sem rótulo
    equivalencias = [0]     # índice 0 reservado para o fundo
    return binaria, rotulos, equivalencias

# ---------------------------------------------------------------------------
# Conectividade 4: vizinhos apenas nos lados (cima, baixo, esquerda, direita)
# ---------------------------------------------------------------------------

def rotular_conectividade_4(mascara):
    binaria, rotulos, equivalencias = preparar_rotulagem(mascara)
    altura, largura = binaria.shape

    # Primeira passagem: varre a imagem de cima para baixo, da esquerda para a direita
    for y in range(altura):
        for x in range(largura):
            if not binaria[y,x]:
                continue

            # Na varredura, só os vizinhos de cima e da esquerda já foram visitados
            #     ↑
            #  ←  P
            vizinhos = []
            if y > 0 and rotulos[y - 1, x] > 0: #Acima
                vizinhos.append(rotulos[y - 1, x])
            if x > 0 and rotulos[y, x - 1] > 0: #Esquerda
                vizinhos.append(rotulos[y, x - 1])

            rotular_pixel(rotulos, equivalencias, y, x, vizinhos)

    return consolidar_rotulos(rotulos, equivalencias)

# ---------------------------------------------------------------------------
# Conectividade 8: vizinhos nos lados e nas diagonais
# ---------------------------------------------------------------------------

def rotular_conectividade_8(mascara):
    binaria, rotulos, equivalencias = preparar_rotulagem(mascara)
    altura, largura = binaria.shape

    for y in range(altura):
        for x in range(largura):
            if not binaria[y, x]:
                continue

            # Além de cima e esquerda, as duas diagonais de cima já foram visitadas
            #  ↖  ↑  ↗
            #  ←  P
            vizinhos = []
            if y > 0 and x > 0 and rotulos[y - 1, x - 1] > 0: # Diagonal cima-esquerda
                vizinhos.append(rotulos[y - 1, x - 1])
            if y > 0 and rotulos[y - 1, x] > 0: #Acima
                vizinhos.append(rotulos[y - 1, x])
            if y > 0 and x < largura - 1 and rotulos[y - 1, x + 1] >0: #Diagonal cima-direita
                vizinhos.append(rotulos[y - 1, x + 1])
            if x > 0 and rotulos[y, x - 1] > 0: #Esquerda
                vizinhos.append(rotulos[y, x - 1])

            rotular_pixel(rotulos, equivalencias, y, x, vizinhos)
    return consolidar_rotulos(rotulos, equivalencias)

# ---------------------------------------------------------------------------
# Escolha da conectividade
# ---------------------------------------------------------------------------

def rotular_componentes_conexos(mascara, conectividade=8):
    # Ponto de entrada usado pelo pipeline e pelo experimento
    # Retorna (quantidade de objetos, matriz de rótulos 0..quantidade)
    if conectividade == 4:
        return rotular_conectividade_4(mascara)
    if conectividade == 8:
        return rotular_conectividade_8(mascara)
    raise ValueError(f"Conectividade inválida: {conectividade}. Use 4 ou 8.")
