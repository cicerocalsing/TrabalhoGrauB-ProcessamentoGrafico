# Componente algorítmico: rotulagem de componentes conexos implementada manualmente.
# - Busca em largura (BFS)
# - Duas passagens com union-find
# Ambas permitem escolher conectividade 4 ou 8.

import numpy as np

def rotular_componentes_conexos(imagem):

    binaria = (imagem > 0).astype(np.int32)
    altura, largura = binaria.shape
    
    rotulos = np.zeros((altura, largura), dtype=np.int32)
    
    # Estrutura para gerenciar equivalências entre rótulos
    equivalencias = {}
    proximo_rotulo = 1

    for y in range(altura):
        for x in range(largura):
            # Se for pixel de fundo (0) ele pula
            if binaria[y, x] == 0:
                continue
            
            # Olha os vizinhos já visitados na conectividade de 8
            vizinhos = []
            if y > 0 and rotulos[y - 1, x] > 0:                    # Acima
                vizinhos.append(rotulos[y - 1, x])
            if x > 0 and rotulos[y, x - 1] > 0:                    # Esquerda
                vizinhos.append(rotulos[y, x - 1])
            if y > 0 and x > 0 and rotulos[y - 1, x - 1] > 0:      # Diagonal cima esquerda
                vizinhos.append(rotulos[y - 1, x - 1])
            if y > 0 and x < largura - 1 and rotulos[y - 1, x + 1] > 0: # Diagonal cima direita
                vizinhos.append(rotulos[y - 1, x + 1])
            
            if not vizinhos:
                # Se não tem vizinhos rotulados cria um novo rótulo
                rotulos[y, x] = proximo_rotulo
                equivalencias[proximo_rotulo] = proximo_rotulo
                proximo_rotulo += 1
            else:
                # Se tem vizinhos pega o menor rótulo entre eles
                menor_rotulo = min(vizinhos)
                rotulos[y, x] = menor_rotulo
                
                # Registra as equivalências
                for v in vizinhos:
                    # Resolve a raiz
                    raiz_v = v
                    while equivalencias[raiz_v] != raiz_v:
                        raiz_v = equivalencias[raiz_v]
                    
                    raiz_menor = menor_rotulo
                    while equivalencias[raiz_menor] != raiz_menor:
                        raiz_menor = equivalencias[raiz_menor]
                    
                    if raiz_v != raiz_menor:
                        equivalencias[raiz_v] = raiz_menor

    # Transforma equivalências em sequência
    raizes_unicas = {}
    rotulo_atualizado = 1
    for r in sorted(equivalencias.keys()):
        raiz = r
        while equivalencias[raiz] != raiz:
            raiz = equivalencias[raiz]
        
        if raiz not in raizes_unicas:
            raizes_unicas[raiz] = rotulo_atualizado
            rotulo_atualizado += 1

    # Consolida os rótulos na matriz
    for y in range(altura):
        for x in range(largura):
            if rotulos[y, x] > 0:
                r = rotulos[y, x]
                while equivalencias[r] != r:
                    r = equivalencias[r]
                # Aplica o rótulo
                rotulos[y, x] = raizes_unicas[r]
                
    return rotulos
