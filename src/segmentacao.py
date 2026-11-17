# Separação objeto/fundo
# Cada foto é classificada pelo tipo de fundo e segmentada pelo método adequado:
#   - fundo escuro (prato): limiarização de Otsu no canal de cor b do LAB + morfologia
#   - fundo de madeira: detecção de bordas (Sobel) + preenchimento + morfologia
# Saída: máscara binária do tamanho da foto, com 255 = amendoim e 0 = fundo
# Para a investigação experimental, a segmentação pode ser feita em 4 configurações (C0 a C3),
# cada uma acrescentando uma etapa à anterior (ver CONFIGURACOES em config.py)
# Todos os parâmetros numéricos ficam em config.py

import cv2
import numpy as np

from config import (AREA_MINIMA, CONFIGURACAO_PADRAO, CONFIGURACOES, FATOR_AREA_GRANDE,
                    FRACAO_CANTO, FRACAO_CANTOS_PRETOS, FRACAO_NUCLEO, ITERACOES_ABERTURA,
                    ITERACOES_FECHAMENTO, KERNEL_ABERTURA_MADEIRA, KERNEL_FECHAMENTO_BORDA,
                    KERNEL_MEDIANA_MADEIRA, KERNEL_MORFOLOGIA_ESCURO, KERNEL_SUAVIZACAO_BRILHO,
                    KERNEL_SUAVIZACAO_COR, LARGURA_PROCESSAMENTO_MADEIRA, LIMIAR_GRADIENTE,
                    LIMIAR_PRETO, LIMIAR_SOMBRA, SIGMA_VIZINHANCA_SOMBRA)
from medicoes import calcular_areas
from rotulagem import rotular_componentes_conexos


def elemento_eliptico(tamanho):

    # Elemento estruturante em forma de elipse (tamanho x tamanho), usado na morfologia
    # A elipse arredonda os contornos, o que combina com o formato dos grãos
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho, tamanho))


def tem_fundo_escuro(imagem):

    # Fotos de prato escuro são recortadas em círculo: os 4 cantos ficam pretos
    cinza = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2GRAY)
    lado = int(max(cinza.shape) * FRACAO_CANTO)

    # Junta os pixels dos quatro cantos em uma lista só
    cantos = np.concatenate([
        cinza[:lado, :lado].ravel(), cinza[:lado, -lado:].ravel(),
        cinza[-lado:, :lado].ravel(), cinza[-lado:, -lado:].ravel(),
    ])

    # (cantos < LIMIAR_PRETO) é True/False; a média dá a fração de pixels pretos
    return (cantos < LIMIAR_PRETO).mean() > FRACAO_CANTOS_PRETOS


def limiarizar_cor_fundo_escuro(imagem):

    lab = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2LAB)

    # Canal b suavizado: o amendoim é amarelado (b alto) e o prato é neutro (b baixo)
    canal_b = cv2.GaussianBlur(lab[..., 2], (KERNEL_SUAVIZACAO_COR, KERNEL_SUAVIZACAO_COR), 0)

    # Região de interesse: tudo o que não é o preto dos cantos
    # Se os cantos entrassem no Otsu, ele separaria "preto x prato" em vez de "prato x amendoim"
    regiao = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2GRAY) > LIMIAR_PRETO

    # Otsu calculado só com os pixels da região de interesse
    limiar, _ = cv2.threshold(canal_b[regiao].reshape(-1, 1), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((canal_b > limiar) & regiao).astype(np.uint8) * 255


def limiarizar_brilho_fundo_madeira(imagem):

    # Na madeira a cor do amendoim é parecida com a do fundo, mas ele é um pouco mais escuro
    # A mediana apaga os veios da madeira; o Otsu INV marca como objeto o grupo mais escuro
    brilho = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2LAB)[..., 0]
    brilho = cv2.medianBlur(brilho, KERNEL_MEDIANA_MADEIRA)
    _, mascara = cv2.threshold(brilho, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mascara


def segmentar_fundo_escuro(imagem):

    # Começa pela limiarização por cor (que sozinha é a configuração C1 do fundo escuro)
    lab = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2LAB)
    mascara = limiarizar_cor_fundo_escuro(imagem)

    # Sombras entre grãos encostados: pixels bem mais escuros que a vizinhança são apagados
    # Isso abre uma "fenda" entre grãos colados, ajudando a rotulagem a separá-los
    brilho = cv2.GaussianBlur(lab[..., 0], (KERNEL_SUAVIZACAO_BRILHO, KERNEL_SUAVIZACAO_BRILHO), 0)
    brilho = brilho.astype(np.float32)
    brilho_vizinhanca = cv2.GaussianBlur(brilho, (0, 0), SIGMA_VIZINHANCA_SOMBRA)
    sombra = (brilho_vizinhanca - brilho) > LIMIAR_SOMBRA
    mascara[sombra] = 0

    # Abertura remove ruído; fechamento tapa buracos dentro dos grãos
    kernel = elemento_eliptico(KERNEL_MORFOLOGIA_ESCURO)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=ITERACOES_ABERTURA)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=ITERACOES_FECHAMENTO)

    return mascara


def segmentar_fundo_madeira(imagem):

    # Reduz a foto para acelerar os filtros grandes (a máscara volta ao tamanho original no fim)
    altura, largura = imagem.shape[:2]
    nova_altura = int(LARGURA_PROCESSAMENTO_MADEIRA * altura / largura)
    reduzida = cv2.resize(imagem[..., :3], (LARGURA_PROCESSAMENTO_MADEIRA, nova_altura),
                          interpolation=cv2.INTER_AREA)

    # Canal L (brilho) com mediana: apaga os veios finos da madeira e preserva as bordas dos grãos
    brilho = cv2.cvtColor(reduzida, cv2.COLOR_BGR2LAB)[..., 0]
    brilho = cv2.medianBlur(brilho, KERNEL_MEDIANA_MADEIRA).astype(np.float32)

    # Magnitude do gradiente (Sobel em x e em y): é alta onde o brilho muda, ou seja, nas bordas
    gradiente = cv2.magnitude(cv2.Sobel(brilho, cv2.CV_32F, 1, 0),
                              cv2.Sobel(brilho, cv2.CV_32F, 0, 1))
    borda = (gradiente > LIMIAR_GRADIENTE).astype(np.uint8) * 255

    # Fechamento liga os pedaços do contorno de cada grão, formando curvas fechadas
    borda = cv2.morphologyEx(borda, cv2.MORPH_CLOSE, elemento_eliptico(KERNEL_FECHAMENTO_BORDA))

    # Preenchimento: pinta de 0 tudo o que é alcançável a partir do canto (0, 0) sem cruzar bordas
    # O que sobra em 255 é o interior das curvas fechadas, ou seja, o interior dos grãos
    fundo = cv2.bitwise_not(borda)
    cv2.floodFill(fundo, None, (0, 0), 0)

    # Grão = contorno + interior; a abertura remove bordas soltas que não formam um grão
    mascara = cv2.morphologyEx(borda | fundo, cv2.MORPH_OPEN,
                               elemento_eliptico(KERNEL_ABERTURA_MADEIRA))

    # Volta ao tamanho original (INTER_NEAREST mantém a máscara só com 0 e 255)
    mascara = cv2.resize(mascara, (largura, altura), interpolation=cv2.INTER_NEAREST)
    return mascara


def limiar_simples(imagem):

    # C0: o mínimo possível - Otsu global no cinza, sem suavização, sem região de interesse
    cinza = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2GRAY)
    _, mascara = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # O Otsu só separa "claro x escuro"; não sabe qual dos dois é o amendoim
    # Os grãos ocupam pouco da foto, então o objeto é o grupo com MENOS pixels
    if (mascara > 0).mean() > 0.5:
        mascara = cv2.bitwise_not(mascara)
    return mascara


def segmentar_por_cor(imagem):

    # C1: limiarização no canal que melhor diferencia o amendoim de cada fundo, sem morfologia
    if tem_fundo_escuro(imagem):
        return limiarizar_cor_fundo_escuro(imagem)
    return limiarizar_brilho_fundo_madeira(imagem)


def segmentar_completo(imagem):

    # C2: método completo de cada tipo de fundo (pipeline principal)
    if tem_fundo_escuro(imagem):
        return segmentar_fundo_escuro(imagem)
    return segmentar_fundo_madeira(imagem)


def separar_encostados(mascara):

    # C3: separa grãos colados que formaram um único componente
    # Ideia: no meio de cada grão a distância até o fundo é grande; no "pescoço" entre
    # dois grãos encostados ela é pequena. Os centros dos grãos viram sementes (marcadores)
    # e o watershed divide o componente na linha entre as sementes

    # 1) Rótulos e áreas de todos os componentes (rotulagem implementada pelo grupo)
    quantidade, rotulos = rotular_componentes_conexos(mascara, 8)
    areas = calcular_areas(rotulos, quantidade)

    # 2) Só os componentes grandes demais para serem um grão só são candidatos à separação
    # A referência é a mediana das áreas da própria foto (ignorando o ruído abaixo de AREA_MINIMA)
    areas_validas = areas[1:][areas[1:] >= AREA_MINIMA]
    if len(areas_validas) == 0:
        return mascara
    limite = FATOR_AREA_GRANDE * np.median(areas_validas)

    grande = areas > limite         # grande[r] = True se o componente r é candidato
    grande[0] = False               # o fundo nunca é candidato
    candidatos = grande[rotulos]    # mesma ideia da LUT: transforma rótulo em True/False por pixel
    if not candidatos.any():
        return mascara

    # 3) Transformada de distância: cada pixel recebe a distância até o pixel de fundo mais próximo
    distancia = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)

    # 4) Maior distância de cada componente (o "raio" do componente)
    # np.maximum.at percorre todos os pixels e guarda, para cada rótulo, o maior valor encontrado
    maior_distancia = np.zeros(quantidade + 1, dtype=np.float32)
    np.maximum.at(maior_distancia, rotulos.ravel(), distancia.ravel())

    # 5) Núcleos: a parte mais "funda" de cada componente candidato
    nucleos = candidatos & (distancia > FRACAO_NUCLEO * maior_distancia[rotulos])
    nucleos = nucleos.astype(np.uint8) * 255

    # 6) Marcadores do watershed: 1 = fundo, 2, 3, ... = cada núcleo, 0 = região a decidir
    _, rotulos_nucleos = rotular_componentes_conexos(nucleos, 8)
    marcadores = (rotulos_nucleos + 1).astype(np.int32)
    marcadores[candidatos & (nucleos == 0)] = 0
    marcadores[~candidatos] = 1

    # 7) Watershed: "inunda" o relevo a partir dos marcadores; onde duas inundações se encontram
    # ele marca -1 (linha divisória). O relevo usado é a distância invertida: os centros dos grãos
    # são vales e o pescoço entre dois grãos é uma crista
    relevo = cv2.normalize(distancia, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    relevo = cv2.cvtColor(255 - relevo, cv2.COLOR_GRAY2BGR)     # o watershed exige imagem colorida
    cv2.watershed(relevo, marcadores)

    # 8) Apaga as linhas divisórias dentro dos candidatos (engrossadas para 3 px, assim nem a
    # conectividade 8 atravessa a linha pela diagonal)
    linhas = (marcadores == -1).astype(np.uint8)
    linhas = cv2.dilate(linhas, np.ones((3, 3), np.uint8)) > 0
    separada = mascara.copy()
    separada[candidatos & linhas] = 0
    return separada


def segmentar_amendoins(imagem, configuracao=CONFIGURACAO_PADRAO):

    # Escolhe a configuração da investigação experimental (ver CONFIGURACOES em config.py)
    if configuracao == "C0":
        return limiar_simples(imagem)
    if configuracao == "C1":
        return segmentar_por_cor(imagem)
    if configuracao == "C2":
        return segmentar_completo(imagem)
    if configuracao == "C3":
        return separar_encostados(segmentar_completo(imagem))
    raise ValueError(f"Configuração inválida: {configuracao}. Use uma de {list(CONFIGURACOES)}.")