# Separação objeto/fundo: redimensionamento, suavização, limiarização
# (Otsu ou HSV) e morfologia (abertura/fechamento). Saída: imagem binária 0/255.

import cv2
import numpy as np

"""def segmentar_amendoins(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # ou o gaussian ou o median, ve qual fica melhor
    #blur = cv2.GaussianBlur(gray, (11,11), 0)
    blur = cv2.medianBlur(gray, 11)

    #filtro = cv2.adaptiveThreshold(
    #    blur,
    #    255, # valor do objeto
    #    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # média local
    #    cv2.THRESH_BINARY_INV, # tipo de limiar. Se não tivesse INV ia ter fundo branco e obj preto
    #    31, # tamanho do bloco de comparação. Se for pequeno pega mais ruído. Deve ser ímpar
    #    4 # constante de rigorosidade. Quanto maior, menos ruídos, mas pode perder detalhes
    #)
    filtro = cv2.Canny(blur, 8, 55)


    kernel = np.ones((2,2), np.uint8)

    mascara = cv2.morphologyEx(filtro, cv2.MORPH_CLOSE, kernel, iterations=1)
    #mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
    #mascara = cv2.morphologyEx(mascara, cv2.dilate, kernel, iterations=1)
    #mascara = cv2.morphologyEx(mascara, cv2.MORPH_ELLIPSE, kernel, iterations=1)

    #contornos, _ = cv2.findContours(
    #    mascara,
    #    cv2.RETR_EXTERNAL,
    #    cv2.CHAIN_APPROX_SIMPLE,
    #)

    return mascara
"""
#   mascara_descolada = cv2.erode(mascara, kernel, iterations=1)
    
#   tentativa de open e close
#   MORPH_ELYPSE


def segmentar_amendoins_val(imagem):
    lab = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2LAB)

    blur = cv2.GaussianBlur(lab[..., 2], (7, 7), 0)

    gray = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2GRAY) > 15

    limiar, _ = cv2.threshold(blur[gray].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara = ((blur > limiar) & gray).astype(np.uint8) * 255
    #_, mascara = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # O OTSU estava sendo calculado junto na imagem ao redor. Então ele separava
    # o fundo do preto nos cantos ao invés do fundo e amendoins
    # Essas duas linhas acima foram uma sugestão de IA para resolver isso

    L = cv2.GaussianBlur(lab[..., 0], (5, 5), 0).astype(np.float32)
    sombra = (cv2.GaussianBlur(L, (0, 0), 12) - L) > 10
    mascara[sombra] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations = 2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations = 1)

    return mascara



"""def segmentar_amendoins_test(imagem):
    
    suave = cv2.GaussianBlur(imagem, (9, 9), 0)
    hsv = cv2.cvtColor(suave, cv2.COLOR_BGR2HSV)
    _, sat, _ = cv2.split(hsv)

    _, binaria = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binaria) > 127:
        binaria = cv2.bitwise_not(binaria)

    # 3. Limpeza morfológica básica (abertura tira ruído, fechamento preenche o grão)
    k_pequeno = np.ones((5, 5), np.uint8)
    k_medio = np.ones((9, 9), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k_pequeno)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, k_medio)

    # 4. Separar grãos encostados com Distance Transform
    dist = cv2.distanceTransform(binaria, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, miolos = cv2.threshold(dist_norm, 0.45, 255, cv2.THRESH_BINARY)
    miolos = np.uint8(miolos)

    # 5. Monta máscara final preenchida (0/255) filtrando farelos muito pequenos
    mascara_preenchida = np.zeros_like(binaria)
    contornos, _ = cv2.findContours(miolos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        if cv2.contourArea(cnt) > 120:
            cv2.drawContours(mascara_preenchida, [cnt], -1, 255, thickness=-1)

    # Leve dilatação para recuperar parte da borda que o distanceTransform encolheu
    mascara_preenchida = cv2.dilate(mascara_preenchida, np.ones((5, 5), np.uint8), iterations=1)
    return mascara_preenchida"""

def segmentar_amendoins_test(imagem):
    gray = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2GRAY)

    # ou o gaussian ou o median, ve qual fica melhor
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    #blur = cv2.medianBlur(gray, 7)

    filtro = cv2.adaptiveThreshold(
        blur,
        255, # valor do objeto
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # média local
        cv2.THRESH_BINARY_INV, # tipo de limiar. Se não tivesse INV ia ter fundo branco e obj preto
        31, # tamanho do bloco de comparação. Se for pequeno pega mais ruído. Deve ser ímpar
        4 # constante de rigorosidade. Quanto maior, menos ruídos, mas pode perder detalhes
    )
    #filtro = cv2.Canny(blur, 8, 55)
    
    mascara = cv2.medianBlur(filtro, 13)


    kernel = np.ones((2,2), np.uint8)

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=3)
    #mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)
    #mascara = cv2.morphologyEx(mascara, cv2.dilate, kernel, iterations=1)
    #mascara = cv2.morphologyEx(mascara, cv2.MORPH_ELLIPSE, kernel, iterations=1)

    #contornos, _ = cv2.findContours(
    #    mascara,
    #    cv2.RETR_EXTERNAL,
    #    cv2.CHAIN_APPROX_SIMPLE,
    #)
    
    return mascara
