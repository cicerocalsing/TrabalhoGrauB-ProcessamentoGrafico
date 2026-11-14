# Separação objeto/fundo: redimensionamento, suavização, limiarização
# (Otsu ou HSV) e morfologia (abertura/fechamento). Saída: imagem binária 0/255.
# Como o background é bem parecido com a cor do objeto vou utilizar HSV

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

    b = cv2.GaussianBlur(lab[..., 2], (7, 7), 0)
    _, mascara = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    L = cv2.GaussianBlur(lab[..., 0], (5, 5), 0).astype(np.float32)
    sombra = (cv2.GaussianBlur(L, (0, 0), 12) - L) > 10
    mascara[sombra] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations = 2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations = 1)

    return mascara

def segmentar_amendoins_test(imagem):
    return imagem
