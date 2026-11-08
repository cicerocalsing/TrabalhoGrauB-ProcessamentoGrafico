# Separação objeto/fundo: redimensionamento, suavização, limiarização
# (Otsu ou HSV) e morfologia (abertura/fechamento). Saída: imagem binária 0/255.
# Como o background é bem parecido com a cor do objeto vou utilizar HSV

import cv2
import numpy as np

def segmentar_amendoins(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # suaviza imagem, melhora um pouquinho a limiarização
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # limiarização adaptativa
    filtro = cv2.adaptiveThreshold(
        blur,
        255, # valor do objeto
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # média local
        cv2.THRESH_BINARY_INV, # tipo de limiar. Se não tivesse INV ia ter fundo branco e obj preto
        31, # tamanho do bloco de comparação. Se for pequeno pega mais ruído. Deve ser ímpar
        4 # constante de rigorosidade. Quanto maior, menos ruídos, mas pode perder detalhes
    )


    # operações morfológicas
    kernel = np.ones((3,3), np.uint8)
    # remove ruídos pequenos envolta mas deteriora o amendoím
    mascara = cv2.morphologyEx(filtro, cv2.MORPH_OPEN, kernel, iterations=1)

    # reconstroi parte do amendoim deteriorado
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
    # --------- TESTAR COM ITERATIONS 1/3 e 1/1 ---------


    
    # mediana para melhorar ruídos
    mascara = cv2.medianBlur(mascara, 11)

    return mascara

#    mascara_descolada = cv2.erode(mascara, kernel, iterations=1)
#    kernel_grande = np.ones((7,7), np.uint8)
#    mascara_solida = cv2.morphologyEx(filtro, cv2.MORPH_CLOSE, kernel_grande, iterations=1)
    
