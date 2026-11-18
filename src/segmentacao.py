# Separação objeto/fundo: redimensionamento, suavização, limiarização
# (Otsu ou HSV) e morfologia (abertura/fechamento). Saída: imagem binária 0/255.

import cv2
import numpy as np

# Parâmetros da detecção do tipo de fundo (definidos olhando só as fotos de val)
FRACAO_CANTO = 1 / 20   # lado do quadrado de cada canto, em fração do maior lado
LIMIAR_PRETO = 15       # brilho abaixo disso conta como preto (0 a 255)


def tem_fundo_escuro(imagem):

    # Fotos de mesa escura são recortadas em círculo: os 4 cantos ficam pretos
    cinza = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2GRAY)
    k = int(max(cinza.shape) * FRACAO_CANTO)
    cantos = np.concatenate([
        cinza[:k, :k].ravel(), cinza[:k, -k:].ravel(),
        cinza[-k:, :k].ravel(), cinza[-k:, -k:].ravel(),
    ])
    return (cantos < LIMIAR_PRETO).mean() > 0.5


def segmentar_fundo_escuro(imagem):

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


def segmentar_fundo_madeira(imagem):
 
    el = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))

    img = cv2.resize(imagem[..., :3], (1000, int(1000* imagem.shape[0] / imagem.shape[1])), interpolation=cv2.INTER_AREA)

    L = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0], 11).astype(np.float32)

    grad = cv2.magnitude(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0 , 1))
    borda = cv2.morphologyEx((grad > 20).astype(np.uint8) * 255, cv2.MORPH_CLOSE, el(15))
    fundo = cv2.bitwise_not(borda)
    cv2.floodFill(fundo, None, (0, 0), 0)

    mascara = cv2.morphologyEx(borda | fundo, cv2.MORPH_OPEN, el(25))

    mascara = cv2.resize(mascara, (imagem.shape[1], imagem.shape[0]), interpolation = cv2.INTER_NEAREST)
    return mascara


def segmentar_amendoins(imagem):

    if tem_fundo_escuro(imagem):
        return segmentar_fundo_escuro(imagem)
    return segmentar_fundo_madeira(imagem)



"""def segmentar_fundo_escuro(imagem):
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

    return mascara"""