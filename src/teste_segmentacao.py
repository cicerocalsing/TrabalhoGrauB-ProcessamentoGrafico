import cv2
from dados_coco import caminho_foto, carregar_anotacoes
from segmentacao import segmentar_amendoins

divisao = "val"
anotacoes = carregar_anotacoes(divisao)
primeiro_nome = list(anotacoes.keys())[1]

caminho = caminho_foto(divisao, primeiro_nome)
imagem = cv2.imread(str(caminho))

if imagem is None:
    print(f"Não achei a imagem em: {caminho}")
else:
    print(f"Testando a imagem: {primeiro_nome}")

    mascara = segmentar_amendoins(imagem)
    
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.imshow("Original", mascara)

# Define a posição da janela (x, y)
    cv2.moveWindow("Original", 100, 100)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
