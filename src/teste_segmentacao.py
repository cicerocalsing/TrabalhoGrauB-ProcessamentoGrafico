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

    cv2.imshow("Imagem Original", imagem)
    cv2.imshow("Mascara Segmentada (HSV + Otsu)", mascara)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
