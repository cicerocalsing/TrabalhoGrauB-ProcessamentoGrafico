import cv2

def espessura_proporcional(imagem):
    altura, largura = imagem.shape[:2]
    return max(2, max(altura, largura) //500)

def desenhar_caixas(imagem, caixas, cor, numerar=True):
    resultado = imagem.copy()

    espessura = espessura_proporcional(imagem)
    escala_fonte = espessura * 0.4

    for numero, (x, y, largura, altura) in enumerate(caixas, start=1):
        cv2.rectangle(
            resultado,
            (x,y),                              # Canto superior esquerdo   
            (x + largura, y + altura),          # Canto inferior direito
            cor,                                # Cor da caixa em BGR
            espessura                           # Espessura da linha em pixels
        )

        if numerar:
            posicao_texto = (x, max(y - espessura * 2, espessura * 10))
            cv2.putText(
                resultado, str(numero), posicao_texto,
                cv2.FONT_HERSHEY_SIMPLEX,
                escala_fonte, cor, espessura
            )

    return resultado