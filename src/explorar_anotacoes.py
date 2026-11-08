import cv2
import numpy as np

from config import COR_ANOTACAO, PASTA_RESULTADOS
from dados_coco import caminho_foto, carregar_anotacoes
from visualizacao import desenhar_caixas

DIVISAO = "val"

pasta_saida = PASTA_RESULTADOS / "exploracao"

pasta_saida.mkdir(parents=True, exist_ok=True)

anotacoes = carregar_anotacoes(DIVISAO)

print(f"Divisão: {DIVISAO}")
print(f"Fotos com amendoim: {len(anotacoes)}")
print()
print(f"{'Foto':<20} {'Largura':>7} {'Altura':>7} {'Caixas':>7}")

medidas = []

for nome, info in anotacoes.items():
    caminho = caminho_foto(DIVISAO, nome)
    imagem = cv2.imread(str(caminho))

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem {caminho}")

    altura, largura = imagem.shape[:2]
    if (largura, altura) != (info["largura"], info["altura"]):
        print(
            f"AVISO: {nome} tem {largura}x{altura}, "
            f"mas o JSON informa {info['largura']}x{info['altura']}"
        )

    caixas = info["caixas"]
    anotada = desenhar_caixas(imagem, caixas, COR_ANOTACAO)

    saida = pasta_saida / f"{caminho.stem}_anotada.jpg"
    if not cv2.imwrite(str(saida), anotada):
        raise OSError(f"falha ao salvar: {saida}")

    print(f"{nome:<20} {largura:>7} {altura:>7} {len(caixas):>7}")

    area_foto = largura * altura
    for (_, _, largura_caixa, altura_caixa) in caixas:
        area_caixa = largura_caixa * altura_caixa
        aspecto = max(largura_caixa, altura_caixa) / min(largura_caixa, altura_caixa)
        area_relativa = 100 * area_caixa / area_foto
        medidas.append([largura_caixa, altura_caixa, area_caixa, aspecto, area_relativa])

medidas = np.array(medidas, dtype=np.float64)

print()
print(f"Total de caixas: {len(medidas)}")
print()

minimos = medidas.min(axis=0)
medias = medidas.mean(axis=0)
maximos = medidas.max(axis=0)

nomes_medidas = ["Largura (px)", "Altura (px)", "Área (px²)", "Aspecto", "Área relativa (%)"]

print(f"{'Medida':<18} {'Mínimo':>12} {'Média':>12} {'Máximo':>12}")
for i, nome_medida in enumerate(nomes_medidas):
    print(f"{nome_medida:<18} {minimos[i]:>12.2f} {medias[i]:>12.2f} {maximos[i]:>12.2f}")

print()
print(f"Imagens salvas em: {pasta_saida}")