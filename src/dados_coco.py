import json

from config import DIVISOES_VALIDAS, ID_PEANUTS, PASTA_DADOS

def validar_divisao(divisao):
    if divisao not in DIVISOES_VALIDAS:
        raise ValueError(f"Divisao Inválida: {divisao}. Use uma destas: {DIVISOES_VALIDAS}")

def caminho_anotacoes(divisao):
    validar_divisao(divisao)
    return PASTA_DADOS / "annotations" / f"{divisao}_photos.json"

def caminho_foto(divisao, nome_arquivo):
    validar_divisao(divisao)
    return PASTA_DADOS / "photos" / divisao / nome_arquivo

def carregar_anotacoes(divisao):
    caminho = caminho_anotacoes(divisao)

    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de anotações não encontrado: {caminho}")

    with open(caminho, encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    fotos_por_id = {}
    for foto in dados["images"]:
        fotos_por_id[foto["id"]] = foto

    anotacoes = {}

    for anotacao in dados["annotations"]:
        if anotacao["category_id"] != ID_PEANUTS:
            continue
        foto = fotos_por_id[anotacao["image_id"]]
        nome = foto["file_name"]

        if nome not in anotacoes:
            anotacoes[nome] = {
                "largura": foto["width"],
                "altura": foto["height"],
                "caixas": []
            }

        x, y, largura, altura = anotacao["bbox"]
        caixa = (round(x), round(y), round(largura), round(altura))
        anotacoes[nome]["caixas"].append(caixa)

    return dict(sorted(anotacoes.items()))