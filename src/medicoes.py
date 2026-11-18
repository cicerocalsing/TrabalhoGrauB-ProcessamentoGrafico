# Medições dos componentes conexos encontrados pela rotulagem
# Entrada: matriz de rótulos (0 = fundo, 1..quantidade = objetos)
# Saída: uma lista com as medidas de cada amendoim

import numpy as np

def calcular_areas(rotulos, quantidade):
    # Conta quantos pixels existem com cada rótulo (é um histograma dos rótulos)
    # areas[r] = área do objeto r; areas[0] = área do fundo
    return np.bincount(rotulos.ravel(), minlength=quantidade+1)

def medir_componente(rotulos, rotulo, area):
    # Coordenadas (linha, coluna) de todos os pixels que têm este rótulo
    ys, xs = np.nonzero(rotulos == rotulo)

    # Caixa delimitadora no mesmo formato do COCO: (x, y, largura, altura)
    # O +1 existe porque um objeto que ocupa só a coluna 5 tem largura 1, e não 0
    x = int(xs.min())
    y = int(ys.min())
    largura = int(xs.max()) - x + 1
    altura = int(ys.max()) - y + 1

    # Razão de aspecto: lado maior / lado menor (1 = quadrado, cresce quanto mais alongado)
    aspecto = max(largura, altura) / min(largura, altura)

    # Centro de massa: média das coordenadas dos pixels do objeto
    centro = (float(xs.mean()), float(ys.mean()))

    # Preenchimento: fração da caixa ocupada pelo objeto (1 = objeto ocupa a caixa toda)
    # Amendoins encostados costumam ter preenchimento baixo, porque a caixa fica cheia de fundo
    preenchimento = float(area / (largura * altura))

    return {
        "rotulo" : rotulo,
        "area" : int(area),
        "caixa" : (x, y, largura, altura),
        "aspecto" : aspecto,
        "centro" : centro,
        "preenchimento" : preenchimento
    }

def medir_componentes(rotulos, quantidade, area_minima=0, area_maxima=None):
    # Mede todos os objetos, descartando os que estão fora da faixa de área
    # area_minima: remove ruído (pontinhos que sobraram da segmentação)
    # area_maxima: remove regiões grandes demais (ex.: sombra ou borda do prato); None = sem limit
    areas = calcular_areas(rotulos, quantidade)

    medidas = []
    for rotulo in range(1, quantidade + 1):
        area = areas[rotulo]

        # A área é calculada antes para não medir componentes que serão descartados
        if area < area_minima:
            continue
        if area_maxima is not None and area > area_maxima:
            continue

        medidas.append(medir_componente(rotulos, rotulo, area))

    return medidas

def reescalar_medidas(medidas, fator):
    # Converte as medidas da imagem redimensionada para a escala da foto original
    # fator = tamanho original / tamanho processado (ex.: 3072 / 1000 = 3.072)
    # Medidas de comprimento são multiplicadas pelo fator e a área pelo fator ao quadrado
    # Aspecto e preenchimento são proporções, então não mudam
    reescaladas = []
    for medida in medidas:
        x, y, largura, altura = medida["caixa"]
        centro_x, centro_y = medida["centro"]

        reescaladas.append({
            "rotulo" : medida["rotulo"],
            "area": round(medida["area"] * fator * fator),
            "caixa": (round(x * fator), round(y * fator),
                      round(largura * fator), round(altura * fator)),
            "aspecto": medida["aspecto"],
            "centro": (centro_x * fator, centro_y * fator),
            "preenchimento": medida["preenchimento"],
        })

    return reescaladas

def extrair_caixas(medidas):
    # Lista só com as caixas, no mesmo formato usado por desenhar_caixas e pela avaliação
    return [medida["caixa"] for medida in medidas]

def resumir_medidas(medidas):
    # Estatísticas de uma foto: usadas no CSV de resumo e no relatório
    if not medidas:
        return {"quantidade": 0, "area_media": 0.0, "area_minima": 0, "area_maxima": 0,
                "aspecto_medio": 0.0}

    areas = np.array([medida["area"] for medida in medidas])
    aspectos = np.array([medida["aspecto"] for medida in medidas])

    return {
        "quantidade": len(medidas),
        "area_media": float(areas.mean()),
        "area_minima": int(areas.min()),
        "area_maxima": int(areas.max()),
        "aspecto_medio": float(aspectos.mean()),
    }

        

