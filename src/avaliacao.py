# Comparação com as anotações da base (gabarito)
# O gabarito é usado só como referência de avaliação, nunca como entrada da detecção
# Duas formas de avaliar:
#   - por contagem: compara só a quantidade de amendoins encontrados
#   - por localização (IoU): verifica se cada caixa encontrada coincide com uma caixa anotada


def avaliar_contagem(contados, gabarito):

    # Compara a quantidade encontrada pelo programa com a quantidade de caixas anotadas
    # erro < 0: contou a menos (ex.: grãos grudados) / erro > 0: contou a mais (ex.: ruído)
    erro = contados - gabarito
    return {
        "erro": erro,
        "erro_absoluto": abs(erro),
        # Sem caixas no gabarito não dá para calcular o erro relativo
        "erro_relativo": abs(erro) / gabarito if gabarito > 0 else None,
    }


def calcular_iou(caixa_a, caixa_b):

    # IoU (Intersection over Union): área da interseção dividida pela área da união das caixas
    # 1 = caixas idênticas / 0 = caixas sem nenhuma sobreposição
    # As caixas estão no formato do COCO: (x, y, largura, altura)
    xa, ya, la, aa = caixa_a
    xb, yb, lb, ab = caixa_b

    # Retângulo da interseção: em cada eixo, começa no maior dos inícios e termina no menor dos fins
    inicio_x = max(xa, xb)
    inicio_y = max(ya, yb)
    fim_x = min(xa + la, xb + lb)
    fim_y = min(ya + aa, yb + ab)

    # Se o fim ficar antes do início, as caixas não se tocam naquele eixo: a medida vira 0
    largura_intersecao = max(0, fim_x - inicio_x)
    altura_intersecao = max(0, fim_y - inicio_y)
    intersecao = largura_intersecao * altura_intersecao

    # União = soma das duas áreas menos a interseção (que seria contada duas vezes)
    uniao = la * aa + lb * ab - intersecao
    if uniao == 0:
        return 0.0
    return intersecao / uniao


def casar_caixas(caixas_detectadas, caixas_gabarito, limiar_iou):

    # Associa cada caixa detectada a no máximo uma caixa do gabarito, e vice-versa
    # Estratégia gulosa: os pares com maior IoU são casados primeiro

    # 1) Todos os pares possíveis com IoU suficiente: (iou, índice da detectada, índice da anotada)
    candidatos = []
    for i, detectada in enumerate(caixas_detectadas):
        for j, anotada in enumerate(caixas_gabarito):
            iou = calcular_iou(detectada, anotada)
            if iou >= limiar_iou:
                candidatos.append((iou, i, j))

    # 2) Do maior IoU para o menor (as tuplas são ordenadas pelo primeiro elemento)
    candidatos.sort(reverse=True)

    # 3) Aceita um par só se nenhuma das duas caixas já tiver sido usada
    detectadas_usadas = set()
    anotadas_usadas = set()
    pares = []
    for iou, i, j in candidatos:
        if i in detectadas_usadas or j in anotadas_usadas:
            continue
        detectadas_usadas.add(i)
        anotadas_usadas.add(j)
        pares.append((i, j, iou))

    return pares


def calcular_metricas(vp, fp, fn):

    # precisão: das caixas que o programa desenhou, quantas estão certas
    # revocação: dos amendoins anotados, quantos o programa encontrou
    # F1: média harmônica das duas (só é alta se as duas forem altas)
    # Quando o denominador é 0 (nada detectado ou nada anotado), a métrica fica 0
    precisao = vp / (vp + fp) if vp + fp > 0 else 0.0
    revocacao = vp / (vp + fn) if vp + fn > 0 else 0.0
    if precisao + revocacao > 0:
        f1 = 2 * precisao * revocacao / (precisao + revocacao)
    else:
        f1 = 0.0

    return {"precisao": precisao, "revocacao": revocacao, "f1": f1}


def avaliar_deteccao(caixas_detectadas, caixas_gabarito, limiar_iou):

    # Avaliação por localização de uma foto
    pares = casar_caixas(caixas_detectadas, caixas_gabarito, limiar_iou)

    # VP (verdadeiro positivo): caixa detectada que casou com uma anotada
    # FP (falso positivo): caixa detectada que não casou com nenhuma (ruído, grãos fundidos...)
    # FN (falso negativo): caixa anotada que ficou sem par (amendoim perdido)
    vp = len(pares)
    fp = len(caixas_detectadas) - vp
    fn = len(caixas_gabarito) - vp

    # IoU médio dos acertos: mostra o quanto as caixas certas se ajustam ao amendoim
    iou_medio = sum(iou for _, _, iou in pares) / vp if vp > 0 else 0.0

    metricas = calcular_metricas(vp, fp, fn)

    return {
        "vp": vp,
        "fp": fp,
        "fn": fn,
        "precisao": metricas["precisao"],
        "revocacao": metricas["revocacao"],
        "f1": metricas["f1"],
        "iou_medio": iou_medio,
        "pares": pares,
    }


def calcular_totais(linhas_resumo):

    # Métricas agregadas de uma divisão inteira, a partir das linhas do resumo por foto
    # Cada linha precisa ter: gabarito, contados, erro, erro_absoluto, vp, fp, fn e tempo_total_s
    n = len(linhas_resumo)
    tempo_total = sum(linha["tempo_total_s"] for linha in linhas_resumo)

    # VP, FP e FN somados em todas as fotos; as métricas são calculadas sobre as somas
    vp = sum(linha["vp"] for linha in linhas_resumo)
    fp = sum(linha["fp"] for linha in linhas_resumo)
    fn = sum(linha["fn"] for linha in linhas_resumo)
    metricas = calcular_metricas(vp, fp, fn)

    return {
        "fotos": n,
        "total_gabarito": sum(linha["gabarito"] for linha in linhas_resumo),
        "total_contados": sum(linha["contados"] for linha in linhas_resumo),
        # Erro médio com sinal: negativo indica tendência a contar a menos
        "erro_medio": sum(linha["erro"] for linha in linhas_resumo) / n,
        # Erro absoluto médio (MAE): quantos amendoins o programa erra, em média, por foto
        "erro_absoluto_medio": sum(linha["erro_absoluto"] for linha in linhas_resumo) / n,
        "contagens_exatas": sum(1 for linha in linhas_resumo if linha["erro"] == 0),
        "vp": vp,
        "fp": fp,
        "fn": fn,
        "precisao": metricas["precisao"],
        "revocacao": metricas["revocacao"],
        "f1": metricas["f1"],
        "tempo_total_s": tempo_total,
        # Causas dos erros somadas em todas as fotos
        "fp_fusao": sum(linha["fp_fusao"] for linha in linhas_resumo),
        "fp_imprecisa": sum(linha["fp_imprecisa"] for linha in linhas_resumo),
        "fp_espuria": sum(linha["fp_espuria"] for linha in linhas_resumo),
        "fn_fundido": sum(linha["fn_fundido"] for linha in linhas_resumo),
        "fn_caixa_imprecisa": sum(linha["fn_caixa_imprecisa"] for linha in linhas_resumo),
        "fn_nao_detectado": sum(linha["fn_nao_detectado"] for linha in linhas_resumo),
        "tempo_por_foto_s": tempo_total / n,
    }

def centro_da_caixa(caixa):

    # Ponto central de uma caixa (x, y, largura, altura)
    x, y, largura, altura = caixa
    return x + largura / 2, y + altura / 2


def contem_ponto(caixa, ponto):

    # True se o ponto (px, py) está dentro da caixa
    x, y, largura, altura = caixa
    px, py = ponto
    return x <= px <= x + largura and y <= py <= y + altura


def classificar_erros(caixas_detectadas, caixas_gabarito, pares):

    # Explica a causa de cada erro (FP e FN) usando os centros das caixas anotadas
    # pares: resultado de casar_caixas, no formato [(índice detectada, índice anotada, iou), ...]

    # 1) Quantos amendoins anotados têm o centro dentro de cada caixa do programa
    centros = [centro_da_caixa(anotada) for anotada in caixas_gabarito]
    centros_dentro = []
    for detectada in caixas_detectadas:
        quantidade = 0
        for centro in centros:
            if contem_ponto(detectada, centro):
                quantidade += 1
        centros_dentro.append(quantidade)

    # Fusão: caixa do programa que engloba 2 ou mais amendoins (grãos encostados viraram um só)
    # Pode até ter casado com um deles; os outros amendoins dentro dela ficam sem par
    fusoes = [i for i, quantidade in enumerate(centros_dentro) if quantidade >= 2]

    # 2) Causa de cada falso positivo (caixa detectada que não casou)
    #    fusao:     engloba 2+ amendoins
    #    imprecisa: está sobre 1 amendoim, mas com IoU < limiar (caixa grande ou pequena demais)
    #    espuria:   não está sobre nenhum amendoim (ruído, sombra, mancha no fundo)
    detectadas_casadas = {i for i, _, _ in pares}
    fp_fusao = fp_imprecisa = fp_espuria = 0
    for i, quantidade in enumerate(centros_dentro):
        if i in detectadas_casadas:
            continue
        if quantidade >= 2:
            fp_fusao += 1
        elif quantidade == 1:
            fp_imprecisa += 1
        else:
            fp_espuria += 1

    # 3) Causa de cada falso negativo (amendoim anotado que ficou sem par)
    #    fundido:        está dentro de uma caixa de fusão
    #    caixa_imprecisa: está dentro de outra caixa do programa, que não teve IoU suficiente
    #    nao_detectado:  nenhuma caixa do programa passa por ele
    anotadas_casadas = {j for _, j, _ in pares}
    fn_fundido = fn_caixa_imprecisa = fn_nao_detectado = 0
    for j, centro in enumerate(centros):
        if j in anotadas_casadas:
            continue
        if any(contem_ponto(caixas_detectadas[i], centro) for i in fusoes):
            fn_fundido += 1
        elif any(contem_ponto(detectada, centro) for detectada in caixas_detectadas):
            fn_caixa_imprecisa += 1
        else:
            fn_nao_detectado += 1

    return {
        "fp_fusao": fp_fusao,
        "fp_imprecisa": fp_imprecisa,
        "fp_espuria": fp_espuria,
        "fn_fundido": fn_fundido,
        "fn_caixa_imprecisa": fn_caixa_imprecisa,
        "fn_nao_detectado": fn_nao_detectado,
    }