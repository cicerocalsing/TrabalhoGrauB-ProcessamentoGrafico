# Pipeline completo em lote: processa todas as fotos de um split (val ou test),
# salva imagens anotadas e os CSVs em resultados/.
#
# Fluxo de cada foto:
#   ler -> segmentar -> rotular -> medir (filtro de área) -> contar
#       -> comparar contagem com o gabarito -> desenhar caixas -> salvar
#
# Uso (a partir da raiz do repositório):
#   python src/main.py                         # val, conectividade 8
#   python src/main.py --divisao val --conectividade 4
#   python src/main.py --divisao test          # só na avaliação final!

import argparse
import csv
import time

import cv2

from avaliacao import avaliar_contagem, calcular_totais
from config import (AREA_MAXIMA, AREA_MINIMA, CONECTIVIDADE_PADRAO, COR_ANOTACAO, COR_PROGRAMA,
                    DIVISOES_VALIDAS, PASTA_DADOS, PASTA_RESULTADOS)
from dados_coco import carregar_anotacoes
from medicoes import extrair_caixas, medir_componentes
from rotulagem import rotular_componentes_conexos
from segmentacao import segmentar_amendoins, tem_fundo_escuro
from visualizacao import desenhar_caixas


def processar_foto(imagem, conectividade, area_minima, area_maxima):
    
    # Executa o pipeline numa foto e mede o tempo de cada etapa
    inicio = time.perf_counter()
    
    mascara = segmentar_amendoins(imagem)
    
    t_segmentacao = time.perf_counter() - inicio

    inicio = time.perf_counter()
    
    quantidade, rotulos = rotular_componentes_conexos(mascara, conectividade)
    
    t_rotulagem = time.perf_counter() - inicio

    inicio = time.perf_counter()
    # A contagem final é o número de componentes que passaram pelo filtro de área,
    # e não "quantidade" (que inclui os pontinhos de ruído)
    medidas = medir_componentes(rotulos, quantidade, area_minima, area_maxima)
    t_medicao = time.perf_counter() - inicio

    tempos = {
        "segmentacao": t_segmentacao,
        "rotulagem": t_rotulagem,
        "medicao": t_medicao,
        "total": t_segmentacao + t_rotulagem + t_medicao,
    }
    return medidas, quantidade, tempos


def executar(divisao, conectividade, area_minima, area_maxima):
    
    # Percorre as fotos da pasta
    pasta_fotos = PASTA_DADOS / "photos" / divisao
    fotos = sorted(pasta_fotos.glob("peanuts_*.jpg"))
    if not fotos:
        raise FileNotFoundError(f"Nenhuma foto encontrada em {pasta_fotos}")

    # O gabarito é usado só para comparar, nunca como entrada da detecção
    anotacoes = carregar_anotacoes(divisao)

    # Uma pasta por configuração, para poder comparar execuções diferentes
    pasta_saida = PASTA_RESULTADOS / f"{divisao}_conectividade{conectividade}"
    pasta_imagens = pasta_saida / "imagens"
    pasta_imagens.mkdir(parents=True, exist_ok=True)

    linhas_resumo = []      # uma linha por foto
    linhas_medicoes = []    # uma linha por amendoim encontrado

    print(f"Divisão: {divisao} | conectividade: {conectividade} | "
          f"área mínima: {area_minima} | área máxima: {area_maxima}")
    print()
    print(f"{'Foto':<18} {'Fundo':<8} {'Gabarito':>8} {'Contados':>8} {'Erro':>5} {'Tempo (s)':>9}")

    for caminho in fotos:
        nome = caminho.name
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            raise FileNotFoundError(f"Não foi possível abrir a imagem {caminho}")

        fundo = "escuro" if tem_fundo_escuro(imagem) else "madeira"

        medidas, componentes_brutos, tempos = processar_foto(
            imagem, conectividade, area_minima, area_maxima)

        caixas_gabarito = anotacoes.get(nome, {}).get("caixas", [])
        contados = len(medidas)
        gabarito = len(caixas_gabarito)
        avaliacao = avaliar_contagem(contados, gabarito)

        # Imagem anotada: gabarito em vermelho (sem número) e programa em verde (numerado)
        # O número de cada caixa verde corresponde à coluna "numero" do medicoes.csv
        anotada = desenhar_caixas(imagem, caixas_gabarito, COR_ANOTACAO, numerar=False)
        anotada = desenhar_caixas(anotada, extrair_caixas(medidas), COR_PROGRAMA)

        saida = pasta_imagens / f"{caminho.stem}_detectada.jpg"

        if not cv2.imwrite(str(saida), anotada):
            raise OSError(f"Falha ao salvar: {saida}")

        linhas_resumo.append({
            "foto": nome,
            "fundo": fundo,
            "largura": imagem.shape[1],
            "altura": imagem.shape[0],
            "componentes_brutos": componentes_brutos,
            "contados": contados,
            "gabarito": gabarito,
            "erro": avaliacao["erro"],
            "erro_absoluto": avaliacao["erro_absoluto"],
            "erro_relativo": ("" if avaliacao["erro_relativo"] is None
                              else round(avaliacao["erro_relativo"], 4)),
            "tempo_segmentacao_s": round(tempos["segmentacao"], 3),
            "tempo_rotulagem_s": round(tempos["rotulagem"], 3),
            "tempo_medicao_s": round(tempos["medicao"], 3),
            "tempo_total_s": round(tempos["total"], 3),
        })

        for numero, medida in enumerate(medidas, start=1):
            x, y, largura, altura = medida["caixa"]
            centro_x, centro_y = medida["centro"]
            linhas_medicoes.append({
                "foto": nome,
                "numero": numero,
                "area_px": medida["area"],
                "x": x,
                "y": y,
                "largura_px": largura,
                "altura_px": altura,
                "aspecto": round(medida["aspecto"], 3),
                "preenchimento": round(medida["preenchimento"], 3),
                "centro_x": round(centro_x, 1),
                "centro_y": round(centro_y, 1),
            })

        print(f"{nome:<18} {fundo:<8} {gabarito:>8} {contados:>8} "
              f"{avaliacao['erro']:>+5} {tempos['total']:>9.2f}")

    salvar_csv(pasta_saida / "resumo.csv", linhas_resumo)
    salvar_csv(pasta_saida / "medicoes.csv", linhas_medicoes)

    imprimir_totais(linhas_resumo)
    print()
    print(f"Resultados salvos em: {pasta_saida}")


def salvar_csv(caminho, linhas):
    
    # Grava uma lista de dicionários como CSV (cabeçalho = chaves do primeiro dicionário)
    if not linhas:
        return
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)


def imprimir_totais(linhas_resumo):
    
    # Mostra no terminal as métricas agregadas da divisão (calculadas em avaliacao.py)
    totais = calcular_totais(linhas_resumo)

    print()
    print(f"Total no gabarito:          {totais['total_gabarito']}")
    print(f"Total contado:              {totais['total_contados']}")
    print(f"Erro médio (com sinal):     {totais['erro_medio']:+.2f} por foto")
    print(f"Erro absoluto médio (MAE):  {totais['erro_absoluto_medio']:.2f} por foto")
    print(f"Fotos com contagem exata:   {totais['contagens_exatas']} de {totais['fotos']}")
    print(f"Tempo total:                {totais['tempo_total_s']:.2f} s "
          f"({totais['tempo_por_foto_s']:.2f} s por foto)")


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------

def ler_argumentos():
    parser = argparse.ArgumentParser(description="Detecção, contagem e medição de amendoins")
    parser.add_argument("--divisao", choices=DIVISOES_VALIDAS, default="val",
                        help="val para ajuste, test só para a avaliação final (padrão: val)")
    parser.add_argument("--conectividade", type=int, choices=(4, 8), default=CONECTIVIDADE_PADRAO,
                        help=f"vizinhança da rotulagem (padrão: {CONECTIVIDADE_PADRAO})")
    parser.add_argument("--area-minima", type=int, default=AREA_MINIMA,
                        help=f"descarta componentes menores que isto, em pixels (padrão: {AREA_MINIMA})")
    parser.add_argument("--area-maxima", type=int, default=AREA_MAXIMA,
                        help="descarta componentes maiores que isto, em pixels (padrão: sem limite)")
    return parser.parse_args()


if __name__ == "__main__":
    argumentos = ler_argumentos()
    executar(argumentos.divisao, argumentos.conectividade,
             argumentos.area_minima, argumentos.area_maxima)
