# Pipeline completo em lote: processa todas as fotos de um split (val ou test),
# salva imagens anotadas e os CSVs em resultados/.
#
# Fluxo de cada foto:
#   ler -> segmentar -> rotular -> medir (filtro de área) -> contar
#       -> comparar contagem com o gabarito -> desenhar caixas -> salvar
#
# Uso (a partir da raiz do repositório):
#   python src/main.py                         # val, configuração C2, conectividade 8
#   python src/main.py --divisao val --conectividade 4
#   python src/main.py --configuracao C3       # outra configuração da segmentação
#   python src/main.py --divisao test          # só na avaliação final!

import argparse
import csv
import time

import cv2

from avaliacao import avaliar_contagem, avaliar_deteccao, calcular_totais, classificar_erros
from config import (AREA_MAXIMA, AREA_MINIMA, CONECTIVIDADE_PADRAO, CONFIGURACAO_PADRAO,
                    CONFIGURACOES, COR_ANOTACAO, COR_PROGRAMA, DIVISOES_VALIDAS, LIMIAR_IOU,
                    PASTA_DADOS, PASTA_RESULTADOS)
from dados_coco import carregar_anotacoes
from medicoes import extrair_caixas, medir_componentes
from rotulagem import rotular_componentes_conexos
from segmentacao import segmentar_amendoins, tem_fundo_escuro
from visualizacao import desenhar_caixas


def processar_foto(imagem, configuracao, conectividade, area_minima, area_maxima):
    
    # Executa o pipeline numa foto e mede o tempo de cada etapa
    inicio = time.perf_counter()
    
    mascara = segmentar_amendoins(imagem, configuracao)
    
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


def executar(divisao, configuracao, conectividade, area_minima, area_maxima, detalhar=True):

    # Roda o pipeline em todas as fotos da divisão, grava imagens e CSVs e DEVOLVE as linhas
    # do resumo (uma por foto), para que o experimento possa comparar várias execuções
    # detalhar=True imprime a tabela de cada foto e os totais; False roda em silêncio    
    # Percorre as fotos da pasta
    
    pasta_fotos = PASTA_DADOS / "photos" / divisao
    fotos = sorted(pasta_fotos.glob("peanuts_*.jpg"))
    if not fotos:
        raise FileNotFoundError(f"Nenhuma foto encontrada em {pasta_fotos}")

    # O gabarito é usado só para comparar, nunca como entrada da detecção
    anotacoes = carregar_anotacoes(divisao)

    # Uma pasta por configuração, para poder comparar execuções diferentes
    pasta_saida = PASTA_RESULTADOS / f"{divisao}_{configuracao}_conectividade{conectividade}"
    pasta_imagens = pasta_saida / "imagens"
    pasta_imagens.mkdir(parents=True, exist_ok=True)

    linhas_resumo = []      # uma linha por foto
    linhas_medicoes = []    # uma linha por amendoim encontrado

    if detalhar:
        print(f"Divisão: {divisao} | configuração: {configuracao} | conectividade: {conectividade} | "
              f"área mínima: {area_minima} | área máxima: {area_maxima}")
        print(f"Segmentação: {CONFIGURACOES[configuracao]}")
        print()
        print(f"{'Foto':<18} {'Fundo':<8} {'Gabarito':>8} {'Contados':>8} {'Erro':>5} "
              f"{'VP':>4} {'FP':>4} {'FN':>4} {'F1':>5} {'Tempo (s)':>9}")
    for caminho in fotos:
        nome = caminho.name
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            raise FileNotFoundError(f"Não foi possível abrir a imagem {caminho}")

        fundo = "escuro" if tem_fundo_escuro(imagem) else "madeira"

        medidas, componentes_brutos, tempos = processar_foto(
            imagem, configuracao, conectividade, area_minima, area_maxima)

        caixas_gabarito = anotacoes.get(nome, {}).get("caixas", [])
        contados = len(medidas)
        gabarito = len(caixas_gabarito)
        avaliacao = avaliar_contagem(contados, gabarito)
        
        # Avaliação por localização: cada caixa verde precisa coincidir (IoU >= LIMIAR_IOU)
        # com uma caixa vermelha diferente para contar como acerto
        caixas_programa = extrair_caixas(medidas)
        deteccao = avaliar_deteccao(caixas_programa, caixas_gabarito, LIMIAR_IOU)

        # IoU de cada caixa do programa que casou: {índice da caixa: iou}
        iou_por_caixa = {i: iou for i, _, iou in deteccao["pares"]}
        
        # Causa de cada erro: fusão de grãos, caixa imprecisa, ruído ou amendoim não detectado
        erros = classificar_erros(caixas_programa, caixas_gabarito, deteccao["pares"])

        # Imagem anotada: gabarito em vermelho (sem número) e programa em verde (numerado)
        # O número de cada caixa verde corresponde à coluna "numero" do medicoes.csv
        anotada = desenhar_caixas(imagem, caixas_gabarito, COR_ANOTACAO, numerar=False)
        anotada = desenhar_caixas(anotada, caixas_programa, COR_PROGRAMA)

        saida = pasta_imagens / f"{caminho.stem}_detectada.jpg"

        if not cv2.imwrite(str(saida), anotada):
            raise OSError(f"Falha ao salvar: {saida}")

        linhas_resumo.append({
            "configuracao": configuracao,
            "conectividade": conectividade,
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
            "vp": deteccao["vp"],
            "fp": deteccao["fp"],
            "fn": deteccao["fn"],
            "precisao": round(deteccao["precisao"], 4),
            "revocacao": round(deteccao["revocacao"], 4),
            "f1": round(deteccao["f1"], 4),
            "iou_medio": round(deteccao["iou_medio"], 4),
            "fp_fusao": erros["fp_fusao"],
            "fp_imprecisa": erros["fp_imprecisa"],
            "fp_espuria": erros["fp_espuria"],
            "fn_fundido": erros["fn_fundido"],
            "fn_caixa_imprecisa": erros["fn_caixa_imprecisa"],
            "fn_nao_detectado": erros["fn_nao_detectado"],
            "tempo_segmentacao_s": round(tempos["segmentacao"], 3),
            "tempo_rotulagem_s": round(tempos["rotulagem"], 3),
            "tempo_medicao_s": round(tempos["medicao"], 3),
            "tempo_total_s": round(tempos["total"], 3),
        })

        for numero, medida in enumerate(medidas, start=1):
            x, y, largura, altura = medida["caixa"]
            centro_x, centro_y = medida["centro"]
            
            # numero começa em 1, mas os índices das caixas começam em 0
            iou = iou_por_caixa.get(numero - 1, 0.0)
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
                "correta": "sim" if (numero - 1) in iou_por_caixa else "nao",
                "iou": round(iou, 3),
            })

        if detalhar:
            print(f"{nome:<18} {fundo:<8} {gabarito:>8} {contados:>8} {avaliacao['erro']:>+5} "
                  f"{deteccao['vp']:>4} {deteccao['fp']:>4} {deteccao['fn']:>4} "
                  f"{deteccao['f1']:>5.2f} {tempos['total']:>9.2f}")

    salvar_csv(pasta_saida / "resumo.csv", linhas_resumo)
    salvar_csv(pasta_saida / "medicoes.csv", linhas_medicoes)

    if detalhar:
        imprimir_totais(linhas_resumo)
        print()
        print(f"Resultados salvos em: {pasta_saida}")

    return linhas_resumo


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
    print()
    print(f"Avaliação por localização (IoU >= {LIMIAR_IOU}):")
    print(f"  VP: {totais['vp']}  FP: {totais['fp']}  FN: {totais['fn']}")
    print(f"  Precisão:   {totais['precisao']:.3f}")
    print(f"  Revocação:  {totais['revocacao']:.3f}")
    print(f"  F1:         {totais['f1']:.3f}")
    print()
    print("Causas dos erros:")
    print(f"  FP: fusão {totais['fp_fusao']} | imprecisa {totais['fp_imprecisa']} | "
          f"espúria {totais['fp_espuria']}")
    print(f"  FN: fundido {totais['fn_fundido']} | caixa imprecisa {totais['fn_caixa_imprecisa']} | "
          f"não detectado {totais['fn_nao_detectado']}")
    print()
    print(f"Tempo total:                {totais['tempo_total_s']:.2f} s "
          f"({totais['tempo_por_foto_s']:.2f} s por foto)")


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------

def ler_argumentos():
    parser = argparse.ArgumentParser(description="Detecção, contagem e medição de amendoins")
    parser.add_argument("--divisao", choices=DIVISOES_VALIDAS, default="val",
                        help="val para ajuste, test só para a avaliação final (padrão: val)")
    parser.add_argument("--configuracao", choices=list(CONFIGURACOES), default=CONFIGURACAO_PADRAO,
                        help=f"configuração da segmentação (padrão: {CONFIGURACAO_PADRAO})")
    parser.add_argument("--conectividade", type=int, choices=(4, 8), default=CONECTIVIDADE_PADRAO,
                        help=f"vizinhança da rotulagem (padrão: {CONECTIVIDADE_PADRAO})")
    parser.add_argument("--area-minima", type=int, default=AREA_MINIMA,
                        help=f"descarta componentes menores que isto, em pixels (padrão: {AREA_MINIMA})")
    parser.add_argument("--area-maxima", type=int, default=AREA_MAXIMA,
                        help="descarta componentes maiores que isto, em pixels (padrão: sem limite)")
    return parser.parse_args()


if __name__ == "__main__":
    argumentos = ler_argumentos()
    executar(argumentos.divisao, argumentos.configuracao, argumentos.conectividade,
             argumentos.area_minima, argumentos.area_maxima)
