# Investigação experimental
#
# Pergunta: é possível distinguir os amendoins individualmente usando apenas filtros clássicos?
# "Distinguir" = cada amendoim anotado tem uma caixa do programa só dele, com IoU >= LIMIAR_IOU
# Critério de sucesso (definido antes do experimento): F1 >= CRITERIO_F1 em cada tipo de fundo
#
# Parte A - Ablação: roda as configurações C0 a C3 com conectividade 4 e 8 e compara
#           precisão, revocação, F1, causas dos erros e tempo, separando por tipo de fundo
# Parte B - Rotulagem: confere a rotulagem implementada pelo grupo contra cv2.connectedComponents
#           nas máscaras reais e compara o tempo das duas
#
# Uso (a partir da raiz do repositório):
#   python src/experimento.py                  # val: ajuste dos parâmetros
#   python src/experimento.py --divisao test   # só na avaliação final, com os parâmetros congelados!

import argparse
import time

import cv2
import numpy as np

from avaliacao import calcular_totais
from config import (AREA_MAXIMA, AREA_MINIMA, CONFIGURACOES, CRITERIO_F1, DIVISOES_VALIDAS,
                    PASTA_DADOS, PASTA_RESULTADOS)
from main import executar, salvar_csv
from rotulagem import rotular_componentes_conexos
from segmentacao import segmentar_amendoins, tem_fundo_escuro

CONECTIVIDADES = (4, 8)
FUNDOS = ("todos", "escuro", "madeira")

# Máscaras usadas na validação da rotulagem:
# C1 (sem morfologia, muito ruído e muitos componentes) e C2 (pipeline principal)
CONFIGURACOES_VALIDACAO = ("C1", "C2")


# ---------------------------------------------------------------------------
# Parte A - Ablação
# ---------------------------------------------------------------------------

def resumir_grupo(linhas, configuracao, conectividade, fundo):

    # Uma linha da tabela comparativa: métricas de um grupo de fotos
    totais = calcular_totais(linhas)
    n = len(linhas)

    return {
        "configuracao": configuracao,
        "conectividade": conectividade,
        "fundo": fundo,
        "fotos": n,
        "gabarito": totais["total_gabarito"],
        "contados": totais["total_contados"],
        "vp": totais["vp"],
        "fp": totais["fp"],
        "fn": totais["fn"],
        "precisao": round(totais["precisao"], 4),
        "revocacao": round(totais["revocacao"], 4),
        "f1": round(totais["f1"], 4),
        "atende_criterio": "sim" if totais["f1"] >= CRITERIO_F1 else "nao",
        "mae_contagem": round(totais["erro_absoluto_medio"], 2),
        "fp_fusao": totais["fp_fusao"],
        "fp_imprecisa": totais["fp_imprecisa"],
        "fp_espuria": totais["fp_espuria"],
        "fn_fundido": totais["fn_fundido"],
        "fn_caixa_imprecisa": totais["fn_caixa_imprecisa"],
        "fn_nao_detectado": totais["fn_nao_detectado"],
        # Componentes antes do filtro de área: mostra quanto ruído cada configuração gera
        "componentes_brutos": sum(linha["componentes_brutos"] for linha in linhas),
        # Tempos médios por foto, separados por etapa
        "segmentacao_s": round(sum(linha["tempo_segmentacao_s"] for linha in linhas) / n, 3),
        "rotulagem_s": round(sum(linha["tempo_rotulagem_s"] for linha in linhas) / n, 3),
        "medicao_s": round(sum(linha["tempo_medicao_s"] for linha in linhas) / n, 3),
        "total_s": round(totais["tempo_por_foto_s"], 3),
    }


def executar_ablacao(divisao):

    # Roda o pipeline em todas as combinações e devolve:
    #   comparacao: uma linha por (configuração, conectividade, fundo)
    #   por_foto:   todas as linhas do resumo de todas as execuções
    comparacao = []
    por_foto = []

    for configuracao in CONFIGURACOES:
        for conectividade in CONECTIVIDADES:
            print(f"  rodando {configuracao} com conectividade {conectividade}...")
            linhas = executar(divisao, configuracao, conectividade, AREA_MINIMA, AREA_MAXIMA,
                              detalhar=False)
            por_foto.extend(linhas)

            for fundo in FUNDOS:
                if fundo == "todos":
                    grupo = linhas
                else:
                    grupo = [linha for linha in linhas if linha["fundo"] == fundo]
                if grupo:
                    comparacao.append(resumir_grupo(grupo, configuracao, conectividade, fundo))

    return comparacao, por_foto


def imprimir_comparacao(comparacao):

    print()
    print(f"{'Config':<7} {'Conect':>6} {'Fundo':<8} {'VP':>4} {'FP':>4} {'FN':>4} "
          f"{'Prec':>6} {'Rev':>6} {'F1':>6} {'F1>=' + str(CRITERIO_F1):>8} "
          f"{'Brutos':>7} {'Tempo/foto':>10}")
    for linha in comparacao:
        print(f"{linha['configuracao']:<7} {linha['conectividade']:>6} {linha['fundo']:<8} "
              f"{linha['vp']:>4} {linha['fp']:>4} {linha['fn']:>4} "
              f"{linha['precisao']:>6.3f} {linha['revocacao']:>6.3f} {linha['f1']:>6.3f} "
              f"{linha['atende_criterio']:>8} {linha['componentes_brutos']:>7} "
              f"{linha['total_s']:>9.2f}s")


def imprimir_conclusao(comparacao):

    # Para cada tipo de fundo, mostra a configuração com maior F1 e se ela atinge o critério
    print()
    print(f"Melhor configuração por tipo de fundo (critério: F1 >= {CRITERIO_F1}):")
    for fundo in ("escuro", "madeira"):
        linhas = [linha for linha in comparacao if linha["fundo"] == fundo]
        if not linhas:
            continue
        melhor = max(linhas, key=lambda linha: linha["f1"])
        situacao = "ATINGE" if melhor["atende_criterio"] == "sim" else "NÃO atinge"
        print(f"  {fundo:<8} {melhor['configuracao']} (conectividade {melhor['conectividade']}): "
              f"F1 = {melhor['f1']:.3f} -> {situacao} o critério")


# ---------------------------------------------------------------------------
# Parte B - Validação e tempo da rotulagem
# ---------------------------------------------------------------------------

def mesma_particao(rotulos_a, rotulos_b):

    # Duas rotulagens são equivalentes se separam os pixels nos mesmos grupos,
    # mesmo que os números dos rótulos sejam diferentes (o OpenCV numera em outra ordem)
    # Cada par (rótulo em A, rótulo em B) que aparece na imagem deve ser único dos dois lados:
    # se um rótulo de A aparecer junto com dois rótulos de B, as partições são diferentes
    pares = np.unique(np.stack([rotulos_a.ravel(), rotulos_b.ravel()], axis=1), axis=0)
    return (len(pares) == len(np.unique(pares[:, 0]))
            and len(pares) == len(np.unique(pares[:, 1])))


def validar_rotulagem(divisao):

    # Compara a rotulagem manual com a do OpenCV em cada máscara real, nas duas conectividades
    pasta_fotos = PASTA_DADOS / "photos" / divisao
    linhas = []

    for caminho in sorted(pasta_fotos.glob("peanuts_*.jpg")):
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            raise FileNotFoundError(f"Não foi possível abrir a imagem {caminho}")
        fundo = "escuro" if tem_fundo_escuro(imagem) else "madeira"

        for configuracao in CONFIGURACOES_VALIDACAO:
            mascara = segmentar_amendoins(imagem, configuracao)

            for conectividade in CONECTIVIDADES:
                inicio = time.perf_counter()
                quantidade, rotulos = rotular_componentes_conexos(mascara, conectividade)
                tempo_manual = time.perf_counter() - inicio

                inicio = time.perf_counter()
                total_opencv, rotulos_opencv = cv2.connectedComponents(mascara,
                                                                       connectivity=conectividade)
                tempo_opencv = time.perf_counter() - inicio

                # O OpenCV conta o fundo como um componente; a nossa função não
                quantidade_opencv = total_opencv - 1
                igual = (quantidade == quantidade_opencv
                         and mesma_particao(rotulos, rotulos_opencv))

                linhas.append({
                    "foto": caminho.name,
                    "fundo": fundo,
                    "configuracao": configuracao,
                    "conectividade": conectividade,
                    "componentes_manual": quantidade,
                    "componentes_opencv": quantidade_opencv,
                    "identico": "sim" if igual else "nao",
                    "tempo_manual_s": round(tempo_manual, 4),
                    "tempo_opencv_s": round(tempo_opencv, 4),
                })

    return linhas


def imprimir_validacao(linhas):

    identicas = sum(1 for linha in linhas if linha["identico"] == "sim")
    print()
    print(f"Rotulagem manual x cv2.connectedComponents: {identicas} de {len(linhas)} "
          f"máscaras com resultado idêntico")
    print()
    print(f"{'Config':<7} {'Conect':>6} {'Componentes':>11} {'Manual (s)':>11} {'OpenCV (s)':>11} "
          f"{'Vezes mais lento':>17}")

    for configuracao in CONFIGURACOES_VALIDACAO:
        for conectividade in CONECTIVIDADES:
            grupo = [linha for linha in linhas
                     if linha["configuracao"] == configuracao
                     and linha["conectividade"] == conectividade]
            componentes = sum(linha["componentes_manual"] for linha in grupo)
            manual = sum(linha["tempo_manual_s"] for linha in grupo) / len(grupo)
            opencv = sum(linha["tempo_opencv_s"] for linha in grupo) / len(grupo)
            vezes = manual / opencv if opencv > 0 else 0
            print(f"{configuracao:<7} {conectividade:>6} {componentes:>11} {manual:>11.3f} "
                  f"{opencv:>11.4f} {vezes:>16.0f}x")


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def ler_argumentos():
    parser = argparse.ArgumentParser(description="Investigação experimental: C0 a C3, conectividade 4 e 8")
    parser.add_argument("--divisao", choices=DIVISOES_VALIDAS, default="val",
                        help="val para ajuste, test só para a avaliação final (padrão: val)")
    return parser.parse_args()


if __name__ == "__main__":
    divisao = ler_argumentos().divisao
    pasta_saida = PASTA_RESULTADOS / "experimento"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    print(f"Investigação experimental - divisão: {divisao}")
    for codigo, descricao in CONFIGURACOES.items():
        print(f"  {codigo}: {descricao}")

    print()
    print("Parte A - Ablação (C0 a C3 x conectividade 4 e 8)")
    inicio = time.perf_counter()
    comparacao, por_foto = executar_ablacao(divisao)
    imprimir_comparacao(comparacao)
    imprimir_conclusao(comparacao)
    salvar_csv(pasta_saida / f"{divisao}_comparacao.csv", comparacao)
    salvar_csv(pasta_saida / f"{divisao}_por_foto.csv", por_foto)

    print()
    print("Parte B - Validação e tempo da rotulagem")
    validacao = validar_rotulagem(divisao)
    imprimir_validacao(validacao)
    salvar_csv(pasta_saida / f"{divisao}_rotulagem.csv", validacao)

    print()
    print(f"Tempo total do experimento: {time.perf_counter() - inicio:.0f} s")
    print(f"Resultados salvos em: {pasta_saida}")