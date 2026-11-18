# Investigação experimental
#
# Pergunta: é possível distinguir os amendoins individualmente usando apenas filtros clássicos?
# "Distinguir" = cada amendoim anotado tem uma caixa do programa só dele, com IoU >= LIMIAR_IOU
# Critério de sucesso (definido antes do experimento): F1 >= CRITERIO_F1 em cada tipo de fundo
#
# A classe Experimento reúne a investigação feita com uma divisão da base (val ou test):
#   Parte A - Ablação: C0 a C3 x conectividade 4 e 8, com precisão, revocação, F1,
#             causas dos erros e tempo, separados por tipo de fundo
#   Parte B - Rotulagem: compara a rotulagem do grupo com cv2.connectedComponents
# O pipeline principal (só a C2) continua em main.py; esta classe é chamada por
# "python src/main.py --experimento"
#
# Configurações da segmentação (cada uma acrescenta uma etapa à anterior):
#   C0  cinza + Otsu global
#   C1  canal adequado a cada fundo + Otsu, sem morfologia
#   C2  segmentação completa de segmentacao.py (pipeline principal)
#   C3  C2 + separação de grãos encostados (transformada de distância + watershed)

import csv
import time

import cv2
import numpy as np

from avaliacao import avaliar_contagem, avaliar_deteccao, calcular_totais, classificar_erros
from config import (AREA_MAXIMA, AREA_MINIMA, CONFIGURACOES, COR_ANOTACAO,
                    COR_PROGRAMA, CRITERIO_F1, DIVISOES_VALIDAS, FATOR_AREA_GRANDE, FRACAO_NUCLEO,
                    KERNEL_MEDIANA_MADEIRA, KERNEL_SUAVIZACAO_COR, LIMIAR_IOU, LIMIAR_PRETO,
                    PASTA_DADOS, PASTA_RESULTADOS)
from dados_coco import carregar_anotacoes
from medicoes import calcular_areas, extrair_caixas, medir_componentes
from rotulagem import rotular_componentes_conexos
from segmentacao import segmentar_amendoins, tem_fundo_escuro
from visualizacao import desenhar_caixas

CONECTIVIDADES = (4, 8)
FUNDOS = ("todos", "escuro", "madeira")

# Configurações que gravam as imagens anotadas (as outras gravam só os CSVs)
CONFIGURACOES_COM_IMAGENS = ("C2", "C3")

# Máscaras usadas na validação da rotulagem:
# C1 (sem morfologia, muito ruído e muitos componentes) e C2 (pipeline principal)
CONFIGURACOES_VALIDACAO = ("C1", "C2")


def salvar_csv(caminho, linhas):

    # Grava uma lista de dicionários como CSV (cabeçalho = chaves do primeiro dicionário)
    if not linhas:
        return
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)


class Experimento:

    def __init__(self, divisao, area_minima=AREA_MINIMA, area_maxima=AREA_MAXIMA):

        # Atributos: dados do experimento, guardados uma vez e usados por todos os métodos
        if divisao not in DIVISOES_VALIDAS:
            raise ValueError(f"Divisão inválida: {divisao}. Use uma de {DIVISOES_VALIDAS}.")
        self.divisao = divisao
        self.area_minima = area_minima
        self.area_maxima = area_maxima

        pasta_fotos = PASTA_DADOS / "photos" / divisao
        self.fotos = sorted(pasta_fotos.glob("peanuts_*.jpg"))
        if not self.fotos:
            raise FileNotFoundError(f"Nenhuma foto encontrada em {pasta_fotos}")

        # O gabarito é usado só para comparar, nunca como entrada da detecção
        self.anotacoes = carregar_anotacoes(divisao)

    # -----------------------------------------------------------------------
    # Segmentação por configuração
    # -----------------------------------------------------------------------

    @staticmethod
    def limiar_simples(imagem):

        # C0: o mínimo possível - Otsu global no cinza, sem suavização, sem região de interesse
        cinza = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2GRAY)
        _, mascara = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # O Otsu só separa "claro x escuro"; não sabe qual dos dois é o amendoim
        # Os grãos ocupam pouco da foto, então o objeto é o grupo com MENOS pixels
        if (mascara > 0).mean() > 0.5:
            mascara = cv2.bitwise_not(mascara)
        return mascara

    @staticmethod
    def segmentar_por_cor(imagem):

        # C1: limiarização no canal que melhor diferencia o amendoim de cada fundo, sem morfologia
        if tem_fundo_escuro(imagem):
            # Prato: canal b do LAB suavizado, com Otsu calculado só fora dos cantos pretos
            # (é a primeira metade de segmentar_fundo_escuro, sem sombras e sem morfologia)
            lab = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2LAB)
            canal_b = cv2.GaussianBlur(lab[..., 2], (KERNEL_SUAVIZACAO_COR, KERNEL_SUAVIZACAO_COR), 0)
            regiao = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2GRAY) > LIMIAR_PRETO
            limiar, _ = cv2.threshold(canal_b[regiao].reshape(-1, 1), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return ((canal_b > limiar) & regiao).astype(np.uint8) * 255

        # Madeira: a cor do amendoim é parecida com a do fundo, mas ele é um pouco mais escuro
        # A mediana apaga os veios; o Otsu INV marca como objeto o grupo mais escuro
        brilho = cv2.cvtColor(imagem[..., :3], cv2.COLOR_BGR2LAB)[..., 0]
        brilho = cv2.medianBlur(brilho, KERNEL_MEDIANA_MADEIRA)
        _, mascara = cv2.threshold(brilho, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mascara

    @staticmethod
    def separar_encostados(mascara):

        # C3: separa grãos colados que formaram um único componente
        # No meio de cada grão a distância até o fundo é grande; no "pescoço" entre dois grãos
        # encostados ela é pequena. Os centros viram marcadores e o watershed corta entre eles

        # 1) Rótulos e áreas de todos os componentes (rotulagem implementada pelo grupo)
        quantidade, rotulos = rotular_componentes_conexos(mascara, 8)
        areas = calcular_areas(rotulos, quantidade)

        # 2) Só os componentes grandes demais para serem um grão só são candidatos
        # A referência é a mediana das áreas da foto (ignorando o ruído abaixo de AREA_MINIMA)
        areas_validas = areas[1:][areas[1:] >= AREA_MINIMA]
        if len(areas_validas) == 0:
            return mascara
        limite = FATOR_AREA_GRANDE * np.median(areas_validas)

        grande = areas > limite         # grande[r] = True se o componente r é candidato
        grande[0] = False               # o fundo nunca é candidato
        candidatos = grande[rotulos]    # mesma ideia da LUT: rótulo -> True/False por pixel
        if not candidatos.any():
            return mascara

        # 3) Transformada de distância: distância de cada pixel até o fundo mais próximo
        distancia = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)

        # 4) Maior distância de cada componente (o "raio" do componente)
        maior_distancia = np.zeros(quantidade + 1, dtype=np.float32)
        np.maximum.at(maior_distancia, rotulos.ravel(), distancia.ravel())

        # 5) Núcleos: a parte mais "funda" de cada componente candidato
        nucleos = candidatos & (distancia > FRACAO_NUCLEO * maior_distancia[rotulos])
        nucleos = nucleos.astype(np.uint8) * 255

        # 6) Marcadores do watershed: 1 = fundo, 2, 3, ... = cada núcleo, 0 = região a decidir
        _, rotulos_nucleos = rotular_componentes_conexos(nucleos, 8)
        marcadores = (rotulos_nucleos + 1).astype(np.int32)
        marcadores[candidatos & (nucleos == 0)] = 0
        marcadores[~candidatos] = 1

        # 7) Watershed sobre a distância invertida: centros dos grãos são vales,
        # o pescoço entre dois grãos é uma crista; a linha divisória recebe -1
        relevo = cv2.normalize(distancia, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        relevo = cv2.cvtColor(255 - relevo, cv2.COLOR_GRAY2BGR)     # o watershed exige imagem colorida
        cv2.watershed(relevo, marcadores)

        # 8) Apaga as linhas divisórias (engrossadas para 3 px, assim nem a conectividade 8
        # atravessa a linha pela diagonal)
        linhas = (marcadores == -1).astype(np.uint8)
        linhas = cv2.dilate(linhas, np.ones((3, 3), np.uint8)) > 0
        separada = mascara.copy()
        separada[candidatos & linhas] = 0
        return separada

    def segmentar(self, imagem, configuracao):

        # Escolhe a configuração da segmentação (ver CONFIGURACOES em config.py)
        if configuracao == "C0":
            return self.limiar_simples(imagem)
        if configuracao == "C1":
            return self.segmentar_por_cor(imagem)
        if configuracao == "C2":
            return segmentar_amendoins(imagem)
        if configuracao == "C3":
            return self.separar_encostados(segmentar_amendoins(imagem))
        raise ValueError(f"Configuração inválida: {configuracao}. Use uma de {list(CONFIGURACOES)}.")

    # -----------------------------------------------------------------------
    # Uma foto
    # -----------------------------------------------------------------------

    def processar_foto(self, imagem, configuracao, conectividade):

        # Executa o pipeline numa foto e mede o tempo de cada etapa
        inicio = time.perf_counter()
        mascara = self.segmentar(imagem, configuracao)
        t_segmentacao = time.perf_counter() - inicio

        inicio = time.perf_counter()
        quantidade, rotulos = rotular_componentes_conexos(mascara, conectividade)
        t_rotulagem = time.perf_counter() - inicio

        # A contagem final é o número de componentes que passaram pelo filtro de área,
        # e não "quantidade" (que inclui os pontinhos de ruído)
        inicio = time.perf_counter()
        medidas = medir_componentes(rotulos, quantidade, self.area_minima, self.area_maxima)
        t_medicao = time.perf_counter() - inicio

        tempos = {
            "segmentacao": t_segmentacao,
            "rotulagem": t_rotulagem,
            "medicao": t_medicao,
            "total": t_segmentacao + t_rotulagem + t_medicao,
        }
        return medidas, quantidade, tempos

    # -----------------------------------------------------------------------
    # Uma combinação (configuração x conectividade) em todas as fotos
    # -----------------------------------------------------------------------

    def rodar_configuracao(self, configuracao, conectividade, salvar_imagens):

        # Roda uma combinação em todas as fotos, grava os CSVs (e as imagens, se pedido)
        # e devolve as linhas do resumo (uma por foto)
        pasta_saida = PASTA_RESULTADOS / f"{self.divisao}_{configuracao}_conectividade{conectividade}"
        pasta_imagens = pasta_saida / "imagens"
        pasta_saida.mkdir(parents=True, exist_ok=True)
        if salvar_imagens:
            pasta_imagens.mkdir(exist_ok=True)

        linhas_resumo = []      # uma linha por foto
        linhas_medicoes = []    # uma linha por amendoim encontrado

        for caminho in self.fotos:
            nome = caminho.name
            imagem = cv2.imread(str(caminho))
            if imagem is None:
                raise FileNotFoundError(f"Não foi possível abrir a imagem {caminho}")

            fundo = "escuro" if tem_fundo_escuro(imagem) else "madeira"
            medidas, componentes_brutos, tempos = self.processar_foto(imagem, configuracao,
                                                                      conectividade)

            # Avaliação por contagem
            caixas_gabarito = self.anotacoes.get(nome, {}).get("caixas", [])
            contados = len(medidas)
            gabarito = len(caixas_gabarito)
            avaliacao = avaliar_contagem(contados, gabarito)

            # Avaliação por localização: cada caixa verde precisa coincidir (IoU >= LIMIAR_IOU)
            # com uma caixa vermelha diferente para contar como acerto
            caixas_programa = extrair_caixas(medidas)
            deteccao = avaliar_deteccao(caixas_programa, caixas_gabarito, LIMIAR_IOU)
            iou_por_caixa = {i: iou for i, _, iou in deteccao["pares"]}

            # Causa de cada erro: fusão de grãos, caixa imprecisa, ruído ou amendoim não detectado
            erros = classificar_erros(caixas_programa, caixas_gabarito, deteccao["pares"])

            # Imagem anotada: gabarito em vermelho (sem número) e programa em verde (numerado)
            if salvar_imagens:
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

        salvar_csv(pasta_saida / "resumo.csv", linhas_resumo)
        salvar_csv(pasta_saida / "medicoes.csv", linhas_medicoes)

        return linhas_resumo

    # -----------------------------------------------------------------------
    # Execução completa (chamada pelo main.py com a opção --experimento)
    # -----------------------------------------------------------------------

    def executar(self):

        # Investigação completa: ablação (parte A) + validação da rotulagem (parte B)
        pasta_saida = PASTA_RESULTADOS / "experimento"
        pasta_saida.mkdir(parents=True, exist_ok=True)

        print(f"Investigação experimental - divisão: {self.divisao}")
        for codigo, descricao in CONFIGURACOES.items():
            print(f"  {codigo}: {descricao}")

        print()
        print("Parte A - Ablação (C0 a C3 x conectividade 4 e 8)")
        inicio = time.perf_counter()
        comparacao, por_foto = self.executar_ablacao()
        self.imprimir_comparacao(comparacao)
        self.imprimir_conclusao(comparacao)
        salvar_csv(pasta_saida / f"{self.divisao}_comparacao.csv", comparacao)
        salvar_csv(pasta_saida / f"{self.divisao}_por_foto.csv", por_foto)

        print()
        print("Parte B - Validação e tempo da rotulagem")
        validacao = self.validar_rotulagem()
        self.imprimir_validacao(validacao)
        salvar_csv(pasta_saida / f"{self.divisao}_rotulagem.csv", validacao)

        print()
        print(f"Tempo total do experimento: {time.perf_counter() - inicio:.0f} s")
        print(f"Resultados salvos em: {pasta_saida}")

    # -----------------------------------------------------------------------
    # Parte A - Ablação
    # -----------------------------------------------------------------------

    def executar_ablacao(self):

        # Roda todas as combinações e devolve:
        #   comparacao: uma linha por (configuração, conectividade, fundo)
        #   por_foto:   todas as linhas do resumo de todas as execuções
        comparacao = []
        por_foto = []

        for configuracao in CONFIGURACOES:
            for conectividade in CONECTIVIDADES:
                print(f"  rodando {configuracao} com conectividade {conectividade}...")
                salvar_imagens = configuracao in CONFIGURACOES_COM_IMAGENS
                linhas = self.rodar_configuracao(configuracao, conectividade,
                                                 salvar_imagens=salvar_imagens)
                por_foto.extend(linhas)

                for fundo in FUNDOS:
                    if fundo == "todos":
                        grupo = linhas
                    else:
                        grupo = [linha for linha in linhas if linha["fundo"] == fundo]
                    if grupo:
                        comparacao.append(self.resumir_grupo(grupo, configuracao, conectividade,
                                                             fundo))

        return comparacao, por_foto

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    # -----------------------------------------------------------------------
    # Parte B - Validação e tempo da rotulagem
    # -----------------------------------------------------------------------

    @staticmethod
    def mesma_particao(rotulos_a, rotulos_b):

        # Duas rotulagens são equivalentes se separam os pixels nos mesmos grupos,
        # mesmo que os números dos rótulos sejam diferentes (o OpenCV numera em outra ordem)
        # Cada par (rótulo em A, rótulo em B) deve ser único dos dois lados
        pares = np.unique(np.stack([rotulos_a.ravel(), rotulos_b.ravel()], axis=1), axis=0)
        return (len(pares) == len(np.unique(pares[:, 0]))
                and len(pares) == len(np.unique(pares[:, 1])))

    def validar_rotulagem(self):

        # Compara a rotulagem manual com a do OpenCV em cada máscara real, nas duas conectividades
        linhas = []

        for caminho in self.fotos:
            imagem = cv2.imread(str(caminho))
            if imagem is None:
                raise FileNotFoundError(f"Não foi possível abrir a imagem {caminho}")
            fundo = "escuro" if tem_fundo_escuro(imagem) else "madeira"

            for configuracao in CONFIGURACOES_VALIDACAO:
                mascara = self.segmentar(imagem, configuracao)

                for conectividade in CONECTIVIDADES:
                    inicio = time.perf_counter()
                    quantidade, rotulos = rotular_componentes_conexos(mascara, conectividade)
                    tempo_manual = time.perf_counter() - inicio

                    inicio = time.perf_counter()
                    total_opencv, rotulos_opencv = cv2.connectedComponents(
                        mascara, connectivity=conectividade)
                    tempo_opencv = time.perf_counter() - inicio

                    # O OpenCV conta o fundo como um componente; a nossa função não
                    quantidade_opencv = total_opencv - 1
                    igual = (quantidade == quantidade_opencv
                             and self.mesma_particao(rotulos, rotulos_opencv))

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

    @staticmethod
    def imprimir_validacao(linhas):

        identicas = sum(1 for linha in linhas if linha["identico"] == "sim")
        print()
        print(f"Rotulagem manual x cv2.connectedComponents: {identicas} de {len(linhas)} "
              f"máscaras com resultado idêntico")
        print()
        print(f"{'Config':<7} {'Conect':>6} {'Componentes':>11} {'Manual (s)':>11} "
              f"{'OpenCV (s)':>11} {'Vezes mais lento':>17}")

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

