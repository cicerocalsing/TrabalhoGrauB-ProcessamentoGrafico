from pathlib import Path

# Pasta raiz do repositório
# __file__ é o caminho deste arquivo (src/config.py) e resolve() o transforma em caminho absoluto
# O primeiro .parent sobe para a pasta src/ e o segundo sobe para a raiz do repositório
# Assim os caminhos funcionam independentemente da pasta de onde o script for executado
RAIZ = Path(__file__).resolve().parent.parent

# Recorte da base Grain Detection Dataset incluído no repositório (fotos de amendoim + anotações)
PASTA_DADOS = RAIZ / "dados" / "grain_detection"

# Pasta onde o programa salva imagens anotadas e tabelas
PASTA_RESULTADOS = RAIZ / "resultados"

# Divisões da base que possuem fotografias completas
# val é usada para ajustar os parâmetros e test apenas para a avaliação final
DIVISOES_VALIDAS = ("val", "test")

# Identificador da classe sorteada (Combinação 2 - Amendoim) no arquivo classes.json da base
ID_PEANUTS = 1

# Cor das caixas anotadas pela base (referência), no padrão BGR do OpenCV: vermelho
COR_ANOTACAO = (0, 0, 255)

# Cor das caixas encontradas pelo programa, no padrão BGR do OpenCV: verde
COR_PROGRAMA = (0, 255, 0)

# Filtro de regiões espúrias: componentes com área (em pixels) fora desta faixa são descartados
# Os valores devem ser escolhidos usando APENAS as fotos de val
AREA_MINIMA = 1000      # 0 = sem filtro (ainda a definir com o val)
AREA_MAXIMA = None      # None = sem limite superior

# Conectividade padrão da rotulagem de componentes conexos (4 ou 8)
CONECTIVIDADE_PADRAO = 8

# Avaliação por localização: IoU mínimo para uma caixa do programa contar como acerto
# 0.5 é o valor mais usado em detecção de objetos (padrão do PASCAL VOC)
LIMIAR_IOU = 0.5


# ---------------------------------------------------------------------------
# Investigação experimental
# ---------------------------------------------------------------------------

# Critério de sucesso definido ANTES de rodar o experimento:
# um tipo de fundo é considerado "resolvido só com filtros" se o F1 por localização for >= 0.80
CRITERIO_F1 = 0.80


# ---------------------------------------------------------------------------
# Parâmetros das configurações C1 e C3 do experimento
# A C2 é a segmentação de segmentacao.py, que tem os próprios parâmetros no código
# Valores escolhidos olhando apenas as fotos de val
# ---------------------------------------------------------------------------

# C1 - fundo escuro: limiarização do canal b do LAB (o amendoim é amarelado e o prato é neutro)
LIMIAR_PRETO = 15               # brilho (0 a 255) abaixo do qual o pixel é o preto dos cantos
KERNEL_SUAVIZACAO_COR = 7       # filtro gaussiano 7x7 no canal b antes do Otsu

# C1 - fundo de madeira: limiarização do brilho (o amendoim é um pouco mais escuro que a madeira)
KERNEL_MEDIANA_MADEIRA = 11     # filtro de mediana no canal L: apaga os veios da madeira


# ---------------------------------------------------------------------------
# Configurações comparadas na investigação experimental
# ---------------------------------------------------------------------------

# Estudo de ablação: cada configuração acrescenta uma etapa de filtro à anterior
CONFIGURACOES = {
    "C0": "Limiar simples: cinza + Otsu global",
    "C1": "Cor + região de interesse: Otsu no canal adequado ao fundo, sem morfologia",
    "C2": "Método completo de cada fundo: C1 + sombras/bordas + morfologia",
    "C3": "C2 + separação de grãos encostados (transformada de distância + watershed)",
}

# Configuração usada quando nenhuma é escolhida (pipeline principal)
CONFIGURACAO_PADRAO = "C2"

# Separação de grãos encostados (C3)
FATOR_AREA_GRANDE = 1.6     # só tenta separar componentes com área > 1.6 x a mediana das áreas da foto
FRACAO_NUCLEO = 0.6         # núcleo de um grão: pixels com distância à borda > 60% da maior distância do componente
