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
