from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "dados" / "grain_detection"
PASTA_RESULTADOS = RAIZ / "resultados"
DIVISOES_VALIDAS = ("val", "test")
ID_PEANUTS = 1
COR_ANOTACAO = (0, 0, 255)