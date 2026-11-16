# Inspeção e quantificação de amendoins com OpenCV

Trabalho do Grau B de Processamento Gráfico (Processamento de Imagens).

## Combinação sorteada

- **Combinação 2** — Tema 1: Inspeção e quantificação de grãos e sementes
- **Subtema:** Amendoim — detectar, contar e medir os objetos da classe `peanuts`
- **Componente algorítmico:** rotulagem de componentes conexos implementada manualmente
  (busca em largura e duas passagens), com conectividade 4 e 8

## Base de dados

[Grain Detection Dataset](https://huggingface.co/datasets/niru-5/manama_course_dataset_grain_detection)
— licença CC BY-NC 4.0.

Parte utilizada: fotografias completas de **validação** e **teste** da classe `peanuts`
(id 1 no COCO) e suas anotações (`annotations/val_photos.json` e `annotations/test_photos.json`).

| Divisão | Fotos | Caixas anotadas | Uso |
|---|---|---|---|
| val  | 11 | 191 | ajuste de parâmetros |
| test | 11 | 344 | avaliação final |

As anotações são usadas apenas como referência de avaliação, nunca como entrada da detecção.

### Dados incluídos no repositório

O recorte usado no trabalho (cerca de 52 MB) já vem no repositório, em `dados/grain_detection/`.
Não é preciso baixar nada por fora. Os arquivos mantêm **os mesmos nomes e a mesma estrutura
da base original**, porque as anotações COCO referenciam as fotos pelo nome do arquivo
(`file_name`) e isso permite conferir cada imagem na fonte.

```
dados/grain_detection/
  README.md            descrição original da base (autoria e licença)
  LICENSE.md           licença CC BY-NC 4.0
  classes.json         lista de classes (peanuts = id 1)
  annotations/
    val_photos.json    anotações COCO das fotos completas de validação (todas as classes)
    test_photos.json   anotações COCO das fotos completas de teste (todas as classes)
  photos/
    val/peanuts_*.jpg  11 fotos de validação
    test/peanuts_*.jpg 11 fotos de teste
```

A redistribuição deste recorte segue a licença CC BY-NC 4.0 da base: uso não comercial,
com atribuição ao autor (ver `dados/grain_detection/README.md` e `LICENSE.md`).

## Instalação

Requer Python 3.10 ou superior.

```bash
git clone https://github.com/cicerocalsing/TrabalhoGrauB-ProcessamentoGrafico.git
cd TrabalhoGrauB-ProcessamentoGrafico
```

Criação do ambiente virtual e instalação das dependências (apenas na primeira vez):

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows**

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Todos os comandos abaixo devem ser executados **de dentro da pasta `TrabalhoGrauB-ProcessamentoGrafico`**
e com o ambiente virtual **ativado** (aparece `(.venv)` no começo da linha do terminal).
Ao abrir um terminal novo, é preciso ativar de novo:

```bash
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

Para desativar: `deactivate`.

## Estrutura

```
dados/              recorte da base (fotos de amendoim + anotações)
src/
  config.py             parâmetros do pipeline
  dados_coco.py         leitura das anotações COCO
  explorar_anotacoes.py desenha as caixas anotadas nas fotos de validação
  segmentacao.py    separação objeto/fundo
  rotulagem.py      rotulagem de componentes conexos (BFS e duas passagens, viz. 4/8)
  medicoes.py       área, largura, altura, razão de aspecto
  visualizacao.py   desenho das regiões
  avaliacao.py      IoU, precisão, revocação, F1, erro de contagem
  main.py           pipeline em lote
  experimento.py    investigação experimental
resultados/         imagens anotadas e tabelas CSV
relatorio/          relatório técnico
docs/               enunciado do trabalho
```

## Execução

Os comandos usam `/` nos caminhos. No Windows, `python src\main.py` também funciona.

### Pipeline principal

Detecta, conta e mede os amendoins de todas as fotos de uma divisão, compara a contagem
com as anotações da base e salva imagens anotadas e tabelas:

```bash
python src/main.py
```

Sem opções, processa a divisão **val** com conectividade 8 e os parâmetros de `src/config.py`.

Opções:

| Opção | Valores | Padrão | Descrição |
|---|---|---|---|
| `--divisao` | `val` ou `test` | `val` | fotos a processar |
| `--conectividade` | `4` ou `8` | `8` | vizinhança da rotulagem de componentes conexos |
| `--area-minima` | número (pixels) | valor de `AREA_MINIMA` em `config.py` | descarta componentes menores (ruído) |
| `--area-maxima` | número (pixels) | sem limite | descarta componentes maiores |

Exemplos:

```bash
python src/main.py --conectividade 4
python src/main.py --area-minima 1000
python src/main.py --conectividade 4 --area-minima 1000
python src/main.py --divisao test        # somente na avaliação final
python src/main.py --help                # lista todas as opções
```

> A divisão `test` deve ser processada apenas na avaliação final, depois que todos os parâmetros
> tiverem sido definidos com a divisão `val`.

Saída em `resultados/<divisao>_conectividade<4|8>/`:

```
imagens/peanuts_XXXX_detectada.jpg   gabarito em vermelho, programa em verde (numerado)
resumo.csv                           uma linha por foto: contagem, gabarito, erro e tempo de cada etapa
medicoes.csv                         uma linha por amendoim: área, caixa, aspecto, preenchimento e centro
```

O número de cada caixa verde na imagem corresponde à coluna `numero` do `medicoes.csv`.

### Scripts auxiliares

Exploração das anotações (desenha as caixas da base nas fotos de validação e imprime estatísticas):

```bash
python src/explorar_anotacoes.py
```

Saída: `resultados/exploracao/peanuts_XXXX_anotada.jpg`.

Visualização da máscara de segmentação de uma foto (abre uma janela; feche com qualquer tecla):

```bash
python src/teste_segmentacao.py
```

### Erros comuns

| Mensagem | Causa | Solução |
|---|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | ambiente virtual não ativado | ativar a `.venv` |
| `can't open file 'src/main.py'` | terminal fora da pasta do projeto | `cd` até a pasta do projeto |
| `command not found: python` | ambiente virtual não ativado | ativar a `.venv` ou usar `python3` |

## Referências

- Grain Detection Dataset — Niranjan Kukkala, Hugging Face, CC BY-NC 4.0.
- Exemplos das Aulas 6, 7 e 8 da disciplina de Processamento Gráfico.
