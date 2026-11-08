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
git clone https://github.com/cicerocalsing/TrabalhoGrauBIoT.git
cd TrabalhoGrauBIoT
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Todos os comandos abaixo devem ser executados **de dentro da pasta `TrabalhoGrauBIoT`**.

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

Exploração das anotações (desenha as caixas da base nas fotos de validação e imprime estatísticas):

```bash
.venv\Scripts\python src\explorar_anotacoes.py
```

Saída: `resultados/exploracao/peanuts_XXXX_anotada.jpg`.

_As demais etapas serão adicionadas conforme a implementação._

## Referências

- Grain Detection Dataset — Niranjan Kukkala, Hugging Face, CC BY-NC 4.0.
- Exemplos das Aulas 6, 7 e 8 da disciplina de Processamento Gráfico.
