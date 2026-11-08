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

## Instalação

Requer Python 3.10 ou superior. Na raiz do repositório:

```bash
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Download somente das fotos de amendoim e das anotações (cerca de 40 MB):

```bash
.venv\Scripts\hf download niru-5/manama_course_dataset_grain_detection --repo-type dataset --local-dir dados/grain_detection --include "photos/val/peanuts_*" --include "photos/test/peanuts_*" --include "annotations/*" --include "classes.json" --include "README.md" --include "LICENSE.md"
```

## Estrutura

```
src/
  config.py         parâmetros do pipeline
  dados_coco.py     leitura das anotações COCO
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

_A preencher conforme a implementação._

## Referências

- Grain Detection Dataset — Niranjan Kukkala, Hugging Face, CC BY-NC 4.0.
- Exemplos das Aulas 6, 7 e 8 da disciplina de Processamento Gráfico.
