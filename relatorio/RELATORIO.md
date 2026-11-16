# Relatório técnico


# Pipeline
O pipeline foi dividido em quatro diferentes funções para executar os filtros. As duas primeiras funções tem_fundo_escuro() e segmentar_amendoins() são funções auxiliares, elas servem para auxiliar a execução.
A função tem_fundo_escuro() identifica o formato da imagem. Caso ela seja o fundo inteiro de madeira clara ela utiliza um filtro, caso contrário utiliza o outro filtro.
O segmentar_amendoins() só faz a comparação de qual função deve executar.

Chegando nos filtros em si há duas funções principais: segmentar_fundo_escuro() e segmentar_fundo_madeira().

segmentar_fundo_escuro() - primeiro foi transformada a imagem de RGB (BGR) para LAB e eliminado o canal de transparência. Após isso foi feito um blur gaussiano e transformado a imagem em cinza. Depois disso ele limiariza a imagem e aplica outro blur para eliminar mais ruídos. Em seguida os amendoins são separados pelas pequenas sombras que tinham entre eles, então os que estavam bem grudados tem uma linha cortada entre eles, separando. Por fim é feito uma erosão com a função open seguida de dilatação em elipse, depois uma dilatação e erosão. Dessa forma, os ruídos que sobraram foram mínimos e os amendoins ficaram com poucos buracos dentro.

segmentar_fundo_madeira() - 

_A preencher: componente algorítmico, investigação, resultados, limitações e referências._
