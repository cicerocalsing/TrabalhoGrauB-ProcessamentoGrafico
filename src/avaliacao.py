# Comparação com as anotações da base (gabarito) por contagem
# O gabarito é usado só como referência de avaliação, nunca como entrada da detecção

def avaliar_contagem(contados, gabarito):
    
    # Compara a quantidade encontrada pelo programa com a quantidade de caixas anotadas
    # erro < 0: contou a menos (ex.: grãos grudados) / erro > 0: contou a mais (ex.: ruído)
    erro = contados - gabarito
    return {
        "erro": erro,
        "erro_absoluto": abs(erro),
        # Sem caixas no gabarito não dá para calcular o erro relativo
        "erro_relativo": abs(erro) / gabarito if gabarito > 0 else None,
    }


def calcular_totais(linhas_resumo):
   
    # Métricas agregadas de uma divisão inteira, a partir das linhas do resumo por foto
    # Cada linha precisa ter: gabarito, contados, erro, erro_absoluto e tempo_total_s
    n = len(linhas_resumo)
    tempo_total = sum(linha["tempo_total_s"] for linha in linhas_resumo)

    return {
        "fotos": n,
        "total_gabarito": sum(linha["gabarito"] for linha in linhas_resumo),
        "total_contados": sum(linha["contados"] for linha in linhas_resumo),
        # Erro médio com sinal: negativo indica tendência a contar a menos
        "erro_medio": sum(linha["erro"] for linha in linhas_resumo) / n,
        # Erro absoluto médio (MAE): quantos amendoins o programa erra, em média, por foto
        "erro_absoluto_medio": sum(linha["erro_absoluto"] for linha in linhas_resumo) / n,
        "contagens_exatas": sum(1 for linha in linhas_resumo if linha["erro"] == 0),
        "tempo_total_s": tempo_total,
        "tempo_por_foto_s": tempo_total / n,
    }
