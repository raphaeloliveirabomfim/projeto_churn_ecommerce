"""
Módulo para pré-processamento dos dados
"""
import pandas as pd
import numpy as np

def carregar_dados():
    """Carrega os dados brutos"""
    clientes = pd.read_csv("../data/raw/clientes.csv")
    transacoes = pd.read_csv("../data/raw/transacoes.csv")
    return clientes, transacoes

def limpar_dados(df):
    """Funções de limpeza serão adicionadas aqui"""
    pass

if __name__ == "__main__":
    print("Módulo de pré-processamento carregado")