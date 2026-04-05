import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

class TesteEstatistico:
    def __init__(self):
        self.input_file = '../output/benchmark/benchmark_temporal_detalhado.csv'

    def run(self):
        print("Carregando dados para análise estatística...")
        df = pd.read_csv(self.input_file, sep=';')

        granularidades = df['Granularidade'].unique()
        
        for gran in granularidades:
            print("\n" + "=" * 50)
            print(f"Testando granularidade {gran}")
            print("=" * 50 + "\n")

            # Filtra granularidade excluindo DBSCAN
            df_filtrado = df[(df['Granularidade'] == gran) & (df['Modelo'] != 'DBSCAN')]

            # Pivota dataset csv: linhas = Semanas; colunas = modelos, valores = PEI
            df_pivot = df_filtrado.pivot(index='Periodo', columns='Modelo', values='PEI').dropna()

            print(f"Total de amostras temporais (Semanas) analisadas: {len(df_pivot)}")
            print(f"Modelos comparados: {list(df_pivot.columns)}\n")

            # TESTE DE FRIEDMAN
            chi, p_value = friedmanchisquare(*[df_pivot[col] for col in df_pivot.columns])
            
            print("-" * 20)
            print("Teste de Friedman")
            print("-" * 20)
            print(f"Chi^2: {chi:.4f}")
            print(f"P-value: {p_value:.6e}")
            
            if p_value < 0.05:
                print("CONCLUSAO: HÁ DIFERENÇA estatisticamente significativa entre os algoritmos, pois p_value < 0.05.\nTeste Post-Hoc de Nemenyi...\n")
                
                # POST-HOC DE NEMENYI
                nemenyi_res = sp.posthoc_nemenyi_friedman(df_pivot.values)
                
                # Nomes das colunas e índices
                nemenyi_res.columns = df_pivot.columns
                nemenyi_res.index = df_pivot.columns
                
                print("-" * 20)
                print("Matriz P-value de Nemenyi")
                print("-" * 250)
                print(nemenyi_res.round(9))
                
                print("\n- Valores da matriz menores que 0.05 indicam que a diferença entre o par de modelos é ESTATISTICAMENTE SIGNIFICATIVA.")
                
            else:
                print("CONCLUSÃO: NÃO HÁ DIFERENÇA estatisticamente significativa entre os algoritmos.")

if __name__ == "__main__":
    teste = TesteEstatistico()
    teste.run()