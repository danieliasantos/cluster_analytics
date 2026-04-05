import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from hdbscan.validity import validity_index  
from kneed import KneeLocator
from base_algorithm import BaseAlgorithm
import time

class DBSCANAnalytics(BaseAlgorithm):
    def __init__(self):
        super().__init__()
        self.X_rad = self.get_haversine_coords()
        
        # JITTERING: Adiciona um ruído (escala de 1e-8 radianos equivale a centímetros)
        # Previne distâncias exatas de zero entre ocorrências no mesmo local, 
        # evitando a divisão por zero (NaN) no cálculo matemático do DBCV.
        np.random.seed(42) # Garante que o ruído seja o mesmo em todas as execuções
        jitter = np.random.normal(loc=0.0, scale=1e-8, size=self.X_rad.shape)
        self.X_rad = self.X_rad + jitter
        
        self.range_min_samples = range(5, 31)
        
    def run(self):
        
        inicio = time.time()
        
        print("Executando DBSCAN e avaliando min_samples (5 a 30) via DBCV...")
        
        melhor_score_dbcv = -2.0 # O DBCV varia de -1 a 1
        melhor_ms = 5
        melhor_eps = 0.001
        limite_ruido_operacional = 30.0  
        
        resultados_sensibilidade = []
        
        for ms in self.range_min_samples:
            # Determinação dinâmica do eps ótimo pelo k-distance
            neighbors = NearestNeighbors(n_neighbors=ms, metric='haversine')
            distances, _ = neighbors.fit(self.X_rad).kneighbors(self.X_rad)
            k_distances = np.sort(distances[:, ms - 1])
            
            kneedle = KneeLocator(range(len(k_distances)), k_distances, curve="convex", direction="increasing")
            eps_otimo = k_distances[kneedle.elbow] if kneedle.elbow else k_distances[-1]
            
            # Execução do DBSCAN
            dbscan = DBSCAN(eps=eps_otimo, min_samples=ms, metric='haversine')
            labels = dbscan.fit_predict(self.X_rad)
            
            # Cálculo do percentual de ruído
            qtd_ruido = np.sum(labels == -1)
            perc_ruido = (qtd_ruido / len(labels)) * 100 if len(labels) > 0 else 0

            # Cálculo do DBCV
            try:
                labels_validos = labels[labels != -1]
                if len(np.unique(labels_validos)) > 1:
                    dbcv_score = validity_index(self.X_rad, labels, metric='haversine')
                else:
                    dbcv_score = -2.0
            except Exception:
                dbcv_score = -2.0
                
            resultados_sensibilidade.append({
                'min_samples': ms, 
                'eps_rad': eps_otimo,
                'dbcv': dbcv_score,
                'percentual_ruido': perc_ruido
            })
            
            # Regra de Seleção: Maior DBCV respeitando o teto de ruído
            if dbcv_score > melhor_score_dbcv and perc_ruido <= limite_ruido_operacional:
                melhor_score_dbcv = dbcv_score
                melhor_ms = ms
                melhor_eps = eps_otimo
                
        # Fallback caso a mancha exija estruturalmente mais ruído do que o limite estipulado
        if melhor_score_dbcv == -2.0:
            print(f"[Aviso] Nenhum teste respeitou o limite de {limite_ruido_operacional}% de ruído.")
            melhor_resultado = min(resultados_sensibilidade, key=lambda x: x['percentual_ruido'])
            melhor_ms = melhor_resultado['min_samples']
            melhor_eps = melhor_resultado['eps_rad']
            melhor_score_dbcv = melhor_resultado['dbcv']

        print(f"Melhor min_samples (DBCV c/ restrição de ruído): {melhor_ms} (Score: {melhor_score_dbcv:.4f})")
        
        # Exportação dos Dados e Gráficos de Sensibilidade
        df_sensibilidade = pd.DataFrame(resultados_sensibilidade)
        
        os.makedirs(os.path.join(self.output_dir, 'dbscan'), exist_ok=True)
        
        caminho_csv_sens = os.path.join(self.output_dir, 'dbscan/dbscan_sensibilidade_metrics.csv')
        df_sensibilidade.to_csv(caminho_csv_sens, sep=';', index=False)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        cor_dbcv = 'tab:blue'
        ax1.set_xlabel('Min Samples')
        ax1.set_ylabel('Score DBCV', color=cor_dbcv)
        ax1.plot(df_sensibilidade['min_samples'], df_sensibilidade['dbcv'], marker='o', color=cor_dbcv, label='DBCV')
        ax1.tick_params(axis='y', labelcolor=cor_dbcv)
        
        ax2 = ax1.twinx()  
        cor_ruido = 'tab:orange'
        ax2.set_ylabel('Percentual de Ruído (%)', color=cor_ruido)  
        ax2.plot(df_sensibilidade['min_samples'], df_sensibilidade['percentual_ruido'], marker='s', linestyle=':', color=cor_ruido, label='% Ruído')
        ax2.tick_params(axis='y', labelcolor=cor_ruido)
        
        ax1.axvline(x=melhor_ms, color='red', linestyle='--', label=f'MS Escolhido ({melhor_ms})')
        ax2.axhline(y=limite_ruido_operacional, color='gray', linestyle='-.', alpha=0.7, label=f'Teto de Ruído ({limite_ruido_operacional}%)')
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        
        plt.title('Sensibilidade: DBCV e Ruído vs Min Samples (DBSCAN)')
        fig.tight_layout()  
        plt.grid(True, linestyle=':', alpha=0.5)
        
        caminho_plot_sens = os.path.join(self.output_dir, 'dbscan/dbscan_sensibilidade_grafico.png')
        plt.savefig(caminho_plot_sens, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Processamento Final do Melhor Modelo
        dbscan_final = DBSCAN(eps=melhor_eps, min_samples=melhor_ms, metric='haversine')
        labels = dbscan_final.fit_predict(self.X_rad)
        m_final = df_sensibilidade[df_sensibilidade['min_samples'] == melhor_ms].iloc[0]
        
        metrics = {
            'algoritmo': 'DBSCAN',
            'min_samples': melhor_ms,
            'eps_rad': melhor_eps,
            'dbcv': m_final['dbcv'],
            'percentual_ruido': m_final['percentual_ruido']
        }
        
        self.save_metrics_to_csv(metrics, f'dbscan/dbscan_metrics_ms{melhor_ms}.csv')
        self.save_clustered_data(labels, f'dbscan/dbscan_data_ms{melhor_ms}.csv')
        self.plot_and_save_map(labels, f'dbscan/dbscan_map_ms{melhor_ms}.png', title=f"DBSCAN (ms={melhor_ms}, eps={melhor_eps:.4f})")
        
        fim = time.time()
        tempo_total = fim - inicio
        print(f"Tempo de execução do algoritmo: {tempo_total:.4f} segundos")
    
        print("Processamento do DBSCAN concluído.")

if __name__ == "__main__":
    analytics = DBSCANAnalytics()
    analytics.run()
