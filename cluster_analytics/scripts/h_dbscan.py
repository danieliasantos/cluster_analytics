import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from base_algorithm import BaseAlgorithm
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*unrecognized user_version.*')

class HDBSCANAnalytics(BaseAlgorithm):
    def __init__(self):
        super().__init__()
        self.X_rad = self.get_haversine_coords()
        
        # --- JITTERING ESPACIAL ---
        # Adiciona ruído de ordem microscópica (1e-8 radianos, centímetros no mundo real)
        # para evitar a sobreposição perfeita de pontos (mesma coordenada) que
        # gera divisão por zero (NaN) no cálculo do DBCV.
        np.random.seed(42)
        jitter = np.random.normal(loc=0.0, scale=1e-8, size=self.X_rad.shape)
        self.X_rad = self.X_rad + jitter
        
        self.range_min_cluster_size = range(30, 31)#range(3, 31) #5 é o limite mínimo

    def _gerar_mapas_evolucao(self, melhor_mcs):
        print("\nGerando mapas de evolução temporal dos clusters (2024-2025)...")

        # Carrega o mapa base
        mapa_bh = gpd.read_file('../data/bh_regional.gpkg').to_crs("EPSG:31983")

        # Treino e teste apenas em memória
        df_total = pd.concat([self.df, self.df_teste]).copy()

        # Converte para GeoDataFrame Métrico
        gdf_total = gpd.GeoDataFrame(
            df_total, 
            geometry=gpd.points_from_xy(df_total['longitude'], 
            df_total['latitude']),
            crs="EPSG:4326"
        ).to_crs("EPSG:31983")

        # Garante colunas temporais
        gdf_total['data_hora'] = pd.to_datetime(gdf_total['data_hora'])
        gdf_total['ano'] = gdf_total['data_hora'].dt.year
        gdf_total['trimestre'] = gdf_total['data_hora'].dt.quarter

        for ano in [2024, 2025]:
            for trim in [1, 2, 3, 4]:
                gdf_trim = gdf_total[(gdf_total['ano'] == ano) & (gdf_total['trimestre'] == trim)].copy()

                if len(gdf_trim) < melhor_mcs:
                    continue

                # Roda o HDBSCAN para treinar dados do trimestre - visualização da dinâmica espacial
                coords_rad = np.radians(gdf_trim[['latitude', 'longitude']])
                clusterer = hdbscan.HDBSCAN(min_cluster_size=melhor_mcs, metric='haversine')
                gdf_trim['trim_cluster'] = clusterer.fit_predict(coords_rad)

                # Configura o Plot
                fig, ax = plt.subplots(figsize=(10, 10))
                mapa_bh.plot(ax=ax, color='#cfcfcf', edgecolor='white', linewidth=1)

                clusters_validos = [c for c in gdf_trim['trim_cluster'].unique() if c != -1]
                poligonos = []

                # Desenha os Polígonos Convex Hull com Buffer de 50 metros
                for c in clusters_validos:
                    pts = gdf_trim[gdf_trim['trim_cluster'] == c]
                    if len(pts) >= 3:
                        hull = pts.geometry.union_all().convex_hull.buffer(50)
                        poligonos.append(hull)
                    elif len(pts) > 0:
                        hull = pts.geometry.union_all().buffer(50)
                        poligonos.append(hull)

                if poligonos:
                    gdf_poly = gpd.GeoDataFrame(geometry=poligonos, crs="EPSG:31983")
                    gdf_poly.plot(ax=ax, color='red', alpha=0.5, edgecolor='darkred', linewidth=1.5)

                # Desenha o Ruído
                ruido = gdf_trim[gdf_trim['trim_cluster'] == -1]
                ruido.plot(ax=ax, color='black', markersize=3, alpha=0.5)
                
                # Estética e Legenda
                ax.set_title(f'{trim}º Trimestre de {ano}', fontsize=16, pad=15)
                ax.set_axis_off()

                # red_patch = mpatches.Patch(color='red', alpha=0.5, label='Clusters HDBSCAN')
                # black_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=5, alpha=0.6, label='Ruído')
                # ax.legend(handles=[red_patch, black_dot], loc='lower right', fontsize=11, framealpha=0.9)

                # Salva a imagem
                out_path = os.path.join(self.output_dir, f'hdbscan/evolucao_hdbscan_{ano}-T{trim}.png')
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  -> Mapa salvo: evolucao_hdbscan_{ano}-T{trim}.png")
        
    def run(self):
        
        inicio = time.time()
        
        print("Executando HDBSCAN e avaliando min_cluster_size (5 a 30) via DBCV...")
        
        melhor_score_validade = -2.0 #fallback
        melhor_mcs = 5 #fallback
        limite_ruido_operacional = 30.0  # Teto de ruído máximo aceitável
        
        resultados_sensibilidade = []
        
        for mcs in self.range_min_cluster_size:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                metric='haversine',
                gen_min_span_tree=True
            )
            clusterer.fit(self.X_rad)
            validade = clusterer.relative_validity_
            labels_temp = clusterer.labels_
            
            # Tratamento de segurança caso o algoritmo agrupe mal e o DBCV retorne NaN
            if np.isnan(validade):
                validade = -2.0
                
            qtd_ruido = np.sum(labels_temp == -1)
            perc_ruido = (qtd_ruido / len(labels_temp)) * 100 if len(labels_temp) > 0 else 0
            
            resultados_sensibilidade.append({
                'min_cluster_size': mcs, 
                'dbcv_score': validade,
                'percentual_ruido': perc_ruido
            })
            
            # Seleciona o melhor modelo com base no DBCV, respeitando o limite de ruído
            if validade > melhor_score_validade and perc_ruido <= limite_ruido_operacional:
                melhor_score_validade = validade
                melhor_mcs = mcs
                
        # Fallback caso a mancha exija estruturalmente mais ruído que o limite
        if melhor_score_validade == -2.0:
            print(f"[Aviso] Nenhum teste respeitou o limite de {limite_ruido_operacional}% de ruído. Priorizando menor ruído.")
            melhor_resultado = min(resultados_sensibilidade, key=lambda x: x['percentual_ruido'])
            melhor_mcs = melhor_resultado['min_cluster_size']
            melhor_score_validade = melhor_resultado['dbcv_score']

        print(f"Melhor min_cluster_size (DBCV c/ restrição de ruído): {melhor_mcs} (Score: {melhor_score_validade:.4f})")
        
        # Processamento Final do Melhor Modelo
        clusterer_final = hdbscan.HDBSCAN(
            min_cluster_size=melhor_mcs,
            metric='haversine',
            gen_min_span_tree=True
        )
        labels = clusterer_final.fit_predict(self.X_rad)
        
        # Exportação da Sensibilidade e Gráfico de Eixo Duplo
        df_sensibilidade = pd.DataFrame(resultados_sensibilidade)
        os.makedirs(os.path.join(self.output_dir, 'hdbscan'), exist_ok=True)
        caminho_csv_sens = os.path.join(self.output_dir, 'hdbscan/hdbscan_sensibilidade_metrics.csv')
        df_sensibilidade.to_csv(caminho_csv_sens, sep=';', index=False)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        cor_dbcv = 'tab:blue'
        ax1.set_xlabel('Min Cluster Size')
        ax1.set_ylabel('Score DBCV', color=cor_dbcv)
        ax1.plot(df_sensibilidade['min_cluster_size'], 
            df_sensibilidade['dbcv_score'], 
            marker='o', 
            color=cor_dbcv, 
            label='DBCV'
        )
        ax1.tick_params(axis='y', labelcolor=cor_dbcv)
        
        ax2 = ax1.twinx()  
        cor_ruido = 'tab:orange'
        ax2.set_ylabel('Percentual de Ruído (%)', color=cor_ruido)  
        ax2.plot(
            df_sensibilidade['min_cluster_size'], 
            df_sensibilidade['percentual_ruido'], 
            marker='s', 
            linestyle=':', 
            color=cor_ruido, 
            label='% Ruído'
        )
        ax2.tick_params(axis='y', labelcolor=cor_ruido)
        
        ax1.axvline(x=melhor_mcs, color='red', linestyle='--', label=f'MCS Escolhido ({melhor_mcs})')
        ax2.axhline(y=limite_ruido_operacional, color='gray', linestyle='-.', alpha=0.7, label=f'Teto de Ruído ({limite_ruido_operacional}%)')
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        
        plt.title('Sensibilidade: DBCV e Ruído vs Min Cluster Size (HDBSCAN)')
        fig.tight_layout()  
        plt.grid(True, linestyle=':', alpha=0.5)
        
        caminho_plot_sens = os.path.join(self.output_dir, 'hdbscan/hdbscan_sensibilidade_grafico.png')
        plt.savefig(caminho_plot_sens, dpi=300, bbox_inches='tight')
        plt.close()
        
        m_final = df_sensibilidade[df_sensibilidade['min_cluster_size'] == melhor_mcs].iloc[0]
        metrics = {
            'algoritmo': 'HDBSCAN',
            'min_cluster_size': melhor_mcs,
            'dbcv': m_final['dbcv_score'],
            'percentual_ruido': m_final['percentual_ruido']
        }
        
        self.save_metrics_to_csv(metrics, f'hdbscan/hdbscan_metrics_mcs{melhor_mcs}.csv')
        self.save_clustered_data(labels, f'hdbscan/hdbscan_data_mcs{melhor_mcs}.csv')
        self.plot_and_save_map(labels, f'hdbscan/hdbscan_map_mcs{melhor_mcs}.png', title=f"HDBSCAN (mcs={melhor_mcs})")

        fim = time.time()
        tempo_total = fim - inicio
        print(f"Tempo de execução do algoritmo: {tempo_total:.4f} segundos")
            
        self._gerar_mapas_evolucao(melhor_mcs)
        
        print("Processamento do HDBSCAN concluído.")

if __name__ == "__main__":
    analytics = HDBSCANAnalytics()
    analytics.run()
