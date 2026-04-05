import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon
import time

class BenchmarkTemporal:
    def __init__(self):
        self.output_dir = '../output'
        self.gpkg_path = '../data/bh_regional.gpkg'
        self.crs_projetado = "EPSG:31983"

        # ==========================================================================
        # PARÂMETRO TÁTICO DE PATRULHAMENTO (ÁREA ALVO DO TESTE)
        # ==========================================================================
        # A área total de Belo Horizonte é de aproximadamente 331.4 km².
        # Parâmetro é usado para simular a capacidade operacional de patrulhamento.
        # 0.01 (1%)  = ~3.31 km²
        # 0.03 (3%)  = ~9.94 km²
        # 0.05 (5%)  = ~16.57 km²
        # 0.10 (10%) = ~33.14 km²
        # 0.15 (15%) = ~49.71 km²
        # ==========================================================================
        self.target_area_percentage = 0.10
        
        self.modelos = {
            'HDBSCAN': 'hdbscan/hdbscan_data_mcs5.csv',
            'DBSCAN': 'dbscan/dbscan_data_ms9.csv',
            'K-Means': 'kmeans/kmeans_data_k5.csv',
            'K-Medoids': 'kmedoids/kmedoids_data_k7.csv',
            'HAC (Ward)': 'hierarchical/hac_data_k5.csv',
            'KDE': 'kde/kde_baseline_data.csv' 
        }

    def _load_and_project_map(self):
        mapa_bh = gpd.read_file(self.gpkg_path).to_crs(self.crs_projetado)
        area_total_km2 = mapa_bh.geometry.area.sum() / 10**6
        return mapa_bh, area_total_km2

    def _create_geodataframe(self, df):
        colunas_lower = {c.lower(): c for c in df.columns}
        lat_col = colunas_lower.get('latitude', colunas_lower.get('lat'))
        lon_col = colunas_lower.get('longitude', colunas_lower.get('lon'))
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        return gdf.to_crs(self.crs_projetado)

    def _calcular_n_estrela_fatia(self, gdf_fatia, mapa_bh, area_alvo_km2, tamanho_grid_m=100):
        if gdf_fatia.empty or area_alvo_km2 <= 0:
            return 0
            
        xmin, ymin, xmax, ymax = mapa_bh.total_bounds
        cols = np.arange(xmin, xmax, tamanho_grid_m)
        rows = np.arange(ymin, ymax, tamanho_grid_m)
        
        polygons = [Polygon([(x,y), (x+tamanho_grid_m, y), (x+tamanho_grid_m, y+tamanho_grid_m), (x, y+tamanho_grid_m)]) 
                    for x in cols[:-1] for y in rows[:-1]]
                    
        grid = gpd.GeoDataFrame({'geometry': polygons}, crs=self.crs_projetado)
        joined = gpd.sjoin(gdf_fatia, grid, how="left", predicate="within")
        contagem = joined['index_right'].value_counts()
        
        area_celula_km2 = (tamanho_grid_m ** 2) / 10**6
        n_estrela, area_acumulada = 0, 0.0
        
        for num_crimes in contagem:
            if area_acumulada >= area_alvo_km2:
                break
            n_estrela += num_crimes
            area_acumulada += area_celula_km2
            
        return n_estrela

    def _processar_fatias_temporais(self, gdf_teste, poly_tatico, area_tatico_km2, area_total_bh, mapa_bh, nome_modelo):
        resultados_temporais = []
        
        # Forca conversao data_hora
        gdf_teste['data_hora'] = pd.to_datetime(gdf_teste['data_hora'])
        
        # Cria granularidades para ano teste
        gdf_teste['semana'] = gdf_teste['data_hora'].dt.isocalendar().week
        gdf_teste['mes'] = gdf_teste['data_hora'].dt.month
        gdf_teste['bimestre'] = (gdf_teste['data_hora'].dt.month - 1) // 2 + 1
        gdf_teste['trimestre'] = gdf_teste['data_hora'].dt.quarter

        granularidades = {
            'Semanal': 'semana',
            'Mensal': 'mes',
            'Bimestral': 'bimestre',
            'Trimestral': 'trimestre'
        }

        for nome_gran, col_gran in granularidades.items():
            fatias = sorted(gdf_teste[col_gran].unique())
            
            for fatia in fatias:
                gdf_fatia = gdf_teste[gdf_teste[col_gran] == fatia]
                N_fatia = len(gdf_fatia)
                
                if N_fatia == 0:
                    continue
                
                # Cruza poligo teste com poligono de treino
                capturados = gpd.sjoin(gdf_fatia, gpd.GeoDataFrame(geometry=[poly_tatico], crs=self.crs_projetado), how="inner", predicate="within")
                n_capturado = len(capturados)
                
                # pai pei
                pai = (n_capturado / N_fatia) / (area_tatico_km2 / area_total_bh) if area_tatico_km2 > 0 else 0
                n_estrela = self._calcular_n_estrela_fatia(gdf_fatia, mapa_bh, area_tatico_km2)
                pai_max = (n_estrela / N_fatia) / (area_tatico_km2 / area_total_bh) if area_tatico_km2 > 0 else 0
                pei = pai / pai_max if pai_max > 0 else 0
                
                resultados_temporais.append({
                    'Modelo': nome_modelo,
                    'Granularidade': nome_gran,
                    'Periodo': int(fatia),
                    'PAI': pai,
                    'PEI': pei,
                    'Acerto (%)': (n_capturado / N_fatia) * 100
                })
                
        return resultados_temporais
    
    def run(self):
        
        inicio = time.time()
        
        print("Iniciando Benchmark Preditivo Temporal (Treino vs Teste)...")
        mapa_bh, area_total_bh = self._load_and_project_map()
        target_area_km2 = area_total_bh * self.target_area_percentage
        
        todos_resultados = []
        
        for nome, arquivo in self.modelos.items():
            caminho_completo = os.path.join(self.output_dir, arquivo)
            if not os.path.exists(caminho_completo):
                print(f"  [!] Ignorando {nome}: Arquivo não encontrado.")
                continue
                
            print(f"Processando {nome}...")
            df = pd.read_csv(caminho_completo, sep=';', engine='python')
            gdf = self._create_geodataframe(df)
            col_label = 'cluster_label' if 'cluster_label' in gdf.columns else 'cluster'
            
            gdf_treino = gdf[gdf['tipo_dado'] == 'treino'].copy()
            gdf_teste = gdf[gdf['tipo_dado'] == 'teste'].copy()
            
            if gdf_teste.empty:
                continue

            area_tatico_km2 = 0.0
            
            if nome == 'KDE':
                hotspots_gdf = gdf_treino[gdf_treino[col_label] == 1]
                if hotspots_gdf.empty: continue
                poly_tatico = hotspots_gdf.geometry.buffer(50).union_all()
                area_tatico_km2 = poly_tatico.area / 10**6
            else:
                clusters_validos = [c for c in gdf_treino[col_label].unique() if c not in [-1, -2]]
                dados_clusters = []
                
                for c in clusters_validos:
                    pts_cluster = gdf_treino[gdf_treino[col_label] == c]
                    if len(pts_cluster) < 3: continue 
                    hull = pts_cluster.geometry.union_all().convex_hull.buffer(10)
                    area_c = hull.area / 10**6
                    qtd_c = len(pts_cluster)
                    dados_clusters.append({'cluster': c, 'area': area_c, 'densidade': qtd_c / area_c if area_c > 0 else 0, 'poly': hull})
                
                dados_clusters = sorted(dados_clusters, key=lambda x: x['densidade'], reverse=True)
                poligonos_selecionados = []
                
                for dc in dados_clusters:
                    if area_tatico_km2 + dc['area'] <= target_area_km2 or area_tatico_km2 == 0:
                        area_tatico_km2 += dc['area']
                        poligonos_selecionados.append(dc['poly'])
                    else:
                        break 
                
                if not poligonos_selecionados: continue
                poly_tatico = gpd.GeoSeries(poligonos_selecionados).union_all()

            resultados = self._processar_fatias_temporais(gdf_teste, poly_tatico, area_tatico_km2, area_total_bh, mapa_bh, nome)
            todos_resultados.extend(resultados)
            
        # Exportação e Geração de Gráficos
        if todos_resultados:
            df_res = pd.DataFrame(todos_resultados)
            dir_bench = os.path.join(self.output_dir, 'benchmark')
            os.makedirs(dir_bench, exist_ok=True)
            
            df_res.to_csv(os.path.join(dir_bench, 'benchmark_temporal_detalhado.csv'), sep=';', index=False)
            self._gerar_graficos(df_res, dir_bench)
            print(f"Concluído! Tabelas e gráficos guardados na pasta: {dir_bench}")
            
        fim = time.time()
        tempo_total = fim - inicio
        print(f"Tempo de execução do algoritmo: {tempo_total:.4f} segundos")

    def _gerar_graficos(self, df_res, output_folder):
        sns.set_theme(style="whitegrid")
        
        granularidades_plot = ['Semanal', 'Mensal', 'Bimestral', 'Trimestral']
        
        for metrica in ['PAI', 'PEI', 'Acerto (%)']:
            for gran in granularidades_plot:
                df_plot = df_res[df_res['Granularidade'] == gran]
                if df_plot.empty: continue
                
                fig_width = 15 if gran == 'Semanal' else 12
                plt.figure(figsize=(fig_width, 6))
                
                sns.lineplot(data=df_plot, x='Periodo', y=metrica, hue='Modelo', marker='o', linewidth=2, palette='tab10')
                
                titulo_metrica = "Taxa de Acerto Preditiva" if metrica == 'Acerto (%)' else f"Índice {metrica} Preditivo"
                plt.title(f'Evolução da {titulo_metrica} em 2025 ({gran})', fontsize=14, pad=15)
                
                plt.xlabel(gran[:-2] if gran != 'Mensal' else 'Mês', fontsize=12)
                plt.ylabel(metrica, fontsize=12)
                
                # Ajuste dos ticks do eixo X baseado na granularidade
                if gran == 'Mensal': plt.xticks(range(1, 13))
                elif gran == 'Bimestral': plt.xticks(range(1, 7))
                elif gran == 'Trimestral': plt.xticks(range(1, 5))
                elif gran == 'Semanal': plt.xticks(range(1, 53, 2))
                    
                plt.legend(title='Algoritmo', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                nome_arq = f"evolucao_{metrica.lower().replace(' (%)', '').replace(' ', '_')}_{gran.lower()}.png"
                arq_grafico = os.path.join(output_folder, nome_arq)
                
                plt.savefig(arq_grafico, dpi=300, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    benchmark_temporal = BenchmarkTemporal()
    benchmark_temporal.run()
