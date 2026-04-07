import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*unrecognized user_version.*')

class BenchmarkPaiPei:
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

    def _calcular_n_estrela(self, gdf_pontos, mapa_bh, area_alvo_km2, tamanho_grid_m=100):
        # Calcula o cenário ótimo teórico (PEI)
        if gdf_pontos.empty or area_alvo_km2 <= 0:
            return 0
            
        xmin, ymin, xmax, ymax = mapa_bh.total_bounds
        cols = np.arange(xmin, xmax, tamanho_grid_m)
        rows = np.arange(ymin, ymax, tamanho_grid_m)
        
        polygons = [Polygon([(x,y), (x+tamanho_grid_m, y), (x+tamanho_grid_m, y+tamanho_grid_m), (x, y+tamanho_grid_m)]) 
                    for x in cols[:-1] for y in rows[:-1]]
                    
        grid = gpd.GeoDataFrame({'geometry': polygons}, crs=self.crs_projetado)
        joined = gpd.sjoin(gdf_pontos, grid, how="left", predicate="within")
        contagem = joined['index_right'].value_counts()
        
        area_celula_km2 = (tamanho_grid_m ** 2) / 10**6
        n_estrela, area_acumulada = 0, 0.0
        
        for num_crimes in contagem:
            if area_acumulada >= area_alvo_km2:
                break
            n_estrela += num_crimes
            area_acumulada += area_celula_km2
            
        return n_estrela

    def _avaliar_modelo_tatico(self, nome_modelo, caminho_arquivo, mapa_bh, area_total_bh, target_area_km2):
        if not os.path.exists(caminho_arquivo):
            print(f"Arquivo não encontrado: {caminho_arquivo}")
            return None
            
        df = pd.read_csv(caminho_arquivo, sep=';', engine='python')
        gdf = self._create_geodataframe(df)
        col_label = 'cluster_label' if 'cluster_label' in gdf.columns else 'cluster'
        
        gdf_treino = gdf[gdf['tipo_dado'] == 'treino']
        gdf_teste = gdf[gdf['tipo_dado'] == 'teste']
        
        N_total_teste = len(gdf_teste) # Denominador da Taxa de Acerto: TOTAL de crimes 2025
        if N_total_teste == 0:
            print(f"  [!] Sem dados de teste para {nome_modelo}")
            return None

        # Desenho do polígono usando os dados de treino (label 1)
        if nome_modelo == 'KDE':
            hotspots_gdf = gdf_treino[gdf_treino[col_label] == 1]
            if hotspots_gdf.empty: return None
            
            poly_tatico = hotspots_gdf.geometry.buffer(50).union_all()
            area_tatico_km2 = poly_tatico.area / 10**6
            
        else:
            # Demais algoritmos: polígonos somente com treino
            clusters_validos = [c for c in gdf_treino[col_label].unique() if c not in [-1, -2]]
            dados_clusters = []
            
            for c in clusters_validos:
                pts_cluster = gdf_treino[gdf_treino[col_label] == c]
                if len(pts_cluster) < 3: 
                    continue 
                    
                hull = pts_cluster.geometry.union_all().convex_hull.buffer(10) #convex hull
                area_c = hull.area / 10**6
                qtd_c = len(pts_cluster)
                densidade = qtd_c / area_c if area_c > 0 else 0
                
                dados_clusters.append({'cluster': c, 'area': area_c, 'crimes': qtd_c, 'densidade': densidade, 'poly': hull})
            
            # Ordenacao decrescente de densidade ate o limite de 30%
            dados_clusters = sorted(dados_clusters, key=lambda x: x['densidade'], reverse=True)
            area_tatico_km2 = 0.0
            poligonos_selecionados = []
            
            for dc in dados_clusters:
                if area_tatico_km2 + dc['area'] <= target_area_km2:
                    area_tatico_km2 += dc['area']
                    poligonos_selecionados.append(dc['poly'])
                elif area_tatico_km2 == 0:
                    area_tatico_km2 += dc['area']
                    poligonos_selecionados.append(dc['poly'])
                    break
                else:
                    break 
            
            if not poligonos_selecionados:
                return None
                
            poly_tatico = gpd.GeoSeries(poligonos_selecionados).union_all()

        # Cruzamento do poligono de treino com o de teste
        capturados_no_futuro = gpd.sjoin(gdf_teste, gpd.GeoDataFrame(geometry=[poly_tatico], crs=self.crs_projetado), how="inner", predicate="within")
        n_capturado = len(capturados_no_futuro)

        # pai pei 
        pai = (n_capturado / N_total_teste) / (area_tatico_km2 / area_total_bh) if area_tatico_km2 > 0 else 0
        n_estrela = self._calcular_n_estrela(gdf_teste, mapa_bh, area_tatico_km2)
        pai_max = (n_estrela / N_total_teste) / (area_tatico_km2 / area_total_bh) if area_tatico_km2 > 0 else 0
        pei = pai / pai_max if pai_max > 0 else 0
        taxa_acerto = (n_capturado / N_total_teste) * 100

        return {
            'Modelo': nome_modelo,
            'Área Tática Alvo (km²)': round(target_area_km2, 2),
            'Área Gerada (km²)': round(area_tatico_km2, 2),
            'Taxa de Acerto (%)': round(taxa_acerto, 2),
            'PAI': round(pai, 4),
            'PEI': round(pei, 4)
        }
        
    def run(self):
        print("Iniciando Benchmark de Eficiência Preditiva...")
        mapa_bh, area_total_bh = self._load_and_project_map()
        target_area_km2 = area_total_bh * self.target_area_percentage
        
        print(f"Limitação Logística (Esforço Tático): {self.target_area_percentage*100}% de BH = {target_area_km2:.2f} km²")
        print("-" * 60)
        
        resultados = []
        for nome, arquivo in self.modelos.items():
            caminho_completo = os.path.join(self.output_dir, arquivo)
            res = self._avaliar_modelo_tatico(nome, caminho_completo, mapa_bh, area_total_bh, target_area_km2)
            if res:
                resultados.append(res)
                print(f"[{nome}] Sucesso -> Acerto: {res['Taxa de Acerto (%)']}% | PAI: {res['PAI']}")
        
        if resultados:
            df_res = pd.DataFrame(resultados)
            # Ordena decrescente pelo PAI
            df_res = df_res.sort_values(by='PAI', ascending=False)
            
            os.makedirs(os.path.join(self.output_dir, 'benchmark'), exist_ok=True)
            caminho_csv = os.path.join(self.output_dir, 'benchmark/tabela_geral_benchmark.csv')
            df_res.to_csv(caminho_csv, sep=';', index=False)
            
            print("-" * 60)
            print(f"\nBenchmark concluído! Tabela salva em: {caminho_csv}")
            print("\nRanking de Eficiência (Melhor PAI):")
            print(df_res[['Modelo', 'Taxa de Acerto (%)', 'PAI', 'PEI']].to_string(index=False))

if __name__ == "__main__":
    benchmark = BenchmarkPaiPei()
    benchmark.run()
