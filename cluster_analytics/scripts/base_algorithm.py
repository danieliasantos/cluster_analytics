import os
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

class BaseAlgorithm:
    def __init__(self):
        self.data_path = '../data/base_dados.tsv'
        self.gpkg_path = '../data/bh_regional.gpkg'
        self.json_path = '../data/LIMITE_MUNICIPIO.json'
        self.output_dir = '../output'
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = self._load_data()
        self._preparar_divisao_temporal() 
        self.mapa_bh = gpd.read_file(self.gpkg_path)

    def _load_data(self):
        # Carrega o dataset, remove nulos e aplica Jittering para evitar distâncias = 0.
        df = pd.read_csv(self.data_path, sep='\t')
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Seed para reprodutibilidade
        np.random.seed(42)
        
        # Jittering espacial: Adiciona ruído de um metro para evitar distâncias = 0
        ruido_lat = np.random.normal(0, 0.00001, size=len(df))
        ruido_lon = np.random.normal(0, 0.00001, size=len(df))
        
        df['latitude'] = df['latitude'] + ruido_lat
        df['longitude'] = df['longitude'] + ruido_lon
        
        return df

    def _preparar_divisao_temporal(self):
        # Divide os dados em Treino (2022-2024) e Teste (2025)
        if 'ano' not in self.df.columns:
            if 'data_hora' in self.df.columns:
                self.df['ano'] = pd.to_datetime(self.df['data_hora']).dt.year
            else:
                raise ValueError("O dataset precisa de uma coluna 'ano' ou 'data_hora'.")

        # Filtro anos
        self.df['tipo_dado'] = 'ignorar'
        self.df.loc[self.df['ano'].isin([2022, 2023, 2024]), 'tipo_dado'] = 'treino'
        self.df.loc[self.df['ano'] == 2025, 'tipo_dado'] = 'teste'
        df_completo = self.df[self.df['tipo_dado'] != 'ignorar'].copy()

        self.df_treino = df_completo[df_completo['tipo_dado'] == 'treino'].copy()
        self.df_teste = df_completo[df_completo['tipo_dado'] == 'teste'].copy()
        
        self.df = self.df_treino.copy()
        
        print(f"Dados divididos -> Treino (22-24): {len(self.df_treino)} ocorrências | Teste (25): {len(self.df_teste)} ocorrências")

    def get_haversine_coords(self):
        return np.radians(self.df[['latitude', 'longitude']].values)

    def get_standardized_coords(self):
        scaler = StandardScaler()
        return scaler.fit_transform(self.df[['latitude', 'longitude']])

    def save_metrics_to_csv(self, metrics_dict, filename):
        metrics_df = pd.DataFrame([metrics_dict])
        output_path = os.path.join(self.output_dir, filename)
        metrics_df.to_csv(output_path, sep=';', index=False)
        print(f"Métricas salvas em: {output_path}")

    def save_clustered_data(self, labels, filepath):
        self.df['cluster_label'] = labels        
        self.df_teste['cluster_label'] = -2 
        df_export = pd.concat([self.df, self.df_teste])
        caminho_completo = os.path.join(self.output_dir, filepath)
        df_export.to_csv(caminho_completo, sep=';', index=False)
        print(f"Dados clusterizados salvos em: {caminho_completo}")

    def plot_and_save_map(self, labels, filename, title="Clusters de Furtos de Cabo"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        mapa_bh_wgs84 = self.mapa_bh.to_crs("EPSG:4326")
        
        gdf_pontos = gpd.GeoDataFrame(
            self.df, geometry=gpd.points_from_xy(self.df.longitude, self.df.latitude), crs="EPSG:4326"
        )
        gdf_pontos['cluster'] = labels

        fig, ax = plt.subplots(figsize=(10, 10))
        mapa_bh_wgs84.plot(ax=ax, color='whitesmoke', edgecolor='black', linewidth=0.8)
        
        ruido = gdf_pontos[gdf_pontos['cluster'] == -1]
        clusters = gdf_pontos[gdf_pontos['cluster'] != -1]
        
        handles = []
        if not ruido.empty:
            ruido.plot(ax=ax, color='dimgrey', markersize=5, alpha=0.5)
            handles.append(mpatches.Patch(color='dimgrey', label=f'Ruído ({len(ruido)})'))

        if not clusters.empty:
            unique_clusters = clusters['cluster'].unique()
            
            if len(unique_clusters) <= 20:
                cmap = plt.get_cmap('tab20')
                unique_clusters_sorted = sorted(unique_clusters)
                
                for idx, c in enumerate(unique_clusters_sorted):
                    cluster_pts = clusters[clusters['cluster'] == c]
                    color = cmap(idx / max(1, len(unique_clusters_sorted) - 1)) if len(unique_clusters_sorted) > 1 else cmap(0)
                    cluster_pts.plot(ax=ax, color=color, markersize=15)
                    handles.append(mpatches.Patch(color=color, label=f'C {c} ({len(cluster_pts)})'))
            else:
                clusters.plot(ax=ax, column='cluster', cmap='tab20', markersize=15)
                handles.append(mpatches.Patch(color='none', label=f'Total de Clusters: {len(unique_clusters)}'))

        if handles:
            num_colunas = 1 if len(handles) <= 8 else (2 if len(handles) <= 16 else 3)
            ax.legend(handles=handles, title="Clusterização 10% Área Coberta", loc="lower right", fontsize='small', ncol=num_colunas, framealpha=0.9)

        plt.title(title)
        plt.axis('off')
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Mapa salvo em: {output_path}")