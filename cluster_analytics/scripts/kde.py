import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from shapely.geometry import shape
from rasterio import features
from affine import Affine
from base_algorithm import BaseAlgorithm
import time

class KDEAnalytics(BaseAlgorithm):
    def __init__(self):
        super().__init__()
        self.crs_projetado = "EPSG:31983"
        self.bandwidth_metros = 1000  # Raio de suavização em metros
        self.grid_size_metros = 50   # Resolução espacial do mapa de calor (50x50m)
        self.hotspot_percentage = 0.10
        
    def run(self):
        
        inicio = time.time()
        
        print(f"Iniciando KDE (Grid Contínuo) - Bandwidth: {self.bandwidth_metros}m...")
        
        # Reprojeção para UTM (Metros)
        colunas_lower = {c.lower(): c for c in self.df.columns}
        lat_col = colunas_lower.get('latitude', colunas_lower.get('lat'))
        lon_col = colunas_lower.get('longitude', colunas_lower.get('lon'))
        
        pontos_gdf = gpd.GeoDataFrame(
            self.df, 
            geometry=gpd.points_from_xy(self.df[lon_col], self.df[lat_col]),
            crs="EPSG:4326"
        ).to_crs(self.crs_projetado)
        
        limite_bh = self.mapa_bh.to_crs(self.crs_projetado)
        limite_bh['dissolve_col'] = 1
        limite_unificado = limite_bh.dissolve(by='dissolve_col')
        area_total_bh = limite_unificado.geometry.area.iloc[0]
        
        print(f"Área Total de BH: {area_total_bh / 10**6:.2f} km²")
        
        # Construção do Grid
        minx, miny, maxx, maxy = limite_unificado.total_bounds
        minx = np.floor(minx / self.grid_size_metros) * self.grid_size_metros
        miny = np.floor(miny / self.grid_size_metros) * self.grid_size_metros
        maxx = np.ceil(maxx / self.grid_size_metros) * self.grid_size_metros
        maxy = np.ceil(maxy / self.grid_size_metros) * self.grid_size_metros
        
        xx, yy = np.meshgrid(
            np.arange(minx, maxx + self.grid_size_metros, self.grid_size_metros),
            np.arange(miny, maxy + self.grid_size_metros, self.grid_size_metros)
        )
        grid_points_xy = np.vstack([xx.ravel(), yy.ravel()]).T
        
        grid_centers_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(grid_points_xy[:, 0], grid_points_xy[:, 1]),
            crs=self.crs_projetado
        )
        
        # Filtra apenas as células que caem dentro de BH
        valid_cells = gpd.sjoin(grid_centers_gdf, limite_unificado, how="inner", predicate="within")
        valid_mask_1d = np.full(len(grid_centers_gdf), False)
        valid_mask_1d[valid_cells.index] = True
        valid_mask_2d = valid_mask_1d.reshape(xx.shape)
        
        # Treinamento do KDE
        points_xy = np.array(list(zip(pontos_gdf.geometry.x, pontos_gdf.geometry.y)))
        kde = KernelDensity(bandwidth=self.bandwidth_metros, kernel='gaussian')
        kde.fit(points_xy)
        
        # Cálculo de Densidade na Superfície
        Z = np.zeros(xx.shape)
        Z_log = kde.score_samples(grid_points_xy[valid_mask_1d])
        Z[valid_mask_2d] = np.exp(Z_log)
        
        # Definição do Threshold baseado em Área
        hotspot_target_area = area_total_bh * self.hotspot_percentage
        grid_cell_area = self.grid_size_metros ** 2
        
        valid_densities = Z[valid_mask_2d]
        sorted_indices = np.argsort(valid_densities)[::-1]
        sorted_densities = valid_densities[sorted_indices]
        
        cumulative_area = np.cumsum(np.full_like(sorted_densities, grid_cell_area))
        
        try:
            threshold_index = np.where(cumulative_area >= hotspot_target_area)[0][0]
            density_threshold = sorted_densities[threshold_index]
        except IndexError:
            density_threshold = 0.0
            
        # Extração dos Polígonos de Hotspot
        transform = Affine(self.grid_size_metros, 0.0, minx, 0.0, -self.grid_size_metros, yy.max())
        shapes = features.shapes(Z, mask=(Z >= density_threshold), transform=transform)
        
        polygons = [shape(poly_shape) for poly_shape, value in shapes if value > 0]
        
        if not polygons:
            print("Erro: Nenhum polígono de hotspot gerado.")
            return
            
        hotspot_gdf = gpd.GeoDataFrame(geometry=polygons, crs=self.crs_projetado).union_all()
        hotspot_gdf = gpd.GeoDataFrame(geometry=[hotspot_gdf], crs=self.crs_projetado)
        
        print(f"Área delimitada pelo KDE: {hotspot_gdf.geometry.area.iloc[0] / 10**6:.2f} km²")
        
        # Classificação dos Pontos (PAI/PEI)
        hits_gdf = gpd.sjoin(pontos_gdf, hotspot_gdf, how="inner", predicate="within")
        self.df['cluster_label'] = 0
        self.df.loc[hits_gdf.index, 'cluster_label'] = 1

        self.save_clustered_data(self.df['cluster_label'].values, 'kde/kde_baseline_data.csv')        
        
        # Geração do Mapa Visual
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Desenha o mapa base
        limite_unificado.plot(ax=ax, color='whitesmoke', edgecolor='black', linewidth=0.8)
        
        # Plota a superfície contínua do KDE (Gradiente de Calor)
        # Substitui os zeros por NaN para que as áreas sem crime fiquem transparentes
        Z_plot = np.where(Z > 0, Z, np.nan)
        heatmap = ax.contourf(xx, yy, Z_plot, levels=50, cmap='Reds', alpha=0.6)
        
        # Desenha apenas a LINHA de fronteira do polígono
        hotspot_gdf.boundary.plot(ax=ax, color='darkred', linewidth=2, linestyle='--', label=f'Limite Tático ({self.hotspot_percentage*100:.1f}% Área)')
        
        # Adiciona uma barra de legendas de densidade
        cbar = plt.colorbar(heatmap, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label('Densidade Criminal Estimada')
        
        plt.title(f"Baseline: Superfície KDE e Limite Tático de Patrulhamento\nBandwidth: {self.bandwidth_metros}m")
        plt.axis('off')
        
        # Ajuste de legenda
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='lower right')
        
        map_path = os.path.join(self.output_dir, 'kde/kde_baseline_map.png')
        plt.savefig(map_path, dpi=300, bbox_inches='tight')
        plt.close()

        fim = time.time()
        tempo_total = fim - inicio
        print(f"Tempo de execução do algoritmo: {tempo_total:.4f} segundos")
        
        print(f"Concluído! Dados salvos em: kde/kde_baseline_data.csv")

if __name__ == "__main__":
    analytics = KDEAnalytics()
    analytics.run()
