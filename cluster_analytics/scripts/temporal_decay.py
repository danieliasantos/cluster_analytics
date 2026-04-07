import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*unrecognized user_version.*')


class GeradorPainelDecaimentoHorizontal:
    def __init__(self):
        self.gpkg_path = '../data/bh_regional.gpkg'
        self.hdbscan_csv = '../output/hdbscan/hdbscan_data_mcs5.csv' 
        self.output_dir = '../output/benchmark'
        self.crs_projetado = "EPSG:31983" # Sirgas 2000 / UTM zone 23S (Métrico)
        self.target_area_percentage = 0.10

    def _create_geodataframe(self, df):
        colunas_lower = {c.lower(): c for c in df.columns}
        lat_col = colunas_lower.get('latitude', colunas_lower.get('lat'))
        lon_col = colunas_lower.get('longitude', colunas_lower.get('lon'))
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        return gdf.to_crs(self.crs_projetado)

    def run(self):
        print("Iniciando geração do painel horizontal de decaimento preditivo (1x4)...")
        
        if not os.path.exists(self.gpkg_path):
            print(f"[!] Erro: Arquivo de mapa {self.gpkg_path} não encontrado.")
            return
        mapa_bh = gpd.read_file(self.gpkg_path).to_crs(self.crs_projetado)
        area_total_bh = mapa_bh.geometry.area.sum() / 10**6
        target_area_km2 = area_total_bh * self.target_area_percentage

        if not os.path.exists(self.hdbscan_csv):
            print(f"[!] Erro: Arquivo de dados {self.hdbscan_csv} não encontrado.")
            return

        df = pd.read_csv(self.hdbscan_csv, sep=';', engine='python')
        gdf = self._create_geodataframe(df)
        
        gdf['data_hora'] = pd.to_datetime(gdf['data_hora'])
        gdf['trimestre'] = gdf['data_hora'].dt.quarter
        
        gdf_treino = gdf[gdf['tipo_dado'] == 'treino'].copy()
        gdf_teste = gdf[gdf['tipo_dado'] == 'teste'].copy()

        col_label = 'cluster_label'
        clusters_validos = [c for c in gdf_treino[col_label].unique() if c not in [-1, -2]]
        dados_clusters = []
        
        for c in clusters_validos:
            pts_cluster = gdf_treino[gdf_treino[col_label] == c]
            if len(pts_cluster) < 3: 
                continue 
            hull = pts_cluster.geometry.union_all().convex_hull.buffer(50) 
            area_c = hull.area / 10**6
            qtd_c = len(pts_cluster)
            dados_clusters.append({'cluster': c, 'area': area_c, 'densidade': qtd_c / area_c if area_c > 0 else 0, 'poly': hull})
        
        dados_clusters = sorted(dados_clusters, key=lambda x: x['densidade'], reverse=True)
        area_tatico_km2 = 0.0
        poligonos_selecionados = []
        
        for dc in dados_clusters:
            if area_tatico_km2 + dc['area'] <= target_area_km2 or area_tatico_km2 == 0:
                area_tatico_km2 += dc['area']
                poligonos_selecionados.append(dc['poly'])
            else:
                break
                
        poly_tatico = gpd.GeoDataFrame(geometry=poligonos_selecionados, crs=self.crs_projetado)

        fig, axes = plt.subplots(1, 4, figsize=(24, 12))
        
        fig.suptitle('Decaimento Preditivo HDBSCAN (mcs=5, 10% área coberta)\nTreinamento vs. Avaliação', fontsize=18, fontweight='bold', y=0.98)
        
        trimestres = [1, 2, 3, 4]
        titulos = ['1º Trimestre (Jan-Mar)', '2º Trimestre (Abr-Jun)', '3º Trimestre (Jul-Set)', '4º Trimestre (Out-Dez)']
        
        for ax, trim, titulo in zip(axes.flatten(), trimestres, titulos):
            mapa_bh.plot(ax=ax, color='#c7c7c7', edgecolor='#dddddd', linewidth=0.5)
            
            poly_tatico.plot(ax=ax, color='red', alpha=0.4, edgecolor='darkred', linewidth=1.0, label='Dados de Treino')
            
            gdf_trim = gdf_teste[gdf_teste['trimestre'] == trim]
            
            if not gdf_trim.empty:
                gdf_trim.plot(ax=ax, color='black', markersize=6, alpha=0.6, label='Furtos Reais (Teste)')
            
            ax.set_title(titulo, fontsize=16, pad=8)
            ax.set_axis_off() # Remove eixos com coordenadas
            
            pontos = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=5, alpha=0.6, label=f'{len(gdf_trim)} ocorrências')
            ax.legend(handles=[pontos], loc='lower center', fontsize=10, framealpha=0.9)

        red_patch = mpatches.Patch(color='red', alpha=0.4, label='Dados de Treino (Clusters 2022-2024)')
        black_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=7, alpha=0.6, label='Dados de Avaliação (Ocorrências 2025)')
        
        fig.legend(handles=[red_patch, black_dot], loc='upper center', bbox_to_anchor=(0.5, 0.88), ncol=2, fontsize=15, frameon=False)

        plt.tight_layout(rect=[0, 0.02, 1, 0.82])
        
        os.makedirs(self.output_dir, exist_ok=True)
        out_file = os.path.join(self.output_dir, 'comparativo_decaimento_preditivo.png')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"-> Visualização do decaimento temporal criada com sucesso em: {out_file}")

if __name__ == "__main__":
    GeradorPainelDecaimentoHorizontal().run()
