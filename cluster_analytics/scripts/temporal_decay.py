import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

class VisualizacaoDecaimento():
    def __init__(self):
        super().__init__()
        self.gpkg_path = '../data/bh_regional.gpkg'
        self.hdbscan_csv = '../output/hdbscan/hdbscan_data_mcs5.csv'
        self.output_dir = '../output/benchmark'
        self.crs_projetado = "EPSG:31983"
        self.target_area_percentage = 0.10

    def _create_geodataframe(self, df):
        colunas_lower = {c.lower(): c for c in df.columns}
        lat_col = colunas_lower.get('latitude', colunas_lower.get('lat'))
        lon_col = colunas_lower.get('longitude', colunas_lower.get('lon'))
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        return gdf.to_crs(self.crs_projetado)
    
    def run(self):
        print("Iniciando geração do comparativo do decaimento preditivo...")
        
        # Carrega o mapa base
        mapa_bh = gpd.read_file(self.gpkg_path).to_crs(self.crs_projetado)
        area_total_bh = mapa_bh.geometry.area.sum() / 10**6
        target_area_km2 = area_total_bh * self.target_area_percentage
        
        # Carrega os dados processados do HDBSCAN
        if not os.path.exists(self.hdbscan_csv):
            print(f"[!] Erro: Arquivo {self.hdbscan_csv} não encontrado.")
            return
        df = pd.read_csv(self.hdbscan_csv, sep=';', engine='python')
        self.gdf = self._create_geodataframe(df)
        # Garante a coluna de data e extrai o trimestre
        self.gdf['data_hora'] = pd.to_datetime(self.gdf['data_hora'])
        self.gdf['trimestre'] = self.gdf['data_hora'].dt.quarter
        df = None        
        
        # Separa Treino (2022-2024) e Teste (2025)
        col_label = 'cluster_label' if 'cluster_label' in self.gdf.columns else 'cluster'
        gdf_treino = self.gdf[self.gdf['tipo_dado'] == 'treino'].copy()
        gdf_teste = self.gdf[self.gdf['tipo_dado'] == 'teste'].copy()

        # Reconstrói os clusteres de treino (com restricao 10% de área)
        clusters_validos = [c for c in gdf_treino[col_label].unique() if c not in [-1, -2]]
        dados_clusters = []
        
        for c in clusters_validos:
            pts_cluster = gdf_treino[gdf_treino[col_label] == c]
            if len(pts_cluster) < 3: 
                continue 
            hull = pts_cluster.geometry.union_all().convex_hull.buffer(10) # Buffer para criar o polígono
            area_c = hull.area / 10**6
            qtd_c = len(pts_cluster)
            dados_clusters.append({'cluster': c, 'area': area_c, 'densidade': qtd_c / area_c if area_c > 0 else 0, 'poly': hull})
        
        # Ordena por densidade e seleciona até atingir 10% da área
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

        # Configuração da saida plot 2x2
        fig, axes = plt.subplots(2, 2, figsize=(18, 18))
        fig.suptitle('Decaimento Preditivo:\nDados de Treino (Treino 2022-2024) vs. Dados de Avaliação (2025)', fontsize=16, fontweight='bold', y=0.92)
        
        trimestres = [1, 2, 3, 4]
        titulos = ['1º Trimestre 2025', '2º Trimestre 2025', '3º Trimestre 2025', '4º Trimestre 2025']
        
        for ax, trim, titulo in zip(axes.flatten(), trimestres, titulos):
            # Plota o mapa base
            mapa_bh.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.5)
            
            # Plota os polígonos dos dados de treino do HDBSCAN
            poly_tatico.plot(ax=ax, color='red', alpha=0.4, edgecolor='darkred', linewidth=1.5, label='Dados de treino (Clusters 2022-2024)')
            
            # Filtra os dados de teste para o trimestre atual
            gdf_trim = gdf_teste[gdf_teste['trimestre'] == trim]
            #print(f"Total de ocorrências plotadas no {trim}º Trimestre: {len(gdf_trim)}")
            
            # Plota os crimes reais ocorridos no trimestre
            if not gdf_trim.empty:
                gdf_trim.plot(ax=ax, color='black', markersize=8, alpha=0.7, label='Ocorrências (Teste)')
            
            ax.set_title(titulo, fontsize=14, pad=10)
            ax.set_axis_off()
            
            
            #if trim == 1: # Adiciona a legenda apenas no primeiro gráfico
            # Criar patches manuais para a legenda
            red_patch = mpatches.Patch(color='red', alpha=0.4, label='Treino: Clusters 2022-2024')
            black_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=5, label=f'Avaliação: {len(gdf_trim)} ocorrências {trim}º 2025')
            ax.legend(handles=[red_patch, black_dot], loc='lower right', fontsize=11)

        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        
        # Salva a imagem
        os.makedirs(self.output_dir, exist_ok=True)
        out_file = os.path.join(self.output_dir, 'comparativo_decaimento_preditivo.png')
        plt.savefig(out_file, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"-> Comparativo decaimento preditivo salvo em: {out_file}")

if __name__ == "__main__":
    VisualizacaoDecaimento().run()