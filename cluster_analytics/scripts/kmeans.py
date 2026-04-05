import os
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator
from base_algorithm import BaseAlgorithm
import time

class KMeansAnalytics(BaseAlgorithm):
    def __init__(self):
        super().__init__()
        self.crs_projetado = "EPSG:31983"  # UTM 22S - Metros
        self.k_range = range(2, 16)

    def _get_utm_coords(self):
        # Converte as coordenadas originais de graus
        colunas_lower = {c.lower(): c for c in self.df.columns}
        lat_col = colunas_lower.get('latitude', colunas_lower.get('lat'))
        lon_col = colunas_lower.get('longitude', colunas_lower.get('lon'))

        pontos_gdf = gpd.GeoDataFrame(
            self.df,
            geometry=gpd.points_from_xy(self.df[lon_col], self.df[lat_col]),
            crs="EPSG:4326"
        ).to_crs(self.crs_projetado)

        return np.array(list(zip(pontos_gdf.geometry.x, pontos_gdf.geometry.y)))

    def run(self):
        
        inicio = time.time()
        
        print("Preparando K-Means com projeção UTM...")
        self.X_utm = self._get_utm_coords()

        wcss_bic_data, val_metrics_data = [], []

        print("Avaliando candidatos a K ótimo (Elbow e BIC)...")
        for k in self.k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_utm) # Algoritmo treina sobre metros reais

            wcss = kmeans.inertia_
            bic = GaussianMixture(n_components=k, random_state=42).fit(self.X_utm).bic(self.X_utm)
            wcss_bic_data.append({'k': k, 'WCSS': wcss, 'BIC': bic})

            val_metrics_data.append({
                'k': k,
                'silhouette': silhouette_score(self.X_utm, labels),
                'davies_bouldin': davies_bouldin_score(self.X_utm, labels),
                'calinski_harabasz': calinski_harabasz_score(self.X_utm, labels)
            })

        # Salva os CSVs
        os.makedirs(os.path.join(self.output_dir, 'kmeans'), exist_ok=True)
        pd.DataFrame(wcss_bic_data).to_csv(f"{self.output_dir}/kmeans/kmeans_elbow_bic.csv", sep=';', index=False)
        pd.DataFrame(val_metrics_data).to_csv(f"{self.output_dir}/kmeans/kmeans_metrics_all_k.csv", sep=';', index=False)

        kneedle = KneeLocator(list(self.k_range), [d['WCSS'] for d in wcss_bic_data], curve="convex", direction="decreasing")
        k_elbow = kneedle.elbow if kneedle.elbow else 2
        k_bic = self.k_range[np.argmin([d['BIC'] for d in wcss_bic_data])]

        print(f"Candidatos -> Elbow: {k_elbow} | BIC: {k_bic}")

        if k_elbow == k_bic:
            final_k = k_elbow
        else:
            m_e = next(m for m in val_metrics_data if m['k'] == k_elbow)
            m_b = next(m for m in val_metrics_data if m['k'] == k_bic)

            votos_e = sum([m_e['silhouette'] > m_b['silhouette'], m_e['davies_bouldin'] < m_b['davies_bouldin'], m_e['calinski_harabasz'] > m_b['calinski_harabasz']])
            votos_b = 3 - votos_e
            final_k = k_elbow if votos_e > votos_b else k_bic
            print(f"Desempate: Elbow ({votos_e} votos) x BIC ({votos_b} votos) -> Vencedor: k={final_k}")

        # Re-treina e processa saídas para o K ótimo
        kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
        labels = kmeans_final.fit_predict(self.X_utm)

        self.save_clustered_data(labels, f'kmeans/kmeans_data_k{final_k}.csv')
        self.plot_and_save_map(labels, f'kmeans/kmeans_map_k{final_k}.png', title=f"K-Means (Projeção UTM, k={final_k})")

        # Salva métricas de validação da partição final
        metricas_finais = next(m for m in val_metrics_data if m['k'] == final_k)
        df_finais = pd.DataFrame([metricas_finais])
        df_finais['algoritmo'] = 'KMeans'
        df_finais.to_csv(f"{self.output_dir}/kmeans/kmeans_metrics_k{final_k}.csv", sep=';', index=False)

        fim = time.time()
        tempo_total = fim - inicio
        print(f"Tempo de execução do algoritmo: {tempo_total:.4f} segundos")

        print("Processamento do K-Means concluído.")

if __name__ == "__main__":
    analytics = KMeansAnalytics()
    analytics.run()
