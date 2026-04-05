import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AnaliseGraficaTemporal:
    def __init__(self):
        self.input_file = '../output/benchmark/benchmark_temporal_detalhado.csv'
        self.output_dir = '../output/benchmark'

    def run(self):
        if not os.path.exists(self.input_file):
            print(f"[!] Arquivo {self.input_file} não encontrado.")
            return

        print("Lendo dados detalhados e processando consolidação...")
        df = pd.read_csv(self.input_file, sep=';')

        ordem_gran = ['Semanal', 'Mensal', 'Bimestral', 'Trimestral']
        df['Granularidade'] = pd.Categorical(df['Granularidade'], categories=ordem_gran, ordered=True)

        agrupado = df.groupby(['Modelo', 'Granularidade'], observed=False)[['PAI', 'PEI', 'Acerto (%)']].agg(['mean', 'std']).reset_index()
        
        agrupado.columns = ['_'.join(col).strip('_') for col in agrupado.columns.values]
        agrupado = agrupado.rename(columns={
            'PAI_mean': 'PAI_Media', 'PAI_std': 'PAI_Desvio',
            'PEI_mean': 'PEI_Media', 'PEI_std': 'PEI_Desvio',
            'Acerto (%)_mean': 'Acerto_Media', 'Acerto (%)_std': 'Acerto_Desvio'
        }).dropna()

        csv_path = os.path.join(self.output_dir, 'benchmark_temporal_consolidado.csv')
        agrupado.to_csv(csv_path, sep=';', index=False)
        print(f"-> CSV Consolidado salvo em: {csv_path}")

        sns.set_theme(style="whitegrid")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        metricas = [
            ('PAI', 'Índice PAI (Média e Desvio Padrão)', axes[0]),
            ('PEI', 'Índice PEI (Média e Desvio Padrão)', axes[1]),
            ('Acerto (%)', 'Taxa de Acerto % (Média e Desvio)', axes[2])
        ]

        markers = ['o', 's', 'D', '^', 'v', 'p']

        for metrica, titulo, ax in metricas:
            sns.pointplot(
                data=df, 
                x='Granularidade', 
                y=metrica, 
                hue='Modelo', 
                errorbar='sd',
                dodge=0.4,
                markers=markers,
                linestyles='-',
                ax=ax,
                palette='tab10',
                capsize=0.1
            )
            
            ax.set_title(titulo, fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel('Agrupamento Temporal', fontsize=11)
            ax.set_ylabel(metrica, fontsize=11)
            
            if ax != axes[2]:
                ax.get_legend().remove()
            else:
                ax.legend(title='Algoritmo', bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=3, framealpha=1)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'grafico_consolidado_artigo.png')
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    AnaliseGraficaTemporal().run()
