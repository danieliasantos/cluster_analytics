# Detecção Dinâmica de Zonas de Criminalidade: Avaliação de Métodos de Agrupamento

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-Spatial_Analysis-brightgreen.svg)](https://geopandas.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Submissão_SBBD-success.svg)]()

Repositório oficial com o código-fonte e a metodologia do artigo **"Avaliando Métodos de Agrupamento Baseados em Densidade para Detecção Dinâmica de Zonas de Criminalidade em Ambientes Urbanos"**, submetido ao Simpósio Brasileiro de Bancos de Dados (SBBD).

Este projeto propõe uma arquitetura de inteligência geográfica para o policiamento eficiente no contexto de Cidades Inteligentes, por meio da análise de crimes de furtos de cabos de cobre em Belo Horizonte. O estudo analisa diferentes métodos de clusterização para identificar o seu desempenho e aderência à análise geoespacial, e demonstra o fenômeno do **Decaimento Preditivo**, fundamentando a necessidade de Sistemas de Gerenciamento de Bancos de Dados (SGBDs) processarem inteligência espacial em *streaming* e com indexação otimizada.

---

## Arquitetura do Projeto

A base de código foi construída para garantir rigor metodológico. O experimento adota uma abordagem *out-of-sample*, em que os modelos foram treinados com dados dos anos de 2022 a 2024, e avaliados com dados de 2025.

### Estrutura de Diretórios
```text
.
├── data/                                   # Diretório de dados brutos e shapes
│   ├── base_dados.csv                      # Dataset georreferenciado (18.590 registros)
│   └── bh_regional.gpkg                    # Malha municipal de Belo Horizonte
├── scripts/                                # Códigos fonte da pesquisa
│   ├── base_algorithm.py                   # Classe-mãe (Divisão Temporal e Jittering)
│   ├── kde.py                              # Modelo contínuo de base (KDE)
│   ├── kmeans.py                           # Agrupamento Particional (K-Means)
│   ├── kmedoids.py                         # Agrupamento Particional (K-Medoids)
│   ├── hierarchical.py                     # Agrupamento Hierárquico Aglomerativo (HAC - Ward)
│   ├── dbscan.py                           # Algoritmo de Densidade Clássico (DBSCAN)
│   ├── h_dbscan.py                         # Algoritmo de Densidade Hierárquica (HDBSCAN)
│   ├── pai_pei_benchmark.py                # Avaliador Global
│   ├── pai_pei_benchmark_temporal.py       # Avaliador de Granularidades Temporais
│   ├── pai_pei_benchmark_graphic.py        # Gerador de visualizações gráficas consolidadas
│   ├── friedman_test.py                    # Teste de Frieadman + Post-hoc DE Nemenyi
|   └── temporal_decay.py                   # Demonstra do decaimento preditivo do HDBSCAN
└── output/                                 # Diretório de artefatos gerados
    ├── benchmark/                          # Tabelas CSV e Gráficos conslidados
    └── [nome_algoritmo]/                   # Malhas, mapas HTML/PNG e CSVs por algoritmo
```    
    
### Metodologia e Integridade Espacial

Para garantir a validade topológica e justiça na comparação (fairness):

<ol type="1">
    <li>Modelos de Partição (K-Means, K-Medoids, HAC): As coordenadas são projetadas para a escala cartesiana métrica real (UTM 22S - EPSG:31983) para que as distâncias euclidianas reflitam a realidade urbana.</li>
    <li>Modelos de Densidade (DBSCAN, HDBSCAN): As coordenadas são operadas em radianos puros com a métrica Haversine, respeitando a curvatura terrestre sem distorções de escala.</li>
    <li>Limitação de Esforço Tático: Todos os algoritmos são forçados a ranquear as suas zonas criminais mais densas e parar a captura ao atingir o limite estrito de 15% do território municipal (~49,6 km²).</li>
</ol>


## Como Executar (Reproducibilidade)

Os scripts foram desenvolvidos e testados nativamente em ambiente Linux Debian e empacotados para execução otimizada.

<ol>
  <li>
    <b>Preparando o Ambiente</b><br>
      Recomenda-se o uso de um ambiente virtual Python com Docker. As dependências incluem: <code>pandas</code>, <code>geopandas</code>, <code>numpy</code>, <code>scikit-learn</code>, <code>hdbscan</code>, <code>kneed</code>, <code>folium</code>, <code>matplotlib</code>, <code>seaborn</code> e <code>shapely</code>.
      <br><br>
      <details>
        <summary><b>Bash</b></summary>
 
        # Clone o repositório
        git clone https://github.com/danieliasantos/cluster_analytics
        cd cluster_analytics
        
        # Execute o ambiente Docker e acesse
        sudo docker compose build cable_theft_analysis
        sudo docker compose up -d
        sudo docker exec -it clustering_analytics bash
    
  </details>
  
  </li>
  
  <li>
    <b>Treinamento da Modelagem Espacial</b><br>
      Execute os scripts de clusterização sequencialmente. Anos de treino: 2022 a 2024.
      <br><br>
    <details>
      <summary><b>Bash</b></summary>

        python h_dbscan.py
        python kde.py
        python kmeans.py
        python kmedoids.py
        python hierarchical.py
        python dbscan.py
      
  </details>
    
  </li>

<li>
  <b>Teste da Avaliação Preditiva</b><br>
  Benchmark global para obter a tabela geral com os Índices PAI e PEI do ano de avaliação: 2025.
  <br><br>
      <details>
      <summary><b>Bash</b></summary>

        python pai_pei_benchmark.py

  </details>
</li>

<li>
  <b>Análise do Decaimento Preditivo</b><br>
  Gera os recálculos mensais, bimestrais e trimestrais para analisar a migração da mancha criminal e análise consolidada
  <br><br>
  <details>
    <summary><b>Bash</b></summary>

        python pai_pei_benchmark_temporal.py
        python pai_pei_benchmark_graphic.py
        python friedman_test.py
        python temporal_decay.py
        
  </details>
</li>

</ol>


## Principais Descobertas

<ul>
    <li>Paradoxo do PAI: O KDE atinge métricas de concentração ilusórias ao delimitar perímetros conservadores contínuos, mas é derrotado na proporção absoluta de captura.</li>
    <li>Adequação Topológica: O HDBSCAN atinge a melhor eficiência tática (PEI), pois a sua arquitetura baseada em Grafos (Árvore Geradora Mínima) mapeia crimes de infraestrutura delineando ruas de forma irregular, superando os polígonos inflados gerados pelo K-Means e pelo HAC.</li>
    <li>SGBDs em Cidades Inteligentes: A acurácia preditiva degrada acentuadamente no decorrer do ano, comprovando a obsolescência de mapas criminais estáticos. A governança baseada em dados requer bancos de dados espaço-temporais atuando com ingestão em streaming e consultas de vizinhança indexadas (R-trees).</li>
</ul>

## Autores

<ul>
  <li>Daniel E. Santos - Aluno - Instituto Federal de Educação, Ciência e Tecnologia de Minas Gerais (IFMG).</li>
  <li>Prof. Dr. Carlos A. Silva - Orientador - Instituto Federal de Educação, Ciência e Tecnologia de Minas Gerais (IFMG).</li>
</ul>
