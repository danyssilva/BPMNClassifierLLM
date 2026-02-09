#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera relatório HTML interativo com resultados e rankings
"""

import pandas as pd
import os
from collections import defaultdict
from datetime import datetime

# ==========================================
# 1. CARREGAR DADOS
# ==========================================

csv_files = [f for f in os.listdir('.') if f.startswith('resultados_comparacao_') and f.endswith('.csv')]

if not csv_files:
    print("❌ Nenhum arquivo de resultados encontrado!")
    exit(1)

csv_file = max(csv_files, key=lambda x: os.path.getctime(x))
df = pd.read_csv(csv_file)

# ==========================================
# 1.5 EXTRAIR NOME DO PROCESSO
# ==========================================

process_name = None
if 'SBMN_Arquivo' in df.columns and len(df) > 0:
    sbmn_file = df.iloc[0]['SBMN_Arquivo']
    if '_' in sbmn_file:
        parts = sbmn_file.split('_')
        if len(parts) >= 2:
            process_name = '_'.join(parts[1:]).replace('.json', '')

if not process_name:
    process_name = "results"

output_dir = os.path.join('OUTPUTS', process_name)
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. PREPARAR DADOS
# ==========================================

# Contar votos
votos = {}
for idx, row in df.iterrows():
    vencedor = row['LLM_Voto']
    modelo1 = row['BPMN1_Arquivo']
    modelo2 = row['BPMN2_Arquivo']
    
    if vencedor == "Modelo 1":
        votos[modelo1] = votos.get(modelo1, 0) + 1
    elif vencedor == "Modelo 2":
        votos[modelo2] = votos.get(modelo2, 0) + 1

votos_sorted = sorted(votos.items(), key=lambda x: x[1], reverse=True)
total_votos = sum(votos.values())

# Calcular métricas médias
metricas = defaultdict(lambda: {'hmean': [], 'fitness': [], 'precision': []})

for idx, row in df.iterrows():
    modelo1 = row['BPMN1_Arquivo']
    modelo2 = row['BPMN2_Arquivo']
    
    metricas[modelo1]['hmean'].append(row.get('M1_Hmean', 0))
    metricas[modelo1]['fitness'].append(row.get('M1_Fitness', 0))
    metricas[modelo1]['precision'].append(row.get('M1_Precision', 0))
    
    metricas[modelo2]['hmean'].append(row.get('M2_Hmean', 0))
    metricas[modelo2]['fitness'].append(row.get('M2_Fitness', 0))
    metricas[modelo2]['precision'].append(row.get('M2_Precision', 0))

metricas_media = {}
for modelo, dados in metricas.items():
    metricas_media[modelo] = {
        'hmean': sum(dados['hmean']) / len(dados['hmean']) if dados['hmean'] else 0,
        'fitness': sum(dados['fitness']) / len(dados['fitness']) if dados['fitness'] else 0,
        'precision': sum(dados['precision']) / len(dados['precision']) if dados['precision'] else 0,
    }

# ==========================================
# 3. GERAR HTML
# ==========================================

html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Ranking - Classificador BPMN</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #333;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .ranking-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}
        
        .ranking-table th {{
            background-color: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .ranking-table td {{
            padding: 15px;
            border-bottom: 1px solid #eee;
        }}
        
        .ranking-table tbody tr:hover {{
            background-color: #f9f9f9;
        }}
        
        .ranking-table tbody tr:nth-child(1) {{
            background-color: #f0f8ff;
            font-weight: 600;
        }}
        
        .ranking-table tbody tr:nth-child(2) {{
            background-color: #fff8f0;
        }}
        
        .ranking-table tbody tr:nth-child(3) {{
            background-color: #f5f5f5;
        }}
        
        .medal {{
            font-size: 1.5em;
            margin-right: 10px;
        }}
        
        .medal-1 {{ color: #ffd700; }} /* Ouro */
        .medal-2 {{ color: #c0c0c0; }} /* Prata */
        .medal-3 {{ color: #cd7f32; }} /* Bronze */
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .stat-card h3 {{
            font-size: 0.95em;
            opacity: 0.9;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        
        .comparison-card {{
            background: #f9f9f9;
            border-left: 5px solid #667eea;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 5px;
        }}
        
        .comparison-card h4 {{
            color: #333;
            margin-bottom: 10px;
        }}
        
        .comparison-row {{
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 20px;
            margin-bottom: 10px;
            align-items: center;
        }}
        
        .model-info {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #eee;
        }}
        
        .model-info .name {{
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }}
        
        .model-info .metric {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .vs {{
            text-align: center;
            font-weight: bold;
            color: #667eea;
        }}
        
        .winner {{
            background: #d4edda;
            color: #155724;
            padding: 8px 15px;
            border-radius: 5px;
            text-align: center;
            font-weight: 600;
        }}
        
        .charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        
        .chart-img {{
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .footer {{
            background: #f9f9f9;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #eee;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Relatório de Ranking</h1>
            <p>Classificador Multimodelo BPMN com LLM</p>
            <p style="margin-top: 10px; font-size: 0.95em;">Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <!-- ESTATÍSTICAS GERAIS -->
            <div class="section">
                <h2>📊 Estatísticas Gerais</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Total de Comparações</h3>
                        <div class="value">{len(df)}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Total de Votos</h3>
                        <div class="value">{total_votos}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Modelos Comparados</h3>
                        <div class="value">{len(votos)}</div>
                    </div>
                </div>
            </div>
            
            <!-- RANKING FINAL -->
            <div class="section">
                <h2>🏆 Ranking Final por Votos</h2>
                <table class="ranking-table">
                    <thead>
                        <tr>
                            <th>Posição</th>
                            <th>Arquivo</th>
                            <th>Votos</th>
                            <th>Percentual</th>
                            <th>Hmean Médio</th>
                            <th>Fitness Médio</th>
                            <th>Precision Médio</th>
                        </tr>
                    </thead>
                    <tbody>
"""

for idx, (arquivo, votos_count) in enumerate(votos_sorted, 1):
    percentual = (votos_count / total_votos * 100) if total_votos > 0 else 0
    hmean = metricas_media.get(arquivo, {}).get('hmean', 0)
    fitness = metricas_media.get(arquivo, {}).get('fitness', 0)
    precision = metricas_media.get(arquivo, {}).get('precision', 0)
    
    medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "➖"
    
    html_content += f"""
                        <tr>
                            <td><span class="medal medal-{idx}">{medal}</span> #{idx}</td>
                            <td>{arquivo}</td>
                            <td><strong>{votos_count}</strong></td>
                            <td>{percentual:.1f}%</td>
                            <td>{hmean:.3f}</td>
                            <td>{fitness:.3f}</td>
                            <td>{precision:.3f}</td>
                        </tr>
"""

html_content += """
                    </tbody>
                </table>
            </div>
            
            <!-- DETALHAMENTO DAS COMPARAÇÕES -->
            <div class="section">
                <h2>📋 Detalhamento das Comparações</h2>
"""

for idx, row in df.iterrows():
    m1_file = row['BPMN1_Arquivo']
    m2_file = row['BPMN2_Arquivo']
    vencedor = row['LLM_Voto']
    
    m1_hmean = row.get('M1_Hmean', 0)
    m2_hmean = row.get('M2_Hmean', 0)
    
    m1_fitness = row.get('M1_Fitness', 0)
    m2_fitness = row.get('M2_Fitness', 0)
    
    m1_precision = row.get('M1_Precision', 0)
    m2_precision = row.get('M2_Precision', 0)
    
    if vencedor == "Modelo 1":
        winner = m1_file
        winner_class = "model-info"
    elif vencedor == "Modelo 2":
        winner = m2_file
        winner_class = "model-info"
    else:
        winner = "Empate"
    
    html_content += f"""
                <div class="comparison-card">
                    <h4>Comparação #{idx+1}</h4>
                    <div class="comparison-row">
                        <div class="model-info">
                            <div class="name">{m1_file}</div>
                            <div class="metric">Hmean: {m1_hmean:.3f}</div>
                            <div class="metric">Fitness: {m1_fitness:.3f}</div>
                            <div class="metric">Precision: {m1_precision:.3f}</div>
                        </div>
                        <div class="vs">vs</div>
                        <div class="model-info">
                            <div class="name">{m2_file}</div>
                            <div class="metric">Hmean: {m2_hmean:.3f}</div>
                            <div class="metric">Fitness: {m2_fitness:.3f}</div>
                            <div class="metric">Precision: {m2_precision:.3f}</div>
                        </div>
                    </div>
                    <div class="winner">✅ Vencedor: {winner}</div>
                </div>
"""

html_content += """
            </div>
            
            <!-- GRÁFICOS -->
            <div class="section">
                <h2>📈 Gráficos de Análise</h2>
                <div class="charts">
                    <div>
                        <h4 style="margin-bottom: 10px;">Ranking de Votos</h4>
                        <img src="{process_name}/01_ranking_votos.png" class="chart-img" alt="Ranking de Votos">
                    </div>
                    <div>
                        <h4 style="margin-bottom: 10px;">Hmean Médio</h4>
                        <img src="{process_name}/02_hmean_medio.png" class="chart-img" alt="Hmean Médio">
                    </div>
                    <div>
                        <h4 style="margin-bottom: 10px;">Métricas Detalhadas</h4>
                        <img src="{process_name}/03_metricas_detalhadas.png" class="chart-img" alt="Métricas Detalhadas">
                    </div>
                    <div>
                        <h4 style="margin-bottom: 10px;">Pontuação Ponderada</h4>
                        <img src="{process_name}/04_pontuacao_ponderada.png" class="chart-img" alt="Pontuação Ponderada">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Relatório gerado automaticamente pelo sistema de classificação BPMN com LLM</p>
            <p>Arquivo de origem: {csv_file}</p>
        </div>
    </div>
</body>
</html>
"""

# ==========================================
# 4. SALVAR HTML
# ==========================================

html_file = os.path.join(output_dir, 'ranking_relatorio.html')
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ Relatório HTML gerado com sucesso!")
print(f"📄 Arquivo: {html_file}")
print(f"\n💡 Abra no navegador para visualizar interativamente")
print(f"   O arquivo inclui: ranking, comparações e gráficos")
