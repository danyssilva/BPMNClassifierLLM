#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizações gráficas dos resultados - Ranking com gráficos de barras
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI
import os
from collections import defaultdict

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
# 2. PREPARAR DADOS PARA GRÁFICOS
# ==========================================

# Contar votos por modelo
votos = {}
for idx, row in df.iterrows():
    vencedor = row['LLM_Voto']
    modelo1 = row['BPMN1_Arquivo']
    modelo2 = row['BPMN2_Arquivo']
    
    if vencedor == "Modelo 1":
        votos[modelo1] = votos.get(modelo1, 0) + 1
    elif vencedor == "Modelo 2":
        votos[modelo2] = votos.get(modelo2, 0) + 1

# Calcular métricas médias por modelo
metricas = defaultdict(lambda: {'hmean': [], 'fitness': [], 'precision': [], 'generalization': [], 'simplicity': []})

for idx, row in df.iterrows():
    modelo1 = row['BPMN1_Arquivo']
    modelo2 = row['BPMN2_Arquivo']
    
    metricas[modelo1]['hmean'].append(row.get('M1_Hmean', 0))
    metricas[modelo1]['fitness'].append(row.get('M1_Fitness', 0))
    metricas[modelo1]['precision'].append(row.get('M1_Precision', 0))
    metricas[modelo1]['generalization'].append(row.get('M1_Generalization', 0))
    metricas[modelo1]['simplicity'].append(row.get('M1_Simplicity', 0))
    
    metricas[modelo2]['hmean'].append(row.get('M2_Hmean', 0))
    metricas[modelo2]['fitness'].append(row.get('M2_Fitness', 0))
    metricas[modelo2]['precision'].append(row.get('M2_Precision', 0))
    metricas[modelo2]['generalization'].append(row.get('M2_Generalization', 0))
    metricas[modelo2]['simplicity'].append(row.get('M2_Simplicity', 0))

# Calcular médias
metricas_media = {}
for modelo, dados in metricas.items():
    metricas_media[modelo] = {
        'hmean': sum(dados['hmean']) / len(dados['hmean']) if dados['hmean'] else 0,
        'fitness': sum(dados['fitness']) / len(dados['fitness']) if dados['fitness'] else 0,
        'precision': sum(dados['precision']) / len(dados['precision']) if dados['precision'] else 0,
        'generalization': sum(dados['generalization']) / len(dados['generalization']) if dados['generalization'] else 0,
        'simplicity': sum(dados['simplicity']) / len(dados['simplicity']) if dados['simplicity'] else 0,
    }

# ==========================================
# 3. GRÁFICO 1: VOTOS POR MODELO
# ==========================================

modelos = list(votos.keys())
votos_count = list(votos.values())

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(modelos))]
bars = ax.bar(modelos, votos_count, color=colors, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Número de Votos', fontsize=12, fontweight='bold')
ax.set_title('🏆 Ranking de Votos por Modelo', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(votos_count) * 1.2)

# Adicionar valores nas barras
for i, (bar, count) in enumerate(zip(bars, votos_count)):
    height = bar.get_height()
    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count)}\n{medal}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_ranking_votos.png'), dpi=150, bbox_inches='tight')
print("✅ Gráfico 1: 01_ranking_votos.png")
plt.close()

# ==========================================
# 4. GRÁFICO 2: HMEAN MÉDIO POR MODELO
# ==========================================

modelos_hmean = list(metricas_media.keys())
hmean_values = [metricas_media[m]['hmean'] for m in modelos_hmean]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if i == 1 else '#2ecc71' if i == 0 else '#3498db' for i in range(len(modelos_hmean))]
bars = ax.bar(modelos_hmean, hmean_values, color=colors, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Hmean (Harmonic Mean)', fontsize=12, fontweight='bold')
ax.set_title('📊 Hmean Médio por Modelo', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)

# Adicionar valores nas barras
for bar, val in zip(bars, hmean_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_hmean_medio.png'), dpi=150, bbox_inches='tight')
print("✅ Gráfico 2: 02_hmean_medio.png")
plt.close()

# ==========================================
# 5. GRÁFICO 3: COMPARAÇÃO DE MÉTRICAS
# ==========================================

fig, axes = plt.subplots(1, len(modelos_hmean), figsize=(6 + 5*len(modelos_hmean), 6))

if len(modelos_hmean) == 1:
    axes = [axes]

for idx, modelo in enumerate(modelos_hmean):
    ax = axes[idx]
    
    metrics = ['Fitness', 'Precision', 'Generalization', 'Simplicity', 'Hmean']
    values = [
        metricas_media[modelo]['fitness'],
        metricas_media[modelo]['precision'],
        metricas_media[modelo]['generalization'],
        metricas_media[modelo]['simplicity'],
        metricas_media[modelo]['hmean']
    ]
    
    colors_metrics = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    ax.barh(metrics, values, color=colors_metrics, edgecolor='black', linewidth=1)
    
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Valor', fontsize=11, fontweight='bold')
    ax.set_title(f'{modelo}', fontsize=12, fontweight='bold')
    
    # Adicionar valores
    for i, (metric, val) in enumerate(zip(metrics, values)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

plt.suptitle('📈 Detalhamento de Métricas por Modelo', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_metricas_detalhadas.png'), dpi=150, bbox_inches='tight')
print("✅ Gráfico 3: 03_metricas_detalhadas.png")
plt.close()

# ==========================================
# 6. GRÁFICO 4: PONTUAÇÃO PONDERADA
# ==========================================

# Calcular score ponderado
scores_data = []
for modelo in modelos_hmean:
    votos_m = votos.get(modelo, 0)
    hmean_m = metricas_media[modelo]['hmean']
    score = (votos_m * 100) + (hmean_m * 50)
    scores_data.append({'modelo': modelo, 'votos': votos_m, 'hmean': hmean_m, 'score': score})

# Ordenar por score
scores_data.sort(key=lambda x: x['score'], reverse=True)

modelos_score = [s['modelo'] for s in scores_data]
scores_values = [s['score'] for s in scores_data]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ecc71' if i == 0 else '#3498db' if i == 1 else '#95a5a6' for i in range(len(modelos_score))]
bars = ax.bar(modelos_score, scores_values, color=colors, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Pontuação Final', fontsize=12, fontweight='bold')
ax.set_title('🎯 Pontuação Ponderada (Votos × 100 + Hmean × 50)', fontsize=14, fontweight='bold')

# Adicionar valores nas barras
for i, (bar, score) in enumerate(zip(bars, scores_values)):
    height = bar.get_height()
    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.1f}\n{medal}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_pontuacao_ponderada.png'), dpi=150, bbox_inches='tight')
print("✅ Gráfico 4: 04_pontuacao_ponderada.png")
plt.close()

# ==========================================
# 7. RESUMO
# ==========================================

print("\n" + "="*70)
print("📊 GRÁFICOS GERADOS COM SUCESSO!")
print("="*70)
print(f"\nArquivos criados em: {output_dir}/")
print("  1. 01_ranking_votos.png - Votos por modelo")
print("  2. 02_hmean_medio.png - Hmean médio")
print("  3. 03_metricas_detalhadas.png - Comparação de todas as métricas")
print("  4. 04_pontuacao_ponderada.png - Ranking final com pontuação")
print("="*70)
