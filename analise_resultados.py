#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise e visualização dos resultados de classificação
Gera tabelas com pontuação e estatísticas dos modelos
"""

import pandas as pd
import os
from collections import defaultdict

# ==========================================
# 1. CARREGAR DADOS
# ==========================================

csv_files = [f for f in os.listdir('.') if f.startswith('resultados_comparacao_') and f.endswith('.csv')]

if not csv_files:
    print("❌ Nenhum arquivo de resultados encontrado!")
    print("   Execute primeiro: python llmbpmnsbmn_classifier_enhanced.py")
    exit(1)

# Usar o arquivo mais recente
csv_file = max(csv_files, key=lambda x: os.path.getctime(x))
print(f"📊 Analisando: {csv_file}\n")

df = pd.read_csv(csv_file)

# ==========================================
# Extrair nome do processo
# ==========================================

# Tentar extrair nome do processo do SBMN_Arquivo
process_name = None
if 'SBMN_Arquivo' in df.columns and len(df) > 0:
    sbmn_file = df.iloc[0]['SBMN_Arquivo']
    # Tentar extrair do padrão: DECS_ProcessName.json
    if '_' in sbmn_file:
        parts = sbmn_file.split('_')
        if len(parts) >= 2:
            process_name = '_'.join(parts[1:]).replace('.json', '')

if not process_name:
    process_name = "results"

output_dir = os.path.join('OUTPUTS', process_name)
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Salvando resultados em: {output_dir}\n")

# ==========================================
# 2. CONTAR VOTOS POR MODELO
# ==========================================

print("="*70)
print("🏆 RANKING GERAL - VOTOS POR MODELO")
print("="*70)

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

# Ordenar por votos
votos_sorted = sorted(votos.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'Posição':<10} {'Arquivo':<40} {'Votos':<10} {'%':<10}")
print("-"*70)

total_votos = sum(votos.values())
for idx, (arquivo, votos_count) in enumerate(votos_sorted, 1):
    percentual = (votos_count / total_votos * 100) if total_votos > 0 else 0
    medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
    print(f"{medal} #{idx:<7} {arquivo:<40} {votos_count:<10} {percentual:>6.1f}%")

# ==========================================
# 3. TABELA DE COMPARAÇÃO COM MÉTRICAS
# ==========================================

print(f"\n{'='*70}")
print("📈 DETALHAMENTO DAS COMPARAÇÕES")
print("="*70 + "\n")

# Criar tabela detalhada
comparacoes = []
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
    
    comparacoes.append({
        'Modelo 1': m1_file,
        'M1_Hmean': m1_hmean,
        'M1_Fitness': m1_fitness,
        'M1_Precision': m1_precision,
        'vs': 'vs',
        'Modelo 2': m2_file,
        'M2_Hmean': m2_hmean,
        'M2_Fitness': m2_fitness,
        'M2_Precision': m2_precision,
        'Vencedor': vencedor
    })

df_comparacoes = pd.DataFrame(comparacoes)

# Exibir com formatação
for idx, row in df_comparacoes.iterrows():
    print(f"Comparação #{idx+1}")
    print(f"  {row['Modelo 1']:<40} {row['M1_Hmean']:.3f}  |  {row['Modelo 2']:<40} {row['M2_Hmean']:.3f}")
    print(f"    Fitness:  {row['M1_Fitness']:.3f}              |    Fitness:  {row['M2_Fitness']:.3f}")
    print(f"    Precision: {row['M1_Precision']:.3f}             |    Precision: {row['M2_Precision']:.3f}")
    
    if row['Vencedor'] == "Modelo 1":
        vencedor_nome = row['Modelo 1']
        marca = "✅"
    elif row['Vencedor'] == "Modelo 2":
        vencedor_nome = row['Modelo 2']
        marca = "✅"
    else:
        vencedor_nome = "Empate"
        marca = "⚖️"
    
    print(f"  {marca} Vencedor: {vencedor_nome}\n")

# ==========================================
# 4. PONTUAÇÃO PONDERADA
# ==========================================

print(f"\n{'='*70}")
print("🎯 PONTUAÇÃO PONDERADA (Votos + Métricas)")
print("="*70 + "\n")

# Calcular score ponderado
scores = defaultdict(lambda: {'votos': 0, 'hmean_media': 0, 'count': 0})

for idx, row in df.iterrows():
    vencedor = row['LLM_Voto']
    modelo1 = row['BPMN1_Arquivo']
    modelo2 = row['BPMN2_Arquivo']
    
    hmean1 = row.get('M1_Hmean', 0)
    hmean2 = row.get('M2_Hmean', 0)
    
    # Contar votos
    if vencedor == "Modelo 1":
        scores[modelo1]['votos'] += 1
        scores[modelo1]['hmean_media'] += hmean1
        scores[modelo1]['count'] += 1
    elif vencedor == "Modelo 2":
        scores[modelo2]['votos'] += 1
        scores[modelo2]['hmean_media'] += hmean2
        scores[modelo2]['count'] += 1

# Calcular média de hmean
for model in scores:
    if scores[model]['count'] > 0:
        scores[model]['hmean_media'] /= scores[model]['count']

# Calcular pontuação final: (votos * 100) + (hmean_media * 50)
score_final = {}
for model, data in scores.items():
    score = (data['votos'] * 100) + (data['hmean_media'] * 50)
    score_final[model] = {
        'votos': data['votos'],
        'hmean': data['hmean_media'],
        'score': score
    }

# Ordenar por score
score_sorted = sorted(score_final.items(), key=lambda x: x[1]['score'], reverse=True)

print(f"{'Posição':<10} {'Arquivo':<40} {'Votos':<8} {'Hmean':<8} {'Score':<10}")
print("-"*70)

for idx, (arquivo, data) in enumerate(score_sorted, 1):
    medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
    print(f"{medal} #{idx:<7} {arquivo:<40} {data['votos']:<8} {data['hmean']:<8.3f} {data['score']:<10.1f}")

# ==========================================
# 5. SALVAR RELATÓRIO
# ==========================================

print(f"\n{'='*70}")
print("💾 SALVANDO RELATÓRIO")
print("="*70 + "\n")

# Criar DataFrame para exportar
relatorio_data = []
for arquivo, data in score_sorted:
    relatorio_data.append({
        'Posição': len(relatorio_data) + 1,
        'Arquivo': arquivo,
        'Votos': data['votos'],
        'Hmean_Médio': f"{data['hmean']:.3f}",
        'Pontuação_Final': f"{data['score']:.1f}"
    })

df_relatorio = pd.DataFrame(relatorio_data)
ranking_file = os.path.join(output_dir, 'ranking_modelos.csv')
df_relatorio.to_csv(ranking_file, index=False)

print(f"✅ Ranking salvo em: {ranking_file}\n")

# ==========================================
# 6. RESUMO FINAL
# ==========================================

print(f"{'='*70}")
print("📊 RESUMO FINAL")
print("="*70)

if votos_sorted:
    campeao = votos_sorted[0][0]
    print(f"\n🏆 CAMPEÃO: {campeao}")
    print(f"   Votos: {votos_sorted[0][1]}")
    if score_sorted:
        print(f"   Score Final: {score_final[campeao]['score']:.1f}")
    
    print(f"\nTotal de comparações: {len(df)}")
    print(f"Total de votos: {total_votos}")

print("=" * 70)
