# =========================
# 1. Instalação e imports
# =========================
# !pip install openai pm4py lxml pandas networkx tabulate matplotlib

import os
import json
import pandas as pd
import xml.etree.ElementTree as ET
from openai import OpenAI
import zipfile
from IPython.display import display

import pm4py
from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.bpmn.converter import apply as bpmn_to_petri

import matplotlib.pyplot as plt
import numpy as np

# Key configuration
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "SUA_CHAVE_AQUI"))


# =========================
# 2. Helper functions
# =========================

def sbmn_to_text(sbmn_json):
    activities = [a["nome"] for a in sbmn_json["Atividades"]]
    activities_str = ", ".join(activities)

    restrictions = []
    id_to_name = {a["id"]: a["nome"] for a in sbmn_json["Atividades"]}
    for r in sbmn_json["Situacoes"]:
        left = ", ".join([id_to_name[e["id"]] for e in r["esquerda"]])
        right = ", ".join([id_to_name[d["id"]] for d in r["direita"]])
        op = r["operador"]

        if op == "DEP":
            restrictions.append(f"{right} depende de {left}")
        elif op == "DEPC":
            restrictions.append(f"{right} depende condicionalmente de {left}")
        elif op == "XOR":
            restrictions.append(f"{right} e {left} são mutuamente exclusivos")
        elif op == "UNI":
            restrictions.append(f"{right} e {left} podem ocorrer juntos ou separados (união)")
        elif op == "JMP":
            restrictions.append(f"{left} pode saltar o fluxo para {right} (jump)")
        else:
            restrictions.append(f"{left} {op} {right}")

    restrictions_str = "; ".join(restrictions)
    return f"Activities: {activities_str}. Restrictions: {restrictions_str}."


def bpmn_to_text(bpmn_xml_str):
    root = ET.fromstring(bpmn_xml_str)
    tasks = [el.attrib.get("name", el.attrib["id"]) for el in root.findall(".//{*}task")]
    flows = [(f.attrib["sourceRef"], f.attrib["targetRef"]) for f in root.findall(".//{*}sequenceFlow")]
    return f"Tarefas: {', '.join(tasks)}. Fluxos: {', '.join([f'{a}->{b}' for a,b in flows])}."


def generate_sintetic_log(bpmn_path, n_traces=200):
    bpmn_graph = bpmn_importer.apply(bpmn_path)
    net, im, fm = bpmn_to_petri(bpmn_graph)
    log = pm4py.play_out(net, im, fm, no_traces=n_traces)
    return log


def calculate_metrics(bpmn_path, log_path=None, n_traces=200):
    bpmn_graph = bpmn_importer.apply(bpmn_path)
    net, im, fm = bpmn_to_petri(bpmn_graph)

    if log_path and os.path.exists(log_path):
        log = xes_importer.apply(log_path)
        log_used = "real"
    else:
        log = generate_sintetic_log(bpmn_path, n_traces=n_traces)
        log_used = "synthetic"

    # Calcular fitness usando conformance checking
    fitness_result = pm4py.fitness_alignments(log, net, im, fm)
    # fitness_alignments pode retornar um dict com 'average_trace_fitness' ou 'log_fitness'
    if isinstance(fitness_result, dict):
        fitness = fitness_result.get('average_trace_fitness', fitness_result.get('log_fitness', fitness_result.get('averageFitness', 0.0)))
    else:
        fitness = float(fitness_result)

    # Calcular precision, generalization e simplicity
    precision_result = pm4py.precision_alignments(log, net, im, fm)
    precision = float(precision_result) if not isinstance(precision_result, dict) else precision_result.get('precision', 0.0)
    
    generalization_result = pm4py.generalization_tbr(log, net, im, fm)
    generalization = float(generalization_result) if not isinstance(generalization_result, dict) else generalization_result.get('generalization', 0.0)
    
    # Simplicity é calculada pela métrica de arco-grau
    from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
    simplicity_result = simplicity_evaluator.apply(net)
    simplicity = float(simplicity_result) if not isinstance(simplicity_result, dict) else simplicity_result.get('simplicity', 0.0)

    return {
        "fitness": fitness,
        "precision": precision,
        "generalization": generalization,
        "simplicity": simplicity,
        "log_used": log_used
    }


def parse_bpmn_structure(bpmn_path):
    tree = ET.parse(bpmn_path)
    root = tree.getroot()
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

    tasks = set([el.attrib.get("name", el.attrib["id"]) for el in root.findall(".//bpmn:task", ns)])
    flows = set([(f.attrib["sourceRef"], f.attrib["targetRef"]) for f in root.findall(".//bpmn:sequenceFlow", ns)])
    return tasks, flows


def similaridade_estrutural(bpmn_path, ref_path):
    tasks1, flows1 = parse_bpmn_structure(bpmn_path)
    tasks2, flows2 = parse_bpmn_structure(ref_path)

    sim_tasks = len(tasks1 & tasks2) / len(tasks1 | tasks2) if tasks1 | tasks2 else 0
    sim_flows = len(flows1 & flows2) / len(flows1 | flows2) if flows1 | flows2 else 0

    return (sim_tasks + sim_flows) / 2


def harmonic_mean(metrics: dict, eps=1e-6):
    vals = [max(eps, float(metrics[k])) for k in ["fitness", "precision", "generalization", "simplicity"] if k in metrics]
    n = len(vals)
    return n / sum(1.0/v for v in vals) if n > 0 else 0.0


def harmonic_mean_weighted(metrics: dict, weights: dict, eps=1e-6):
    num, den = 0.0, 0.0
    for k, w in weights.items():
        if k not in metrics or metrics[k] is None:
            continue
        m = max(eps, float(metrics[k]))
        num += w
        den += w / m
    return num / den if den > 0 else 0.0


# Standard Weights
# W = {"precision": 0.1, "fitness": 0.1, "generalization": 0.1, "simplicity": 0.1}


def evaluate_complete_models(sbmn_text, bpmn1_text, bpmn2_text, metrics1, metrics2, sim1=None, sim2=None):
    # Extrair valores das métricas para evitar problemas com f-string
    m1_fitness = metrics1['fitness']
    m1_precision = metrics1['precision']
    m1_generalization = metrics1['generalization']
    m1_simplicity = metrics1['simplicity']
    m1_hmean = harmonic_mean(metrics1)
    # m1_hmean_w = harmonic_mean_weighted(metrics1, W)
    
    m2_fitness = metrics2['fitness']
    m2_precision = metrics2['precision']
    m2_generalization = metrics2['generalization']
    m2_simplicity = metrics2['simplicity']
    m2_hmean = harmonic_mean(metrics2)
    # m2_hmean_w = harmonic_mean_weighted(metrics2, W)
    
    sim1_text = f"Similaridade estrutural com referência: {sim1:.3f}" if sim1 is not None else ""
    sim2_text = f"Similaridade estrutural com referência: {sim2:.3f}" if sim2 is not None else ""
    
    prompt = f"""
    Você é um especialista em modelagem de processos.

    Modelo de restrições (SBMN):
    {sbmn_text}

    Modelo 1 (BPMN):
    {bpmn1_text}
    Métricas: Fitness={m1_fitness:.3f}, Precision={m1_precision:.3f},
    Generalization={m1_generalization:.3f}, Simplicity={m1_simplicity:.3f},
    Hmean={m1_hmean:.3f}
    {sim1_text}

    Modelo 2 (BPMN):
    {bpmn2_text}
    Métricas: Fitness={m2_fitness:.3f}, Precision={m2_precision:.3f},
    Generalization={m2_generalization:.3f}, Simplicity={m2_simplicity:.3f},
    Hmean={m2_hmean:.3f}
    {sim2_text}

    Pergunta: considerando restrições (SBMN), métricas, médias harmônicas e, se disponível, a similaridade estrutural,
    qual dos dois modelos BPMN representa melhor o processo?
    Responda apenas com 'Modelo 1' ou 'Modelo 2' e depois explique em 2 frases.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    resposta = response.choices[0].message.content

    if "Modelo 1" in resposta:
        vencedor = "Modelo 1"
    elif "Modelo 2" in resposta:
        vencedor = "Modelo 2"
    else:
        vencedor = "Indefinido"

    return vencedor, resposta


# =========================
# 3. Folders
# =========================
PASTA_SBMN = "F:/Danielle/Mestrado/BPMNClassifierLLM/INPUTS/FOLDERS/ComputerRepair_1/DECS"
PASTA_BPMN = "F:/Danielle/Mestrado/BPMNClassifierLLM/INPUTS/FOLDERS/ComputerRepair_1"
PASTA_LOGS = "F:/Danielle/Mestrado/BPMNClassifierLLM/INPUTS/FOLDERS/ComputerRepair_1/LOGS"
PASTA_REF = "F:/Danielle/Mestrado/BPMNClassifierLLM/INPUTS/FOLDERS/ComputerRepair_1/REF"
PASTA_GRAFICOS = "F:/Danielle/Mestrado/BPMNClassifierLLM/OUTPUTS"

os.makedirs(PASTA_SBMN, exist_ok=True)
os.makedirs(PASTA_BPMN, exist_ok=True)
os.makedirs(PASTA_LOGS, exist_ok=True)
os.makedirs(PASTA_REF, exist_ok=True)
os.makedirs(PASTA_GRAFICOS, exist_ok=True)


# =========================
# 4. Main loop
# =========================
sbmn_files = [f for f in os.listdir(PASTA_SBMN) if f.lower().endswith(".json")]
bpmn_files = [f for f in os.listdir(PASTA_BPMN) if f.lower().endswith((".bpmn", ".xml"))]

resultados = []

for sbmn_file in sbmn_files:
    with open(os.path.join(PASTA_SBMN, sbmn_file), "r", encoding="utf-8") as f:
        sbmn_data = json.load(f)
        sbmn_text = sbmn_to_text(sbmn_data)

    log_ref = os.path.join(PASTA_LOGS, os.path.splitext(sbmn_file)[0] + ".xes")
    log_ref = log_ref if os.path.exists(log_ref) else None

    ref_path = os.path.join(PASTA_REF, os.path.splitext(sbmn_file)[0] + ".bpmn")
    ref_path = ref_path if os.path.exists(ref_path) else None

    for i in range(len(bpmn_files)):
        for j in range(i + 1, len(bpmn_files)):
            bpmn1_path = os.path.join(PASTA_BPMN, bpmn_files[i])
            bpmn2_path = os.path.join(PASTA_BPMN, bpmn_files[j])

            bpmn1_text = bpmn_to_text(open(bpmn1_path, "r", encoding="utf-8").read())
            bpmn2_text = bpmn_to_text(open(bpmn2_path, "r", encoding="utf-8").read())

            metrics1 = calculate_metrics(bpmn1_path, log_ref)
            metrics2 = calculate_metrics(bpmn2_path, log_ref)

            if ref_path:
                sim1 = similaridade_estrutural(bpmn1_path, ref_path)
                sim2 = similaridade_estrutural(bpmn2_path, ref_path)
            else:
                sim1, sim2 = None, None

            winner, explanation = evaluate_complete_models(
                sbmn_text, bpmn1_text, bpmn2_text, metrics1, metrics2, sim1, sim2
            )

            resultados.append({
                "SBMN_Arquivo": sbmn_file,
                "BPMN1_Arquivo": bpmn_files[i],
                "BPMN2_Arquivo": bpmn_files[j],
                "LLM_Voto": winner,
                "Explicacao_LLM": explanation,
                "M1_Fitness": metrics1["fitness"],
                "M1_Precision": metrics1["precision"],
                "M1_Generalization": metrics1["generalization"],
                "M1_Simplicity": metrics1["simplicity"],
                "M1_Hmean": harmonic_mean(metrics1),
                # "M1_Hmean_weighted": harmonic_mean_weighted(metrics1, W),
                "M1_SimEstrutural": sim1,
                "M2_Fitness": metrics2["fitness"],
                "M2_Precision": metrics2["precision"],
                "M2_Generalization": metrics2["generalization"],
                "M2_Simplicity": metrics2["simplicity"],
                "M2_Hmean": harmonic_mean(metrics2),
                # "M2_Hmean_weighted": harmonic_mean_weighted(metrics2, W),
                "M2_SimEstrutural": sim2,
            })

df = pd.DataFrame(resultados)

# =========================
# 5. Exportar resultados
# =========================
df.to_csv("resultados_comparacao_final.csv", index=False)


# =========================
# 6. Exibir tabela estilizada
# =========================
def highlight_winner(row):
    color = [""] * len(row)
    if row["LLM_Voto"] == "Modelo 1":
        idxs = [df.columns.get_loc(c) for c in row.index if c.startswith("M1_")]
        for idx in idxs:
            color[idx] = "background-color: #b3ffb3"
    elif row["LLM_Voto"] == "Modelo 2":
        idxs = [df.columns.get_loc(c) for c in row.index if c.startswith("M2_")]
        for idx in idxs:
            color[idx] = "background-color: #b3ffb3"
    return color

styled = df.style.format(precision=3).apply(highlight_winner, axis=1)
display(styled)


# =========================
# 7. Gráficos comparativos
# =========================
def plot_comparacao(row, save_dir=None):
    # labels = ["Fitness", "Precision", "Generalization", "Simplicity", "Hmean", "Hmean_w"]
    # m1_vals = [row[f"M1_{m}"] for m in ["Fitness", "Precision", "Generalization", "Simplicity", "Hmean", "Hmean_weighted"]]
    # m2_vals = [row[f"M2_{m}"] for m in ["Fitness", "Precision", "Generalization", "Simplicity", "Hmean", "Hmean_weighted"]]

    labels = ["Fitness", "Precision", "Generalization", "Simplicity", "Hmean"]
    m1_vals = [row[f"M1_{m}"] for m in ["Fitness", "Precision", "Generalization", "Simplicity", "Hmean"]]
    m2_vals = [row[f"M2_{m}"] for m in ["Fitness", "Precision", "Generalization", "Simplicity", "Hmean"]]


    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, m1_vals, width, label=row["BPMN1_Arquivo"])
    bars2 = ax.bar(x + width/2, m2_vals, width, label=row["BPMN2_Arquivo"])

    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{row['SBMN_Arquivo']} | Voto LLM: {row['LLM_Voto']}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    if save_dir:
        filename = f"{row['SBMN_Arquivo']}_{row['BPMN1_Arquivo']}_vs_{row['BPMN2_Arquivo']}.png".replace(" ", "_")
        plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
    plt.show()


# Gerar gráficos e salvar
for _, row in df.iterrows():
    plot_comparacao(row, save_dir=PASTA_GRAFICOS)

print(f"Gráficos salvos em {PASTA_GRAFICOS}")