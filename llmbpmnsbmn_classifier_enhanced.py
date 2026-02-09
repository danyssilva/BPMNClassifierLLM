# =========================
# BPMN Classifier with Enhanced LLM
# Versão melhorada com técnicas avançadas de prompting + MULTIPROCESSAMENTO
# =========================
# !pip install openai pm4py lxml pandas networkx tabulate matplotlib jinja2

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

# MULTIPROCESSAMENTO
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import time
from functools import partial

# Key configuration
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "SUA_CHAVE_AQUI"))

# CONFIGURAÇÃO DE MULTIPROCESSAMENTO
# [WARNING] VALORES REDUZIDOS PARA EVITAR TRAVAMENTO DA MÁQUINA
# Se máquina não travar em 5 min, pode aumentar gradualmente:
# NUM_WORKERS = 2 → 3 → 4 (com pausas entre lotes)
# MAX_WORKERS_API = 1 → 2 (com cuidado com rate limits)
NUM_WORKERS = 5             # Reduzido de cpu_count()-1 para segurança
MAX_WORKERS_API = 2          # Reduzido de 3 para evitar sobrecarga
CHECKPOINT_INTERVAL = 10     # Salvar progresso a cada N arquivos
print(f"[CONFIG] Configuração SEGURA: {NUM_WORKERS} workers de processamento, {MAX_WORKERS_API} workers para API")


# =========================
# SISTEMA DE CHECKPOINT - Recuperação de falhas e continuação
# =========================

class CheckpointManager:
    """Gerencia o salvamento e recuperação de progresso para evitar reprocessamento"""
    
    def __init__(self, checkpoint_dir="CHECKPOINTS", process_name="default"):
        self.checkpoint_dir = checkpoint_dir
        self.process_name = process_name
        self.process_checkpoint_dir = os.path.join(checkpoint_dir, process_name)
        os.makedirs(self.process_checkpoint_dir, exist_ok=True)
        
        self.sbmn_processed_file = os.path.join(self.process_checkpoint_dir, "sbmn_processados.json")
        self.bpmn_texts_cache_file = os.path.join(self.process_checkpoint_dir, "cache_textos_bpmn.json")
        self.metrics_cache_file = os.path.join(self.process_checkpoint_dir, "cache_metricas.json")
        self.comparisons_done_file = os.path.join(self.process_checkpoint_dir, "comparacoes_realizadas.json")
        self.results_file = os.path.join(self.process_checkpoint_dir, "resultados_temporarios.json")
        
        # Carregar dados existentes
        self.sbmn_processados = self._load_json(self.sbmn_processed_file, {})
        self.bpmn_texts_cache = self._load_json(self.bpmn_texts_cache_file, {})
        self.metrics_cache = self._load_json(self.metrics_cache_file, {})
        self.comparisons_done = self._load_json(self.comparisons_done_file, {})
        self.resultados = self._load_json(self.results_file, [])
        
        print(f"[OK] CheckpointManager inicializado para '{process_name}'")
        print(f"  - SBMNs já processados: {len(self.sbmn_processados)}")
        print(f"  - Textos BPMN em cache: {len(self.bpmn_texts_cache)}")
        print(f"  - Métricas em cache: {len(self.metrics_cache)}")
        print(f"  - Comparações realizadas: {len(self.comparisons_done)}")
        print(f"  - Resultados salvos: {len(self.resultados)}")
    
    def _load_json(self, filepath, default=None):
        """Carrega JSON de forma segura"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"  [ERROR] Erro ao carregar {filepath}: {e}")
        return default if default is not None else {}
    
    def _save_json(self, filepath, data):
        """Salva JSON de forma segura"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [ERROR] Erro ao salvar {filepath}: {e}")
    
    def mark_sbmn_processed(self, sbmn_file, sbmn_text_preview):
        """Marca um SBMN como processado"""
        self.sbmn_processados[sbmn_file] = {
            "timestamp": time.time(),
            "preview": sbmn_text_preview[:100] if isinstance(sbmn_text_preview, str) else str(sbmn_text_preview)[:100]
        }
        self._save_json(self.sbmn_processed_file, self.sbmn_processados)
    
    def is_sbmn_processed(self, sbmn_file):
        """Verifica se um SBMN já foi processado"""
        return sbmn_file in self.sbmn_processados
    
    def cache_bpmn_text(self, bpmn_file, text):
        """Cacheia o texto convertido de um BPMN"""
        self.bpmn_texts_cache[bpmn_file] = text
        self._save_json(self.bpmn_texts_cache_file, self.bpmn_texts_cache)
    
    def get_cached_bpmn_text(self, bpmn_file):
        """Recupera texto em cache se existir"""
        return self.bpmn_texts_cache.get(bpmn_file)
    
    def cache_metrics(self, bpmn_file, metrics):
        """Cacheia as métricas de um BPMN"""
        self.metrics_cache[bpmn_file] = metrics
        self._save_json(self.metrics_cache_file, self.metrics_cache)
    
    def get_cached_metrics(self, bpmn_file):
        """Recupera métricas em cache se existirem"""
        return self.metrics_cache.get(bpmn_file)
    
    def mark_comparison_done(self, sbmn_file, bpmn1, bpmn2, method):
        """Marca uma comparação como realizada"""
        key = f"{sbmn_file}|{bpmn1}|{bpmn2}|{method}"
        self.comparisons_done[key] = {"timestamp": time.time()}
        self._save_json(self.comparisons_done_file, self.comparisons_done)
    
    def is_comparison_done(self, sbmn_file, bpmn1, bpmn2, method):
        """Verifica se uma comparação já foi realizada"""
        key = f"{sbmn_file}|{bpmn1}|{bpmn2}|{method}"
        return key in self.comparisons_done
    
    def add_result(self, result):
        """Adiciona um resultado à lista temporária"""
        if result:
            self.resultados.append(result)
            # Salvar a cada resultado para garantir persistência
            if len(self.resultados) % max(1, CHECKPOINT_INTERVAL) == 0:
                self._save_json(self.results_file, self.resultados)
    
    def get_all_results(self):
        """Retorna todos os resultados acumulados"""
        return self.resultados
    
    def clear_results(self):
        """Limpa os resultados (após exportar para CSV)"""
        self.resultados = []
        self._save_json(self.results_file, [])
    
    def cleanup(self):
        """Remove arquivos de checkpoint após conclusão bem-sucedida"""
        try:
            import shutil
            if os.path.exists(self.process_checkpoint_dir):
                shutil.rmtree(self.process_checkpoint_dir)
                print(f"  [OK] Checkpoint '{self.process_name}' limpo com sucesso")
        except Exception as e:
            print(f"  [ERROR] Erro ao limpar checkpoint: {e}")


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
    # Com fallback para modelos "not easy sound net"
    fitness = 0.5  # valor padrão
    try:
        fitness_result = pm4py.fitness_alignments(log, net, im, fm)
        if isinstance(fitness_result, dict):
            fitness = fitness_result.get('average_trace_fitness', fitness_result.get('log_fitness', fitness_result.get('averageFitness', 0.5)))
        else:
            fitness = float(fitness_result)
    except Exception as e:
        # Fallback: usar token-based fitness (mais robusto para redes complexas)
        try:
            fitness_result = pm4py.fitness_token_based_replay(log, net, im, fm)
            if isinstance(fitness_result, dict):
                fitness = fitness_result.get('average_trace_fitness', fitness_result.get('log_fitness', 0.5))
            else:
                fitness = float(fitness_result)
        except Exception as e2:
            print(f"      [WARNING] Fitness não calculável (rede complexa): usando 0.5")
            fitness = 0.5

    # Calcular precision, generalization e simplicity
    precision = 0.5
    try:
        precision_result = pm4py.precision_alignments(log, net, im, fm)
        precision = float(precision_result) if not isinstance(precision_result, dict) else precision_result.get('precision', 0.5)
    except Exception as e:
        try:
            precision_result = pm4py.precision_token_based_replay(log, net, im, fm)
            precision = float(precision_result) if not isinstance(precision_result, dict) else precision_result.get('precision', 0.5)
        except:
            precision = 0.5
    
    generalization = 0.5
    try:
        generalization_result = pm4py.generalization_tbr(log, net, im, fm)
        generalization = float(generalization_result) if not isinstance(generalization_result, dict) else generalization_result.get('generalization', 0.5)
    except Exception as e:
        generalization = 0.5
    
    simplicity = 0.5
    try:
        from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
        simplicity_result = simplicity_evaluator.apply(net)
        simplicity = float(simplicity_result) if not isinstance(simplicity_result, dict) else simplicity_result.get('simplicity', 0.5)
    except Exception as e:
        simplicity = 0.5

    # Normalizar valores para [0, 1]
    fitness = max(0, min(1, float(fitness)))
    precision = max(0, min(1, float(precision)))
    generalization = max(0, min(1, float(generalization)))
    simplicity = max(0, min(1, float(simplicity)))

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


# =========================
# FUNÇÕES PARA MULTIPROCESSAMENTO
# =========================

def _process_metric_calculation(args):
    """Wrapper para calcular métricas de um arquivo BPMN (executável em paralelo)"""
    bpmn_path, log_ref, n_traces = args
    try:
        return bpmn_path, calculate_metrics(bpmn_path, log_ref, n_traces)
    except Exception as e:
        print(f"Erro ao calcular métricas para {bpmn_path}: {e}")
        return bpmn_path, None


def _process_similarity_calculation(args):
    """Wrapper para calcular similaridade estrutural (executável em paralelo)"""
    bpmn_path, ref_path = args
    try:
        if ref_path and os.path.exists(ref_path):
            return bpmn_path, similaridade_estrutural(bpmn_path, ref_path)
        return bpmn_path, None
    except Exception as e:
        print(f"Erro ao calcular similaridade para {bpmn_path}: {e}")
        return bpmn_path, None


def _process_single_comparison(args):
    """Wrapper para processar uma única comparação BPMN (executável em paralelo)"""
    (sbmn_text, bpmn1_data, bpmn2_data, metrics1, metrics2, sim1, sim2, 
     sbmn_file, bpmn1_file, bpmn2_file, method) = args
    
    try:
        winner, explanation = evaluate_complete_models(
            sbmn_text, bpmn1_data["text"], bpmn2_data["text"], 
            metrics1, metrics2, sim1, sim2, method=method
        )
        
        result = {
            "SBMN_Arquivo": sbmn_file,
            "BPMN1_Arquivo": bpmn1_file,
            "BPMN2_Arquivo": bpmn2_file,
            "Metodo_Avaliacao": method,
            "LLM_Voto": winner,
            "Explicacao_LLM": explanation,
            "M1_Fitness": metrics1["fitness"],
            "M1_Precision": metrics1["precision"],
            "M1_Generalization": metrics1["generalization"],
            "M1_Simplicity": metrics1["simplicity"],
            "M1_Hmean": harmonic_mean(metrics1),
            "M1_SimEstrutural": sim1,
            "M2_Fitness": metrics2["fitness"],
            "M2_Precision": metrics2["precision"],
            "M2_Generalization": metrics2["generalization"],
            "M2_Simplicity": metrics2["simplicity"],
            "M2_Hmean": harmonic_mean(metrics2),
            "M2_SimEstrutural": sim2,
        }
        return result
    except Exception as e:
        print(f"Erro na comparação {bpmn1_file} vs {bpmn2_file}: {e}")
        return None


def plot_comparacao(row, save_dir=None, method=""):
    """Gera gráfico de comparação entre dois modelos"""
    fig = None
    try:
        # Configurar matplotlib para não usar GUI
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['figure.max_open_warning'] = 0  # Desabilitar aviso de figuras
        
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
            # Criar nome de arquivo com tamanho limitado (evitar erro de caminho muito longo)
            import hashlib
            sbmn_short = row['SBMN_Arquivo'][:20].replace(".json", "")
            bpmn1_short = row['BPMN1_Arquivo'][:20].replace(".bpmn", "")
            bpmn2_short = row['BPMN2_Arquivo'][:20].replace(".bpmn", "")
            
            # Usar hash para garantir unicidade
            unique_id = hashlib.md5(f"{row['SBMN_Arquivo']}{row['BPMN1_Arquivo']}{row['BPMN2_Arquivo']}".encode()).hexdigest()[:8]
            filename = f"comp_{sbmn_short}_vs_{bpmn1_short}_{bpmn2_short}_{unique_id}.png".replace(" ", "_")
            
            try:
                plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight", dpi=100)
                return True
            except Exception as save_error:
                print(f"  [ERROR] Erro ao salvar {filename[:50]}: {str(save_error)[:50]}")
                return False
        else:
            return True
    except Exception as e:
        print(f"  [ERROR] Erro ao gerar gráfico: {str(e)[:100]}")
        return False
    finally:
        # Garantir fechamento da figura
        if fig is not None:
            try:
                plt.close(fig)
            except:
                pass
        # Limpar cache de figuras periodicamente
        if plt.get_fignums():
            plt.close('all')


def _read_single_bpmn_file(args):
    """Wrapper para ler e converter um arquivo BPMN (executável em paralelo)"""
    bpmn_path, bpmn_file = args
    try:
        with open(bpmn_path, "r", encoding="utf-8") as f:
            content = f.read()
        bpmn_text = bpmn_to_text(content)
        return bpmn_file, bpmn_text, None
    except Exception as e:
        print(f"        Erro ao ler {bpmn_file}: {e}")
        return bpmn_file, None, str(e)


def load_bpmn_files_parallel(bpmn_files, bpmn_base_path, checkpoint=None, max_workers=None):
    """Carrega e converte todos os arquivos BPMN em paralelo"""
    if max_workers is None:
        max_workers = min(NUM_WORKERS, 8)  # Máximo 8 para I/O
    
    print(f"     Carregando {len(bpmn_files)} arquivos BPMN em paralelo ({max_workers} workers)...")
    
    bpmn_path_dict = {}
    cached_count = 0
    files_to_load = []
    
    # Verificar cache primeiro
    for bpmn_file in bpmn_files:
        if checkpoint:
            cached_text = checkpoint.get_cached_bpmn_text(bpmn_file)
            if cached_text:
                bpmn_path_dict[bpmn_file] = cached_text
                cached_count += 1
                continue
        
        bpmn_path = os.path.join(bpmn_base_path, bpmn_file)
        files_to_load.append((bpmn_path, bpmn_file))
    
    if cached_count > 0:
        print(f"       [OK] {cached_count} arquivos recuperados do cache")
    
    # Carregar arquivos não-cacheados em paralelo
    if files_to_load:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_read_single_bpmn_file, task): task[1] 
                      for task in files_to_load}
            
            loaded = 0
            errors = 0
            for future in as_completed(futures):
                bpmn_file, bpmn_text, error = future.result()
                if bpmn_text:
                    bpmn_path_dict[bpmn_file] = bpmn_text
                    # Cachear novo
                    if checkpoint:
                        checkpoint.cache_bpmn_text(bpmn_file, bpmn_text)
                    loaded += 1
                else:
                    errors += 1
                
                # Mostrar progresso
                if (loaded + errors) % max(1, len(files_to_load) // 4) == 0:
                    print(f"       Progresso: {loaded + errors}/{len(files_to_load)} arquivos carregados")
        
        print(f"       [OK] {loaded} arquivos carregados com sucesso")
        if errors > 0:
            print(f"       [WARNING] {errors} erros durante carregamento")
    
    return bpmn_path_dict


# =========================
# VERSÃO OTIMIZADA DAS FUNÇÕES PRINCIPAIS
# =========================

def _process_plot_generation(args):
    """Wrapper para gerar gráfico (executável em paralelo)"""
    row_tuple, save_dir, method = args
    try:
        # Reconstruir a série do índice passado
        row = pd.Series(dict(row_tuple))
        success = plot_comparacao(row, save_dir, method)
        return "generated", success
    except Exception as e:
        print(f"[ERROR] Erro ao gerar gráfico: {str(e)[:80]}")
        return None, False

def calculate_all_metrics_parallel(bpmn_files, bpmn_base_path, log_ref, n_traces=200, checkpoint=None):
    """Calcula métricas para múltiplos BPMNs em paralelo"""
    print(f"  Calculando métricas para {len(bpmn_files)} arquivos em paralelo...")
    
    tasks = []
    files_to_process = []
    cached_count = 0
    
    # [OK] CHECKPOINT: Verificar cache de métricas
    for f in bpmn_files:
        if checkpoint:
            cached_metrics = checkpoint.get_cached_metrics(f)
            if cached_metrics:
                tasks.append((f, cached_metrics))
                cached_count += 1
                continue
        
        files_to_process.append(f)
        tasks.append((os.path.join(bpmn_base_path, f), log_ref, n_traces))
    
    if cached_count > 0:
        print(f"    [OK] {cached_count} métricas recuperadas do cache")
    
    metrics_dict = {}
    
    # Adicionar métricas em cache
    for f, metrics in [(t[0], t[1]) for t in tasks if isinstance(t[1], dict)]:
        metrics_dict[f] = metrics
    
    # Processar apenas os não cacheados
    if files_to_process:
        processing_tasks = [(os.path.join(bpmn_base_path, f), log_ref, n_traces) for f in files_to_process]
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(_process_metric_calculation, task): f for task, f in zip(processing_tasks, files_to_process)}
            completed = 0
            
            for future in as_completed(futures):
                bpmn_path, metrics = future.result()
                bpmn_file = futures[future]
                if metrics:
                    metrics_dict[bpmn_file] = metrics
                    # [OK] CHECKPOINT: Cachear métrica
                    if checkpoint:
                        checkpoint.cache_metrics(bpmn_file, metrics)
                
                completed += 1
                if completed % max(1, len(files_to_process) // 5) == 0:
                    print(f"    Progresso: {completed}/{len(files_to_process)} arquivos processados")
    
    return metrics_dict


def calculate_all_similarities_parallel(bpmn_files, bpmn_base_path, ref_path):
    """Calcula similaridades estruturais em paralelo"""
    print(f"  Calculando similaridades para {len(bpmn_files)} arquivos...")
    
    tasks = [(os.path.join(bpmn_base_path, f), ref_path) for f in bpmn_files]
    similarity_dict = {}
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_process_similarity_calculation, task): f for task, f in zip(tasks, bpmn_files)}
        
        for future in as_completed(futures):
            bpmn_path, sim = future.result()
            bpmn_file = futures[future]
            if sim is not None:
                similarity_dict[bpmn_file] = sim
    
    return similarity_dict


def process_all_comparisons_parallel(sbmn_text, bpmn_files, bpmn_path_dict, metrics_dict, 
                                     similarity_dict, sbmn_file, ref_path, method, checkpoint=None):
    """Processa todas as comparações pairwise em paralelo com throttling da API"""
    print(f"  Processando comparações em paralelo...")
    
    # Preparar tarefas
    tasks = []
    task_list = []
    
    skipped_comparisons = 0
    
    for i in range(len(bpmn_files)):
        for j in range(i + 1, len(bpmn_files)):
            bpmn1_file = bpmn_files[i]
            bpmn2_file = bpmn_files[j]
            
            # [OK] CHECKPOINT: Verificar se já foi feita essa comparação
            if checkpoint and checkpoint.is_comparison_done(sbmn_file, bpmn1_file, bpmn2_file, method):
                skipped_comparisons += 1
                print(f"    [SKIP] Comparação já realizada (skip): {bpmn1_file} vs {bpmn2_file}")
                continue
            
            metrics1 = metrics_dict.get(bpmn1_file)
            metrics2 = metrics_dict.get(bpmn2_file)
            
            if metrics1 is None or metrics2 is None:
                print(f"    Pulando comparação {bpmn1_file} vs {bpmn2_file} (métricas não disponíveis)")
                continue
            
            sim1 = similarity_dict.get(bpmn1_file, None)
            sim2 = similarity_dict.get(bpmn2_file, None)
            
            bpmn1_text = bpmn_path_dict[bpmn1_file]
            bpmn2_text = bpmn_path_dict[bpmn2_file]
            
            task = (sbmn_text, 
                   {"text": bpmn1_text, "path": bpmn_files[i]}, 
                   {"text": bpmn2_text, "path": bpmn_files[j]},
                   metrics1, metrics2, sim1, sim2,
                   sbmn_file, bpmn1_file, bpmn2_file, method)
            
            tasks.append(task)
            task_list.append((bpmn1_file, bpmn2_file))
    
    if skipped_comparisons > 0:
        print(f"  [OK] {skipped_comparisons} comparações puladas (já realizadas)")
    
    results = []
    
    # Usar ThreadPoolExecutor para API (melhor que ProcessPoolExecutor para I/O)
    # Limitar a MAX_WORKERS_API para respeitar rate limits do OpenAI
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_API) as executor:
        futures = {executor.submit(_process_single_comparison, task): (bpmn1, bpmn2) 
                  for task, (bpmn1, bpmn2) in zip(tasks, task_list)}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
                # [OK] CHECKPOINT: Marcar comparação como realizada
                if checkpoint:
                    sbmn_f = result['SBMN_Arquivo']
                    bpmn1_f = result['BPMN1_Arquivo']
                    bpmn2_f = result['BPMN2_Arquivo']
                    method_used = result['Metodo_Avaliacao']
                    checkpoint.mark_comparison_done(sbmn_f, bpmn1_f, bpmn2_f, method_used)
                    checkpoint.add_result(result)
            
            completed += 1
            bpmn_pair = futures[future]
            print(f"    Comparação {completed}/{len(tasks)}: {bpmn_pair[0]} vs {bpmn_pair[1]}")
            
            # Pequeno delay entre requisições à API
            time.sleep(0.1)
    
    return results


def generate_all_plots_parallel(df, save_dir, method):
    """Gera todos os gráficos em paralelo com matplotlib"""
    print(f"Gerando {len(df)} gráficos em paralelo...")
    
    tasks = []
    for idx, row in df.iterrows():
        # Converter série para tuple de items para passar entre processos
        row_tuple = tuple(row.items())
        tasks.append((row_tuple, save_dir, method))
    
    # Usar max 2 workers para matplotlib (não é thread-safe com muitos workers paralelos)
    max_plot_workers = min(2, NUM_WORKERS)
    
    with ThreadPoolExecutor(max_workers=max_plot_workers) as executor:
        futures = [executor.submit(_process_plot_generation, task) for task in tasks]
        
        completed = 0
        failed = 0
        for idx, future in enumerate(as_completed(futures), 1):
            try:
                filename, success = future.result(timeout=30)
                if success:
                    completed += 1
                else:
                    failed += 1
                
                # Mostrar progresso a cada 10% concluído
                if idx % max(1, len(df) // 10) == 0:
                    print(f"    Progresso: {idx}/{len(df)} gráficos processados")
            except Exception as e:
                print(f"  [ERROR] Erro ao processar gráfico: {str(e)[:100]}")
                failed += 1
        
        print(f"  [OK] {completed}/{len(df)} gráficos gerados com sucesso")
        if failed > 0:
            print(f"  [WARNING] {failed} gráficos falharam")
    
    return completed


# =========================
# ENHANCED EVALUATION METHODS
# =========================

# Método 1: BASELINE (original)
def evaluate_baseline(sbmn_text, bpmn1_text, bpmn2_text, metrics1, metrics2, sim1=None, sim2=None):
    """Método original - baseline para comparação"""
    m1_fitness = metrics1['fitness']
    m1_precision = metrics1['precision']
    m1_generalization = metrics1['generalization']
    m1_simplicity = metrics1['simplicity']
    m1_hmean = harmonic_mean(metrics1)
    
    m2_fitness = metrics2['fitness']
    m2_precision = metrics2['precision']
    m2_generalization = metrics2['generalization']
    m2_simplicity = metrics2['simplicity']
    m2_hmean = harmonic_mean(metrics2)
    
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
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    resposta = response.choices[0].message.content
    
    if "Modelo 1" in resposta:
        vencedor = "Modelo 1"
    elif "Modelo 2" in resposta:
        vencedor = "Modelo 2"
    else:
        vencedor = "Indefinido"
    
    return vencedor, resposta


# Método 2: EXPLICIT CRITERIA (Recomendado!)
def evaluate_explicit_criteria(sbmn_text, bpmn1_text, bpmn2_text, metrics1, metrics2, sim1=None, sim2=None):
    """Método com critérios explícitos e prioridades definidas"""
    
    m1_fitness = metrics1['fitness']
    m1_precision = metrics1['precision']
    m1_generalization = metrics1['generalization']
    m1_simplicity = metrics1['simplicity']
    m1_hmean = harmonic_mean(metrics1)
    
    m2_fitness = metrics2['fitness']
    m2_precision = metrics2['precision']
    m2_generalization = metrics2['generalization']
    m2_simplicity = metrics2['simplicity']
    m2_hmean = harmonic_mean(metrics2)
    
    sim1_text = f"\n- Similaridade estrutural com referência: {sim1:.3f}" if sim1 is not None else ""
    sim2_text = f"\n- Similaridade estrutural com referência: {sim2:.3f}" if sim2 is not None else ""
    
    prompt = f"""Você é um classificador especializado em modelos BPMN baseado em restrições de processos e métricas de qualidade.

## CRITÉRIOS DE AVALIAÇÃO (em ordem de prioridade):

1. **CONFORMIDADE COM RESTRIÇÕES** (Crítico - Eliminatório)
   - O modelo DEVE respeitar TODAS as restrições estruturais
   - Dependências: A ordem de execução está correta?
   - Exclusões mútuas (XOR): Implementadas corretamente?
   - Se um modelo viola restrições, ele deve ser desconsiderado

2. **FITNESS** (Peso: 40% - Mais Importante)
   - Capacidade de reproduzir o comportamento observado no log
   - Ideal: ≥ 0.90 | Bom: ≥ 0.80 | Aceitável: ≥ 0.70
   - Baixo fitness indica que o modelo não representa bem o processo real

3. **PRECISION** (Peso: 30%)
   - Controle de comportamento extra permitido pelo modelo
   - Ideal: ≥ 0.85 | Bom: ≥ 0.75 | Aceitável: ≥ 0.65
   - Baixa precision indica que o modelo permite execuções inválidas

4. **GENERALIZATION** (Peso: 20%)
   - Capacidade de generalizar além dos casos observados
   - Ideal: ≥ 0.75 | Bom: ≥ 0.65 | Aceitável: ≥ 0.55
   - Evita overfitting ao log de eventos

5. **SIMPLICITY** (Peso: 10%)
   - Simplicidade estrutural do modelo
   - Ideal: ≥ 0.70 | Bom: ≥ 0.60
   - Use como desempate quando outras métricas são similares

## DADOS DO PROCESSO:

### Restrições do Processo (SBMN):
{sbmn_text}

### Modelo 1:
{bpmn1_text}

**Métricas do Modelo 1:**
- Fitness: {m1_fitness:.3f}
- Precision: {m1_precision:.3f}
- Generalization: {m1_generalization:.3f}
- Simplicity: {m1_simplicity:.3f}
- Média Harmônica: {m1_hmean:.3f}{sim1_text}

### Modelo 2:
{bpmn2_text}

**Métricas do Modelo 2:**
- Fitness: {m2_fitness:.3f}
- Precision: {m2_precision:.3f}
- Generalization: {m2_generalization:.3f}
- Simplicity: {m2_simplicity:.3f}
- Média Harmônica: {m2_hmean:.3f}{sim2_text}

## PROCESSO DE DECISÃO:

1. Verifique se ambos os modelos respeitam as restrições estruturais do SBMN
2. Compare FITNESS (métrica mais importante)
3. Se fitness similar (diferença < 0.10), compare PRECISION
4. Considere a média harmônica como visão geral
5. Use similaridade estrutural (se disponível) como complemento

**RESPONDA NO FORMATO:**
DECISÃO: Modelo 1 ou Modelo 2
RAZÃO: [Explique em 2-3 frases focando nos critérios acima]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # Baixa para respostas mais consistentes
    )
    
    resposta = response.choices[0].message.content
    
    if "Modelo 1" in resposta.split('\n')[0]:
        vencedor = "Modelo 1"
    elif "Modelo 2" in resposta.split('\n')[0]:
        vencedor = "Modelo 2"
    else:
        vencedor = "Indefinido"
    
    return vencedor, resposta


# Método 3: CHAIN-OF-THOUGHT
def evaluate_chain_of_thought(sbmn_text, bpmn1_text, bpmn2_text, metrics1, metrics2, sim1=None, sim2=None):
    """Método com raciocínio passo a passo explícito"""
    
    m1_fitness = metrics1['fitness']
    m1_precision = metrics1['precision']
    m1_generalization = metrics1['generalization']
    m1_simplicity = metrics1['simplicity']
    m1_hmean = harmonic_mean(metrics1)
    
    m2_fitness = metrics2['fitness']
    m2_precision = metrics2['precision']
    m2_generalization = metrics2['generalization']
    m2_simplicity = metrics2['simplicity']
    m2_hmean = harmonic_mean(metrics2)
    
    sim1_text = f"Similaridade estrutural: {sim1:.3f}" if sim1 is not None else "Similaridade não disponível"
    sim2_text = f"Similaridade estrutural: {sim2:.3f}" if sim2 is not None else "Similaridade não disponível"
    
    prompt = f"""Você é um especialista em mineração de processos e modelagem BPMN.

## RESTRIÇÕES DO PROCESSO (SBMN):
{sbmn_text}

## MODELO 1:
{bpmn1_text}
Fitness: {m1_fitness:.3f} | Precision: {m1_precision:.3f} | Generalization: {m1_generalization:.3f} | Simplicity: {m1_simplicity:.3f} | Hmean: {m1_hmean:.3f}
{sim1_text}

## MODELO 2:
{bpmn2_text}
Fitness: {m2_fitness:.3f} | Precision: {m2_precision:.3f} | Generalization: {m2_generalization:.3f} | Simplicity: {m2_simplicity:.3f} | Hmean: {m2_hmean:.3f}
{sim2_text}

## ANÁLISE PASSO A PASSO:

**Passo 1 - Conformidade com Restrições:**
Analise se cada modelo respeita as dependências, exclusões e outras restrições definidas no SBMN.

**Passo 2 - Análise de Fitness:**
Compare o fitness dos modelos (métrica mais importante). Quão bem cada um reproduz o comportamento do log?

**Passo 3 - Análise de Precision:**
Compare a precision. Qual modelo tem melhor controle sobre comportamentos extras?

**Passo 4 - Outras Métricas:**
Considere generalization e simplicity como complemento.

**Passo 5 - Decisão Final:**
Com base na análise acima, qual modelo é superior?

**RESPONDA SEGUINDO OS PASSOS:**
Passo 1: [sua análise]
Passo 2: [sua análise]
Passo 3: [sua análise]
Passo 4: [sua análise]
Passo 5: DECISÃO FINAL: Modelo 1 ou Modelo 2
JUSTIFICATIVA: [2 frases resumindo]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    resposta = response.choices[0].message.content
    
    # Procurar a decisão final no texto
    if "Modelo 1" in resposta.lower():
        vencedor = "Modelo 1"
    elif "Modelo 2" in resposta.lower():
        vencedor = "Modelo 2"
    else:
        vencedor = "Indefinido"
    
    return vencedor, resposta


# Método 4: CONSENSUS (Votação com múltiplas chamadas)
def evaluate_consensus(sbmn_text, bpmn1_text, bpmn2_text, metrics1, metrics2, sim1=None, sim2=None, n_calls=3):
    """Método com votação por maioria usando múltiplas chamadas"""
    
    votes = {"Modelo 1": 0, "Modelo 2": 0}
    explanations = []
    
    m1_fitness = metrics1['fitness']
    m1_precision = metrics1['precision']
    m1_generalization = metrics1['generalization']
    m1_simplicity = metrics1['simplicity']
    m1_hmean = harmonic_mean(metrics1)
    
    m2_fitness = metrics2['fitness']
    m2_precision = metrics2['precision']
    m2_generalization = metrics2['generalization']
    m2_simplicity = metrics2['simplicity']
    m2_hmean = harmonic_mean(metrics2)
    
    sim1_text = f"Similaridade: {sim1:.3f}" if sim1 is not None else ""
    sim2_text = f"Similaridade: {sim2:.3f}" if sim2 is not None else ""
    
    prompt = f"""Você é um especialista em classificação de modelos BPMN.

Restrições: {sbmn_text}

Modelo 1: {bpmn1_text}
Métricas: Fitness={m1_fitness:.3f}, Precision={m1_precision:.3f}, Generalization={m1_generalization:.3f}, Simplicity={m1_simplicity:.3f}, Hmean={m1_hmean:.3f} {sim1_text}

Modelo 2: {bpmn2_text}
Métricas: Fitness={m2_fitness:.3f}, Precision={m2_precision:.3f}, Generalization={m2_generalization:.3f}, Simplicity={m2_simplicity:.3f}, Hmean={m2_hmean:.3f} {sim2_text}

Qual modelo é melhor? Priorize: (1) conformidade com restrições, (2) fitness, (3) precision.

Responda EXATAMENTE:
DECISÃO: Modelo X
RAZÃO: [sua explicação]
"""
    
    for i in range(n_calls):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        content = response.choices[0].message.content
        explanations.append(content)
        
        if "Modelo 1" in content.split('\n')[0]:
            votes["Modelo 1"] += 1
        elif "Modelo 2" in content.split('\n')[0]:
            votes["Modelo 2"] += 1
    
    winner = max(votes, key=votes.get)
    confidence = votes[winner] / n_calls
    
    # Consolidar explicação
    explanation = f"CONSENSO ({votes['Modelo 1']}/{n_calls} votos para Modelo 1, {votes['Modelo 2']}/{n_calls} votos para Modelo 2)\n"
    explanation += f"Confiança: {confidence:.0%}\n\nExplicações:\n"
    for i, exp in enumerate(explanations, 1):
        explanation += f"\n--- Chamada {i} ---\n{exp}\n"
    
    return winner, explanation


# =========================
# FUNÇÃO PRINCIPAL DE AVALIAÇÃO
# =========================

def evaluate_complete_models(sbmn_text, bpmn1_text, bpmn2_text, metrics1, metrics2, 
                            sim1=None, sim2=None, method="explicit"):
    """
    Função principal que permite escolher o método de avaliação.
    
    Args:
        method: "baseline", "explicit", "cot" (chain-of-thought), "consensus"
    """
    
    if method == "baseline":
        return evaluate_baseline(sbmn_text, bpmn1_text, bpmn2_text, metrics1, metrics2, sim1, sim2)
    elif method == "explicit":
        return evaluate_explicit_criteria(sbmn_text, bpmn1_text, bpmn2_text, metrics1, metrics2, sim1, sim2)
    elif method == "cot":
        return evaluate_chain_of_thought(sbmn_text, bpmn1_text, bpmn2_text, metrics1, metrics2, sim1, sim2)
    elif method == "consensus":
        return evaluate_consensus(sbmn_text, bpmn1_text, bpmn2_text, metrics1, metrics2, sim1, sim2)
    else:
        # Default: explicit criteria (recomendado)
        return evaluate_explicit_criteria(sbmn_text, bpmn1_text, bpmn2_text, metrics1, metrics2, sim1, sim2)


# =========================
# VISUALIZACAO DO BPMN VENCEDOR
# =========================

def visualize_bpmn_to_png(bpmn_path, output_path):
    """Gera PNG da visualizacao BPMN"""
    try:
        from pm4py.visualization.bpmn import visualizer
        
        # Carregar modelo BPMN
        bpmn_model = pm4py.read_bpmn(bpmn_path)
        
        # Gerar visualizacao
        gviz = visualizer.apply(bpmn_model)
        
        # Salvar como PNG
        gviz.render(output_path, format='png', cleanup=True)
        
        return True
    except Exception as e:
        print(f"  [WARNING] Erro ao visualizar BPMN: {str(e)[:100]}")
        return False


def generate_winner_visualization(df, bpmn_base_path, save_dir):
    """Gera PNG do modelo mais votado"""
    try:
        # Contar votos por BPMN
        votos_dict = {}
        
        for idx, row in df.iterrows():
            voto = row['LLM_Voto']
            
            if voto == "Modelo 1":
                modelo = row['BPMN1_Arquivo']
            elif voto == "Modelo 2":
                modelo = row['BPMN2_Arquivo']
            else:
                continue
            
            if modelo not in votos_dict:
                votos_dict[modelo] = 0
            votos_dict[modelo] += 1
        
        if not votos_dict:
            print("  [WARNING] Nenhum modelo com votos encontrado")
            return
        
        # Encontrar vencedor
        modelo_vencedor = max(votos_dict, key=votos_dict.get)
        votos_vencedor = votos_dict[modelo_vencedor]
        total_votos = sum(votos_dict.values())
        percentual = (votos_vencedor / total_votos * 100) if total_votos > 0 else 0
        
        print(f"\n   [WINNER] MODELO MAIS VOTADO:")
        print(f"   {modelo_vencedor}")
        print(f"   Votos: {votos_vencedor}/{total_votos} ({percentual:.1f}%)\n")
        
        # Gerar PNG
        bpmn_path = os.path.join(bpmn_base_path, modelo_vencedor)
        
        if os.path.exists(bpmn_path):
            # Nome: winner_S10_proc_BPIC14-PreProcessed-Filtered.png
            nome_sem_extensao = modelo_vencedor.replace('.bpmn', '').replace('.xml', '')
            output_path = os.path.join(save_dir, f"winner_{nome_sem_extensao}")
            
            if visualize_bpmn_to_png(bpmn_path, output_path):
                print(f"   [OK] Winner salvo: winner_{nome_sem_extensao}.png")
            else:
                print(f"   [WARNING] Erro ao gerar PNG do vencedor")
        else:
            print(f"  [ERROR] Arquivo nao encontrado: {bpmn_path}")
    
    except Exception as e:
        print(f"  [ERROR] Erro ao gerar visualizacao do vencedor: {str(e)[:100]}")


# =========================
# ANALISE DE RANKING DOS MODELOS
# =========================

def generate_ranking_analysis(df, save_dir, method):
    """Gera ranking dos modelos mais votados por processo"""
    print(f"\n Analisando ranking de modelos...")
    
    # Contar votos por BPMN_Arquivo
    votos_dict = {}
    
    # Percorrer cada linha e contar votos
    for idx, row in df.iterrows():
        voto = row['LLM_Voto']  # "Modelo 1" ou "Modelo 2"
        
        # Extrair qual arquivo venceu
        if voto == "Modelo 1":
            modelo_vencedor = row['BPMN1_Arquivo']
        elif voto == "Modelo 2":
            modelo_vencedor = row['BPMN2_Arquivo']
        else:
            continue  # Pular votos indefinidos
        
        # Inicializar ou incrementar contador
        if modelo_vencedor not in votos_dict:
            votos_dict[modelo_vencedor] = 0
        votos_dict[modelo_vencedor] += 1
    
    # Ordenar por votos decrescentes
    ranking = sorted(votos_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Gerar relatório TXT
    txt_filename = os.path.join(save_dir, f"ranking_modelos_{method}.txt")
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RANKING DOS MODELOS MAIS VOTADOS - Método: {method.upper()}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total de comparações: {len(df)}\n")
        f.write(f"Total de modelos únicos: {len(votos_dict)}\n")
        f.write(f"Votos totais contabilizados: {sum([v for k, v in ranking])}\n\n")
        
        f.write("RANKING:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'POSIÇÃO':<10} {'MODELO':<50} {'VOTOS':<10} {'%':<10}\n")
        f.write("-" * 80 + "\n")
        
        total_votos = sum([v for k, v in ranking])
        
        for pos, (modelo, votos) in enumerate(ranking, 1):
            percentual = (votos / total_votos * 100) if total_votos > 0 else 0
            f.write(f"{pos:<10} {modelo:<50} {votos:<10} {percentual:>6.1f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"  [OK] Ranking salvo em: {txt_filename}")
    
    # Gerar gráficos de ranking
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['figure.max_open_warning'] = 0
        
        # ==========================================
        # GRAFICO 1: TREEMAP (Mapa de Arvore)
        # ==========================================
        try:
            import squarify
            
            top_n = min(20, len(ranking))
            top_ranking = ranking[:top_n]
            
            modelos = [m[0].replace(".bpmn", "").replace(".xml", "") for m, _ in top_ranking]
            votos = [v for _, v in top_ranking]
            
            fig, ax = plt.subplots(figsize=(16, 10))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(modelos)))
            squarify.plot(sizes=votos, label=modelos, color=colors, ax=ax, text_kwargs={'fontsize': 9})
            
            ax.set_title(f"Treemap - Ranking de Modelos Mais Votados ({method.upper()})", 
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            treemap_filename = os.path.join(save_dir, f"ranking_treemap_{method}.png")
            plt.savefig(treemap_filename, bbox_inches="tight", dpi=150)
            plt.close(fig)
            
            print(f"  [OK] Treemap salvo: ranking_treemap_{method}.png")
        except ImportError:
            print(f"  [WARNING] squarify nao instalado para Treemap")
        except Exception as e:
            print(f"  [WARNING] Erro ao gerar Treemap: {str(e)[:100]}")
        
        # ==========================================
        # GRAFICO 2: BARRAS VERTICAIS FINAS (Coluna)
        # ==========================================
        try:
            top_n = min(25, len(ranking))
            top_ranking = ranking[:top_n]
            
            modelos = [m[0].replace(".bpmn", "").replace(".xml", "") for m, _ in top_ranking]
            votos = [v for _, v in top_ranking]
            
            fig, ax = plt.subplots(figsize=(18, 8))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(modelos)))
            # Usar width pequeno para barras finas
            bars = ax.bar(range(len(modelos)), votos, color=colors, width=0.5)
            
            ax.set_xlabel("Modelo BPMN", fontsize=12, fontweight='bold')
            ax.set_ylabel("Numero de Votos", fontsize=12, fontweight='bold')
            ax.set_title(f"Ranking - Barras Verticais ({method.upper()})", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xticks(range(len(modelos)))
            ax.set_xticklabels(modelos, rotation=45, ha='right', fontsize=9)
            
            # Adicionar valores nas barras
            for bar, voto in zip(bars, votos):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(voto)}',
                       ha='center', va='bottom', fontsize=8)
            
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            barras_filename = os.path.join(save_dir, f"ranking_barras_{method}.png")
            plt.savefig(barras_filename, bbox_inches="tight", dpi=150)
            plt.close(fig)
            
            print(f"  [OK] Barras verticais salvo: ranking_barras_{method}.png")
        except Exception as e:
            print(f"  [WARNING] Erro ao gerar Barras: {str(e)[:100]}")
    
    except Exception as e:
        print(f"  [ERROR] Erro ao gerar graficos: {str(e)[:100]}")
    
    # Gerar relatório por SBMN (opcional)
    txt_sbmn_filename = os.path.join(save_dir, f"ranking_por_sbmn_{method}.txt")
    try:
        with open(txt_sbmn_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"VENCEDORES POR PROCESSO (SBMN) - Método: {method.upper()}\n")
            f.write("=" * 100 + "\n\n")
            
            # Agrupar por SBMN
            sbmn_results = {}
            for sbmn in df['SBMN_Arquivo'].unique():
                sbmn_results[sbmn] = {}
            
            for idx, row in df.iterrows():
                sbmn = row['SBMN_Arquivo']
                voto = row['LLM_Voto']
                
                if voto == "Modelo 1":
                    modelo = row['BPMN1_Arquivo']
                elif voto == "Modelo 2":
                    modelo = row['BPMN2_Arquivo']
                else:
                    continue
                
                if modelo not in sbmn_results[sbmn]:
                    sbmn_results[sbmn][modelo] = 0
                sbmn_results[sbmn][modelo] += 1
            
            # Exibir por SBMN
            for sbmn in sorted(sbmn_results.keys()):
                modelos_votos = sbmn_results[sbmn]
                if not modelos_votos:
                    continue
                
                ranking_sbmn = sorted(modelos_votos.items(), key=lambda x: x[1], reverse=True)
                total_votos_sbmn = sum([v for k, v in ranking_sbmn])
                
                f.write(f"SBMN: {sbmn}\n")
                f.write(f"  Total de comparações: {len(df[df['SBMN_Arquivo'] == sbmn])}\n")
                f.write(f"  Modelos votados:\n")
                
                for pos, (modelo, votos) in enumerate(ranking_sbmn, 1):
                    percentual = (votos / total_votos_sbmn * 100) if total_votos_sbmn > 0 else 0
                    f.write(f"    {pos}. {modelo}: {votos} votos ({percentual:.1f}%)\n")
                
                f.write("\n")
            
            f.write("=" * 100 + "\n")
        
        print(f"  [OK] Ranking por SBMN salvo em: {txt_sbmn_filename}")
    except Exception as e:
        print(f"  [WARNING] Erro ao gerar ranking por SBMN: {str(e)[:100]}")
    
    return ranking


# =========================
# 3. Folders
# =========================
if __name__ == '__main__':
    # ==========================================
    # CONFIGURAÇÃO: Escolha o modo de execução
    # ==========================================
    # Modo 1: PASTA_BASE = None -> Usar pasta única (abaixo)
    # Modo 2: PASTA_BASE = "caminho/para/base" -> Processar todas as subpastas
    
    PASTA_BASE = "F:/Danielle/Mestrado/BPMNClassifierLLM/INPUTS/EXPERIMENTOS_COMRM2"
    # PASTA_BASE = None  # Descomente para modo pasta única
    
    # CONFIGURAÇÃO: Escolha o método de avaliação
    # Opções: "baseline", "explicit", "cot", "consensus"
    EVALUATION_METHOD = "explicit"  # RECOMENDADO!
    
    PASTA_GRAFICOS_BASE = "F:/Danielle/Mestrado/BPMNClassifierLLM/OUTPUTS"
    os.makedirs(PASTA_GRAFICOS_BASE, exist_ok=True)

    # ==========================================
    # Definir lista de pastas a processar
    # ==========================================
    if PASTA_BASE and os.path.isdir(PASTA_BASE):
        # MODO BATCH: Processar todas as subpastas em PASTA_BASE
        print(f" Modo BATCH detectado")
        print(f" Pasta base: {PASTA_BASE}")
        subpastas = [d for d in os.listdir(PASTA_BASE) 
                     if os.path.isdir(os.path.join(PASTA_BASE, d))]
        pastas_trabalho = [(PASTA_BASE, subpasta) for subpasta in sorted(subpastas)]
        print(f" Encontradas {len(pastas_trabalho)} subpastas para processar")
        print("=" * 60)
    else:
        # MODO ÚNICO: Pasta única (configurada abaixo)
        PASTA_PASTA_UNICA = "F:/Danielle/Mestrado/BPMNClassifierLLM/INPUTS/FOLDERS/ComputerRepair_1"
        pastas_trabalho = [(PASTA_PASTA_UNICA, None)]
        print(f" Modo pasta única: {PASTA_PASTA_UNICA}")
        print("=" * 60)

    print(f" Iniciando classificação com método: {EVALUATION_METHOD.upper()}")
    print(f"  Usando {NUM_WORKERS} workers para processamento e {MAX_WORKERS_API} para API OpenAI")
    print("=" * 60)

    # ==========================================
    # Loop externo: Processar cada pasta de trabalho
    # ==========================================
    resultados_total = []
    start_time_total = time.time()

    for pasta_base_iter, nome_subpasta in pastas_trabalho:
        # ==========================================
        # Detectar estrutura de pastas
        # ==========================================
        if nome_subpasta:
            # MODO BATCH
            pasta_raiz = os.path.join(pasta_base_iter, nome_subpasta)
            process_name = nome_subpasta
            print(f"\n{'='*60}")
            print(f" Processando pasta: {process_name}")
            print(f"{'='*60}")
        else:
            # MODO ÚNICO
            pasta_raiz = pasta_base_iter
            process_name = "results"
        
        # [OK] CHECKPOINT: Inicializar gerenciador de checkpoint para esta pasta
        checkpoint = CheckpointManager(checkpoint_dir="CHECKPOINTS", process_name=process_name)
        
        # Procurar por arquivos SBMN e BPMN
        PASTA_SBMN = os.path.join(pasta_raiz, "DECS")
        PASTA_BPMN = pasta_raiz
        PASTA_LOGS = os.path.join(pasta_raiz, "LOGS")
        PASTA_REF = os.path.join(pasta_raiz, "REF")
        PASTA_GRAFICOS = os.path.join(PASTA_GRAFICOS_BASE, process_name)

        # Verificar se pastas existem
        if not os.path.isdir(PASTA_SBMN):
            print(f"    Pasta SBMN não existe: {PASTA_SBMN}")
            continue
        
        os.makedirs(PASTA_BPMN, exist_ok=True)
        os.makedirs(PASTA_LOGS, exist_ok=True)
        os.makedirs(PASTA_REF, exist_ok=True)
        os.makedirs(PASTA_GRAFICOS, exist_ok=True)

        print(f"   SBMN: {PASTA_SBMN}")
        print(f"   BPMN: {PASTA_BPMN}")
        print(f"   Saída: {PASTA_GRAFICOS}")

        sbmn_files = [f for f in os.listdir(PASTA_SBMN) if f.lower().endswith(".json")]
        bpmn_files = [f for f in os.listdir(PASTA_BPMN) if f.lower().endswith((".bpmn", ".xml"))]

        if not sbmn_files:
            print(f"    Nenhum arquivo SBMN encontrado em {PASTA_SBMN}")
            continue
        
        if not bpmn_files:
            print(f"    Nenhum arquivo BPMN encontrado em {PASTA_BPMN}")
            continue

        print(f"   Encontrados: {len(sbmn_files)} SBMN(s) e {len(bpmn_files)} BPMN(s)")

        resultados = []
        start_time = time.time()

        for sbmn_file in sbmn_files:
            # [OK] CHECKPOINT: Verificar se SBMN já foi processado
            if checkpoint.is_sbmn_processed(sbmn_file):
                print(f"\n [SKIP] SBMN já processado (skip): {sbmn_file}")
                continue
            
            print(f"\n Processando SBMN: {sbmn_file}")
            sbmn_start = time.time()
            
            with open(os.path.join(PASTA_SBMN, sbmn_file), "r", encoding="utf-8") as f:
                sbmn_data = json.load(f)
                sbmn_text = sbmn_to_text(sbmn_data)
                # [OK] CHECKPOINT: Marcar SBMN como processado
                checkpoint.mark_sbmn_processed(sbmn_file, sbmn_text)

            # Buscar o primeiro arquivo .xes na pasta LOGS
            log_files = [f for f in os.listdir(PASTA_LOGS) if f.lower().endswith(".xes")]
            if log_files:
                log_ref = os.path.join(PASTA_LOGS, log_files[0])
                print(f"     Usando log: {log_files[0]}")
            else:
                log_ref = None
                print(f"      Nenhum log encontrado. Métricas com log sintético.")

            ref_path = os.path.join(PASTA_REF, os.path.splitext(sbmn_file)[0] + ".bpmn")
            ref_path = ref_path if os.path.exists(ref_path) else None

            #  ETAPA 1: Calcular métricas em paralelo (com cache)
            metrics_dict = calculate_all_metrics_parallel(bpmn_files, PASTA_BPMN, log_ref, checkpoint=checkpoint)
            
            #  ETAPA 2: Calcular similaridades em paralelo (se houver referência)
            similarity_dict = {}
            if ref_path:
                similarity_dict = calculate_all_similarities_parallel(bpmn_files, PASTA_BPMN, ref_path)
            
            #  ETAPA 3: Preparar textos BPMN em paralelo (com cache e leitura paralela)
            bpmn_path_dict = load_bpmn_files_parallel(bpmn_files, PASTA_BPMN, 
                                                       checkpoint=checkpoint,
                                                       max_workers=min(NUM_WORKERS, 8))
            
            #  ETAPA 4: Processar comparações em paralelo (com throttling de API e checkpoint)
            comparacoes = process_all_comparisons_parallel(
                sbmn_text, bpmn_files, bpmn_path_dict, metrics_dict, 
                similarity_dict, sbmn_file, ref_path, EVALUATION_METHOD, checkpoint=checkpoint
            )
            
            resultados.extend(comparacoes)
            
            sbmn_elapsed = time.time() - sbmn_start
            print(f"      SBMN processado em {sbmn_elapsed:.2f}s ({len(comparacoes)} comparações)")

        # ==========================================
        # Exportar resultados desta pasta
        # ==========================================
        # [OK] CHECKPOINT: Recuperar resultados do checkpoint
        checkpoint_results = checkpoint.get_all_results()
        if checkpoint_results:
            resultados.extend(checkpoint_results)
        
        if resultados:
            df = pd.DataFrame(resultados)
            
            # Salvar CSV em OUTPUTS/ProcessName/
            csv_filename = os.path.join(PASTA_GRAFICOS, f"resultados_comparacao_{EVALUATION_METHOD}.csv")
            df.to_csv(csv_filename, index=False)
            print(f"   CSV salvo em: {csv_filename}")

            elapsed_time = time.time() - start_time
            print(f"    Tempo desta pasta: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)")

            # Gerar gráficos comparativos (PARALELO)
            print(f"   Gerando gráficos...")
            generate_all_plots_parallel(df, PASTA_GRAFICOS, EVALUATION_METHOD)
            print(f"   Gráficos salvos em: {PASTA_GRAFICOS}")
            
            # Gerar análise de ranking dos modelos
            print(f"   Gerando análise de ranking...")
            generate_ranking_analysis(df, PASTA_GRAFICOS, EVALUATION_METHOD)
            
            resultados_total.extend(resultados)
            
            # [OK] CHECKPOINT: Limpar checkpoint após sucesso
            print(f"   Limpando checkpoint...")
            checkpoint.cleanup()
            
            # Gerar visualização do modelo vencedor
            print(f"   Gerando visualização do vencedor...")
            generate_winner_visualization(df, PASTA_BPMN, PASTA_GRAFICOS)

    # ==========================================
    # Resumo final
    # ==========================================
    elapsed_time_total = time.time() - start_time_total
    print(f"\n{'='*60}")
    print(f" Processo BATCH concluído em {elapsed_time_total:.2f}s ({elapsed_time_total/60:.2f}min)!")
    print(f" Total de comparações: {len(resultados_total)}")
    print(f" Resultados salvos em: {PASTA_GRAFICOS_BASE}")
    print("=" * 60)
    print("RESUMO DOS MÉTODOS:")
    print("  - baseline: Método original")
    print("  - explicit: Critérios explícitos (RECOMENDADO)")
    print("  - cot: Chain-of-thought (raciocínio passo a passo)")
    print("  - consensus: Votação com múltiplas chamadas")
    print("=" * 60)
