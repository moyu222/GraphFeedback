from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import difflib
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from scipy.stats import binomtest
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPHCLIP_ROOT = REPO_ROOT / "external" / "GraphCLIP"
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "config.local.yaml"
OUTPUT_ROOT = REPO_ROOT / "experiments" / "outputs"
MODEL_CACHE_ROOT = REPO_ROOT / "models_cache"
CHECKPOINT = GRAPHCLIP_ROOT / "checkpoints" / "pretrained_graphclip.pt"
CHECKPOINT_DOWNLOAD = GRAPHCLIP_ROOT / "checkpoints" / "graphclip_checkpoint_download"
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?)(?!\w)")
CITATION_RE = re.compile(r"\[[0-9,;\s-]+\]")
SPECIAL_RE = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9-]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b")
NEGATIONS = {"no", "not", "never", "neither", "nor", "without", "cannot", "can't", "isn't", "wasn't", "don't", "doesn't", "didn't"}
POLARITY_TERMS = {"complete", "incomplete", "possible", "impossible", "increase", "decrease", "higher", "lower", "positive", "negative", "with", "without"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def run_directory(run_id: str) -> Path:
    return OUTPUT_ROOT / run_id / "citeseer"


def local_model_path(model_name: str) -> Path:
    return MODEL_CACHE_ROOT / model_name.replace("/", "--")


def resolve_model(model_name: str) -> str:
    local_path = local_model_path(model_name)
    if (local_path / "config.json").exists():
        return str(local_path)
    return model_name


@contextlib.contextmanager
def graphclip_context() -> Iterable[None]:
    previous = Path.cwd()
    inserted = False
    graphclip_string = str(GRAPHCLIP_ROOT)
    if not sys.path or sys.path[0] != graphclip_string:
        sys.path.insert(0, graphclip_string)
        inserted = True
    os.chdir(GRAPHCLIP_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)
        if inserted and sys.path and sys.path[0] == graphclip_string:
            sys.path.pop(0)


def prepare_checkpoint() -> dict[str, Any]:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.exists():
        if not CHECKPOINT_DOWNLOAD.exists():
            raise FileNotFoundError(
                "Checkpoint is missing. Run:\n"
                f"{sys.executable} -m gdown --id 178RikDLXPy-4eMGDhG5V6RzmlJhp-8fy "
                f"-O {CHECKPOINT_DOWNLOAD}"
            )
        try:
            loaded = torch.load(CHECKPOINT_DOWNLOAD, map_location="cpu", weights_only=True)
            if not isinstance(loaded, dict):
                raise TypeError("Downloaded torch object is not a state dictionary")
            CHECKPOINT_DOWNLOAD.replace(CHECKPOINT)
        except Exception as torch_error:
            if not zipfile.is_zipfile(CHECKPOINT_DOWNLOAD):
                raise RuntimeError("Downloaded checkpoint is neither a state dict nor a readable ZIP") from torch_error
            with zipfile.ZipFile(CHECKPOINT_DOWNLOAD) as archive:
                members = [
                    name for name in archive.namelist()
                    if name.lower().endswith(".pt")
                    and not name.startswith("__MACOSX/")
                    and not Path(name).name.startswith("._")
                ]
                if len(members) != 1:
                    raise RuntimeError(f"Expected one .pt inside checkpoint archive, found: {members}")
                temporary = CHECKPOINT.with_suffix(".pt.tmp")
                with archive.open(members[0]) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                temporary.replace(CHECKPOINT)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) < 10:
        raise RuntimeError("Checkpoint does not look like a GraphCLIP state dictionary")
    result = {"path": str(CHECKPOINT), "bytes": CHECKPOINT.stat().st_size, "sha256": sha256_file(CHECKPOINT)}
    print(json.dumps(result, indent=2))
    return result


def load_upstream_data(seed: int) -> tuple[Any, list[str], list[str], list[str]]:
    seed_everything(seed)
    with graphclip_context():
        from data.load import load_data

        data, texts, classes, descriptions = load_data("citeseer", seed=seed)
    return data, texts, classes, descriptions


def load_target_records() -> dict[int, dict[str, Any]]:
    path = GRAPHCLIP_ROOT / "target_data" / "citeseer.json"
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return {int(record["id"]): record for record in records}


def build_graph(data: Any, target_records: dict[int, dict[str, Any]], node_id: int) -> Data:
    record = target_records[node_id]
    edges = torch.as_tensor(record["graph"], dtype=torch.long)
    if edges.numel() == 0 or edges.shape[1] == 0:
        edges = torch.tensor([[node_id], [node_id]], dtype=torch.long)
    node_ids = torch.unique(edges)
    mapping = {int(global_id): local_id for local_id, global_id in enumerate(node_ids.tolist())}
    if node_id not in mapping:
        node_ids = torch.cat([node_ids, torch.tensor([node_id])])
        mapping[node_id] = len(node_ids) - 1
    local_edges = torch.tensor(
        [[mapping[int(value)] for value in edges[0]], [mapping[int(value)] for value in edges[1]]],
        dtype=torch.long,
    )
    graph = Data(
        edge_index=local_edges,
        x=data.x[node_ids].clone(),
        y=data.y[node_id].view(1),
        root_n_index=torch.tensor([mapping[node_id]], dtype=torch.long),
        node_id=torch.tensor([node_id], dtype=torch.long),
    )
    return T.AddRandomWalkPE(walk_length=32, attr_name="pe")(graph)


def adjacency(data: Any) -> dict[int, list[int]]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    for source, target in data.edge_index.t().tolist():
        neighbors[int(source)].add(int(target))
        neighbors[int(target)].add(int(source))
    return {node: sorted(values) for node, values in neighbors.items()}


class Victim:
    def __init__(self, config: dict[str, Any], classes: list[str], descriptions: list[str]):
        prepare_checkpoint()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(config["victim_batch_size"])
        with graphclip_context():
            from models import GraphCLIP
            from models import graphclip as graphclip_module
            from transformers import AutoTokenizer

            graphclip_module.text_ids["tiny"] = resolve_model(config["node_encoder_model"])
            self.model = GraphCLIP(384, 1024, 12, {"dropout": 0.0}, text_model="tiny")
            state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if len(missing) > 10 or len(unexpected) > 10:
                raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
            self.tokenizer = AutoTokenizer.from_pretrained(resolve_model(config["node_encoder_model"]))
        self.model.eval().to(self.device)
        prompts = [f"good paper of {name} {description}" for name, description in zip(classes, descriptions)]
        self.class_embeddings = self.encode_texts(prompts).to(self.device)
        self.class_embeddings = torch.nn.functional.normalize(self.class_embeddings, dim=-1)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], max_length: int = 512) -> torch.Tensor:
        outputs = []
        for start in range(0, len(texts), 32):
            batch = self.tokenizer(
                texts[start : start + 32], truncation=True, padding=True, max_length=max_length, return_tensors="pt"
            ).to(self.device)
            embeddings = self.model.encode_text(
                batch["input_ids"], batch.get("token_type_ids"), batch["attention_mask"]
            )
            outputs.append(embeddings.cpu())
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def infer(self, graphs: list[Data]) -> torch.Tensor:
        score_batches = []
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        for batch in loader:
            batch = batch.to(self.device)
            graph_embeddings, _ = self.model.encode_graph(batch)
            graph_embeddings = torch.nn.functional.normalize(graph_embeddings, dim=-1)
            score_batches.append((100.0 * graph_embeddings @ self.class_embeddings.T).softmax(dim=-1).cpu())
        return torch.cat(score_batches, dim=0)


def stratified_sample(node_ids: list[int], labels: list[int], size: int, seed: int) -> list[int]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for node_id, label in zip(node_ids, labels):
        grouped[int(label)].append(int(node_id))
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < min(size, len(node_ids)):
        progressed = False
        for label in sorted(grouped):
            if grouped[label] and len(selected) < size:
                selected.append(grouped[label].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected)


def classification_margin(scores: list[float] | np.ndarray, label: int) -> float:
    values = np.asarray(scores, dtype=float)
    alternatives = np.delete(values, int(label))
    return float(values[int(label)] - alternatives.max())


def stratified_low_margin_sample(
    node_ids: list[int], labels: list[int], score_by_node: dict[int, list[float]], size: int
) -> list[int]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for node_id, label in zip(node_ids, labels):
        grouped[int(label)].append(int(node_id))
    for label, values in grouped.items():
        values.sort(key=lambda node_id: (classification_margin(score_by_node[node_id], label), node_id))
    selected = []
    while len(selected) < min(size, len(node_ids)):
        progressed = False
        for label in sorted(grouped):
            if grouped[label] and len(selected) < size:
                selected.append(grouped[label].pop(0))
                progressed = True
        if not progressed:
            break
    return sorted(selected)


def base_manifest(config: dict[str, Any], config_path: Path, run_id: str) -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": prepare_checkpoint(),
        "dataset_sha256": sha256_file(GRAPHCLIP_ROOT / "processed_data" / "citeseer.pt"),
        "target_graphs_sha256": sha256_file(GRAPHCLIP_ROOT / "target_data" / "citeseer.json"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "config": config,
    }


def select_nodes(config: dict[str, Any], config_path: Path, run_id: str, sample_size: int, force: bool) -> None:
    directory = run_directory(run_id)
    selection_path = directory / "selection.jsonl"
    if selection_path.exists() and not force:
        print(f"Selection exists, skipping: {selection_path}")
        return
    directory.mkdir(parents=True, exist_ok=True)
    if force and selection_path.exists():
        selection_path.unlink()
    data, texts, classes, descriptions = load_upstream_data(int(config["seed"]))
    target_records = load_target_records()
    test_ids = data.test_mask.nonzero(as_tuple=False).view(-1).tolist()
    graphs = [build_graph(data, target_records, int(node_id)) for node_id in test_ids]
    victim = Victim(config, classes, descriptions)
    scores = victim.infer(graphs)
    predictions = scores.argmax(dim=1).tolist()
    labels = [int(data.y[node_id]) for node_id in test_ids]
    score_by_node = {int(node): scores[index].tolist() for index, node in enumerate(test_ids)}
    correct_ids = [node for node, prediction, label in zip(test_ids, predictions, labels) if prediction == label]
    correct_labels = [int(data.y[node]) for node in correct_ids]
    selection_strategy = str(config.get("selection_strategy", "stratified_random"))
    if selection_strategy == "stratified_low_margin":
        selected_ids = stratified_low_margin_sample(correct_ids, correct_labels, score_by_node, sample_size)
    elif selection_strategy == "stratified_random":
        selected_ids = stratified_sample(correct_ids, correct_labels, sample_size, int(config["seed"]))
    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")
    prediction_by_node = {int(node): int(predictions[index]) for index, node in enumerate(test_ids)}
    graph_neighbors = adjacency(data)
    for node_id in selected_ids:
        node_scores = score_by_node[node_id]
        top_indices = np.argsort(node_scores)[::-1][:3].tolist()
        neighbor_ids = graph_neighbors.get(node_id, [])[:5]
        append_jsonl(
            selection_path,
            {
                "node_id": node_id,
                "label": int(data.y[node_id]),
                "clean_prediction": prediction_by_node[node_id],
                "clean_scores": node_scores,
                "clean_margin": classification_margin(node_scores, int(data.y[node_id])),
                "original_text": texts[node_id],
                "top_classes": [
                    {"name": classes[index], "description": descriptions[index], "score": node_scores[index]}
                    for index in top_indices
                ],
                "neighbor_ids": neighbor_ids,
                "neighbor_texts": [texts[index][:700] for index in neighbor_ids],
            },
        )
    manifest = base_manifest(config, config_path, run_id)
    manifest.update(
        {
            "test_nodes": len(test_ids),
            "clean_correct": len(correct_ids),
            "clean_accuracy": len(correct_ids) / len(test_ids),
            "selected_nodes": len(selected_ids),
            "selected_node_ids": selected_ids,
            "selection_strategy": selection_strategy,
        }
    )
    atomic_json(directory / "run_manifest.json", manifest)
    print(f"Selected {len(selected_ids)} nodes; clean accuracy={manifest['clean_accuracy']:.4f}")


def random_edit_candidates(text: str, count: int, seed: int) -> list[str]:
    tokens = text.split()
    if len(tokens) < 5:
        return [text]
    candidates = []
    for candidate_index in range(count):
        rng = random.Random(seed + candidate_index * 1009)
        edited = tokens.copy()
        budget = max(1, min(3, int(len(tokens) * 0.08)))
        for _ in range(budget):
            if len(edited) < 3:
                break
            position = rng.randrange(1, len(edited) - 1)
            if rng.random() < 0.5:
                edited.pop(position)
            else:
                edited[position - 1], edited[position] = edited[position], edited[position - 1]
        candidates.append(" ".join(edited))
    return candidates


def build_prompt(record: dict[str, Any], method: str, count: int, attempt: int = 0) -> str:
    original = record["original_text"]
    word_count = len(word_tokens(original))
    replacement_count = max(2, min(8, int(np.ceil(word_count * 0.12))))
    common = (
        "\n\n<ORIGINAL>\n" + original + "\n</ORIGINAL>\n\n"
        f"MANDATORY: Propose between 1 and {replacement_count} LOCAL phrase replacements. Each 'old' phrase must be "
        "copied exactly from ORIGINAL and contain at most 6 words. Each 'new' phrase must keep the same meaning and "
        "similar length. Do not replace numbers, citations, named models, acronyms, or negations. Do not rewrite or "
        "summarize the full text.\n"
        f"VARIATION INDEX: {attempt + 1}/{count}. Choose a different set of editable phrases for this variation.\n"
        "FINAL FORMAT ONLY: {\"replacements\":[{\"old\":\"exact source phrase\",\"new\":\"replacement phrase\"}]}"
    )
    if method == "generic_paraphrase":
        return "TASK: Suggest conservative meaning-preserving lexical substitutions." + common
    if method == "non_graph_attack":
        return (
            "TASK: Produce subtle meaning-preserving rewrites intended to change an unknown text classifier's "
            "prediction. Prefer small lexical and syntactic changes; do not add a new topic or explicit class name.\n"
            + common
        )
    if method == "graph_prompt_attack":
        class_context = "\n".join(
            f"- {item['name']} (score={item['score']:.4f}): {item['description'][:350]}"
            for item in record["top_classes"]
        )
        neighbor_context = "\n".join(f"- {text}" for text in record["neighbor_texts"])
        return (
            "TASK: Produce subtle meaning-preserving rewrites intended to change the graph-aware classifier's "
            "prediction. Reduce confidence in the current top class by choosing truthful terminology that is also "
            "compatible with a plausible alternative class. Use the score alternatives and neighborhood only to "
            "choose ambiguous wording; never copy an unrelated topic or state a class label explicitly.\n\nTOP PREDICTED CLASSES:\n"
            + class_context
            + "\n\nONE-HOP NEIGHBOR TEXTS:\n"
            + neighbor_context
            + "\n\n"
            + common
        )
    raise ValueError(f"Unknown generated method: {method}")


def parse_candidates(raw: str, expected: int) -> list[str]:
    start, end = raw.find("["), raw.rfind("]")
    if start >= 0 and end > start:
        try:
            value = json.loads(raw[start : end + 1])
            if isinstance(value, list):
                candidates = []
                for item in value:
                    if isinstance(item, str) and item.strip():
                        candidates.append(item.strip())
                    elif isinstance(item, dict):
                        for key in ("text", "rewritten_text", "candidate"):
                            if isinstance(item.get(key), str) and item[key].strip():
                                candidates.append(item[key].strip())
                                break
                        if not candidates and isinstance(item.get("alternatives"), list):
                            candidates.extend(str(text).strip() for text in item["alternatives"] if str(text).strip())
                return candidates[:expected]
        except json.JSONDecodeError:
            pass
    lines = []
    for line in raw.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"')
        if cleaned:
            lines.append(cleaned)
    return lines[:expected]


def apply_edit_plan(raw: str, source_text: str, max_ratio: float, budget_reference: str | None = None) -> str | None:
    budget_reference = budget_reference or source_text
    start = raw.find("{")
    value = None
    if start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError:
            value = None
    replacements = value.get("replacements") if isinstance(value, dict) else None
    if not isinstance(replacements, list):
        pair = re.search(
            r'Original\s*:\s*["“](.+?)["”].*?New\s*:\s*["“](.+?)["”]',
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        replacements = [{"old": pair.group(1), "new": pair.group(2)}] if pair else None
    if not isinstance(replacements, list):
        if start < 0 and "\n\n" in raw:
            fallback = raw.split("\n\n", 1)[1].strip().strip("`").replace('"', "").strip()
            ratio = len(word_tokens(fallback)) / max(1, len(word_tokens(budget_reference)))
            if 0.80 <= ratio <= 1.20 and changed_token_ratio(budget_reference, fallback) <= max_ratio:
                return fallback
        return None
    candidate = source_text
    applied = 0
    for replacement in replacements:
        if not isinstance(replacement, dict):
            continue
        old = str(replacement.get("old", "")).strip()
        new = str(replacement.get("new", "")).strip()
        old_words, new_words = word_tokens(old), word_tokens(new)
        if not old or not new or old.lower() == new.lower():
            continue
        if len(old_words) > 8 or abs(len(old_words) - len(new_words)) > 2:
            continue
        if protected_items(old) - protected_items(new):
            continue
        pattern = re.compile(re.escape(old), re.IGNORECASE)
        if not pattern.search(candidate):
            continue
        proposed = pattern.sub(new, candidate, count=1)
        if changed_token_ratio(budget_reference, proposed) <= max_ratio:
            candidate = proposed
            applied += 1
    return candidate if applied else None


def feedback_base_method(method: str) -> str:
    mapping = {"feedback_non_graph": "non_graph_attack", "graph_feedback": "graph_prompt_attack"}
    if method not in mapping:
        raise ValueError(f"Unknown feedback method: {method}")
    return mapping[method]


def choose_candidate(candidates: list[dict[str, Any]], label: int) -> dict[str, Any] | None:
    successful = [candidate for candidate in candidates if candidate.get("success")]
    if successful:
        return max(successful, key=lambda candidate: candidate.get("semantic_similarity") or -1.0)
    if candidates:
        return min(candidates, key=lambda candidate: classification_margin(candidate["scores"], label))
    return None


def build_feedback_prompt(
    record: dict[str, Any], method: str, current_text: str, initial: list[dict[str, Any]], attempt: int, count: int
) -> str:
    label = int(record["label"])
    clean_margin = classification_margin(record["clean_scores"], label)
    best = choose_candidate(initial, label)
    best_margin = classification_margin(best["scores"], label) if best else clean_margin
    history = []
    for index, candidate in enumerate(initial, 1):
        margin = classification_margin(candidate["scores"], label)
        history.append(
            f"- trial {index}: top-class margin change={margin - clean_margin:+.4f}; "
            f"top-class score={candidate['scores'][label]:.4f}; prediction_changed={bool(candidate['success'])}"
        )
    graph_context = ""
    if method == "graph_feedback":
        neighbor_context = "\n".join(f"- {text}" for text in record["neighbor_texts"])
        graph_context = (
            "\nUse the neighbor texts only to choose truthful, locally ambiguous terminology; do not copy an unrelated topic."
            "\n\nONE-HOP NEIGHBOR TEXTS:\n" + neighbor_context
        )
    word_count = len(word_tokens(record["original_text"]))
    replacement_count = max(2, min(8, int(np.ceil(word_count * 0.12))))
    return (
        "TASK: Refine the current academic text using black-box score feedback. The previous edits did not change "
        "the prediction. Make different, subtle phrase replacements that further reduce the original top-class "
        "margin while preserving the paper's meaning. Do not state a class label or add a new topic.\n\n"
        f"ORIGINAL TOP CLASS: {record['top_classes'][0]['name']}\n"
        f"ORIGINAL TOP-CLASS SCORE: {record['clean_scores'][label]:.4f}\n"
        f"ORIGINAL MARGIN: {clean_margin:.4f}\nCURRENT BEST MARGIN: {best_margin:.4f}\n"
        "OBSERVED TRIAL FEEDBACK:\n" + ("\n".join(history) if history else "- no valid first-round query")
        + graph_context
        + "\n\n<ORIGINAL>\n" + record["original_text"] + "\n</ORIGINAL>"
        + "\n\n<CURRENT>\n" + current_text + "\n</CURRENT>\n\n"
        f"MANDATORY: Propose between 1 and {replacement_count} LOCAL phrase replacements in CURRENT. Each 'old' "
        "phrase must be copied exactly from CURRENT and contain at most 6 words. Preserve numbers, citations, named "
        "models, acronyms, negations, and meaning. Keep the total change from ORIGINAL below 20%.\n"
        f"REFINEMENT VARIATION: {attempt + 1}/{count}. Use a different edit strategy for this variation.\n"
        'FINAL FORMAT ONLY: {"replacements":[{"old":"exact phrase from CURRENT","new":"replacement phrase"}]}'
    )


def generate(config: dict[str, Any], run_id: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = run_directory(run_id)
    selection = read_jsonl(directory / "selection.jsonl")
    output = directory / "generations.jsonl"
    done = {(record["node_id"], record["method"]) for record in read_jsonl(output)}
    methods = list(config["methods"])
    pending_llm = any((record["node_id"], method) not in done for record in selection for method in methods if method != "random_edit")
    tokenizer = model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pending_llm:
        generator_source = resolve_model(config["generator_model"])
        tokenizer = AutoTokenizer.from_pretrained(generator_source)
        model = AutoModelForCausalLM.from_pretrained(
            generator_source, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        if float(config["temperature"]) == 0:
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
        manifest_path = directory / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["generator_model"] = config["generator_model"]
        manifest["generator_revision"] = getattr(model.config, "_commit_hash", None)
        atomic_json(manifest_path, manifest)
    for item in selection:
        for method in methods:
            key = (item["node_id"], method)
            if key in done:
                continue
            started = time.perf_counter()
            prompt = None
            raw = None
            error = None
            if method == "random_edit":
                candidates = random_edit_candidates(
                    item["original_text"], int(config["candidate_count"]), int(config["seed"]) + int(item["node_id"])
                )
            else:
                attempts = int(config.get("generation_attempts", config["candidate_count"]))
                candidates = []
                prompts = []
                raw_outputs = []
                errors = []
                try:
                    for attempt in range(attempts):
                        prompt = build_prompt(item, method, attempts, attempt)
                        prompts.append(prompt)
                        messages = [
                            {"role": "system", "content": "You are a precise academic text editor."},
                            {"role": "user", "content": prompt},
                        ]
                        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=3072).to(device)
                        kwargs = {
                            "max_new_tokens": int(config["max_new_tokens"]),
                            "do_sample": float(config["temperature"]) > 0,
                            "pad_token_id": tokenizer.eos_token_id,
                        }
                        if kwargs["do_sample"]:
                            kwargs["temperature"] = float(config["temperature"])
                        attempt_seed = int(config["seed"]) + int(item["node_id"]) * 101 + attempt * 1009
                        torch.manual_seed(attempt_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(attempt_seed)
                        try:
                            with torch.inference_mode():
                                generated = model.generate(**inputs, **kwargs)
                            raw = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                            raw_outputs.append(raw)
                            edited = apply_edit_plan(
                                raw, item["original_text"], float(config["max_changed_token_ratio"]) * 0.95
                            )
                            if edited and edited not in candidates:
                                candidates.append(edited)
                            if len(candidates) >= int(config["candidate_count"]):
                                break
                        except Exception as exception:
                            errors.append(f"attempt={attempt}: {type(exception).__name__}: {exception}")
                    prompt = prompts
                    raw = raw_outputs
                    error = "; ".join(errors) if errors else None
                except Exception as exception:
                    candidates = []
                    error = f"{type(exception).__name__}: {exception}"
            append_jsonl(
                output,
                {
                    "node_id": item["node_id"],
                    "method": method,
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(json.dumps(prompt, ensure_ascii=False)) if prompt else None,
                    "raw_output": raw,
                    "candidates": candidates,
                    "generation_seconds": time.perf_counter() - started,
                    "error": error,
                },
            )
            print(f"generated node={item['node_id']} method={method} candidates={len(candidates)}")


def word_tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def changed_token_ratio(original: str, candidate: str) -> float:
    original_tokens, candidate_tokens = word_tokens(original), word_tokens(candidate)
    matcher = difflib.SequenceMatcher(a=original_tokens, b=candidate_tokens, autojunk=False)
    edits = 0
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        if operation != "equal":
            edits += max(i2 - i1, j2 - j1)
    return edits / max(1, len(original_tokens))


def protected_items(text: str) -> set[str]:
    negations = {token for token in word_tokens(text) if token in NEGATIONS}
    polarity = {token for token in word_tokens(text) if token in POLARITY_TERMS}
    return set(NUMBER_RE.findall(text)) | set(CITATION_RE.findall(text)) | set(SPECIAL_RE.findall(text)) | negations | polarity


def lexical_checks(original: str, candidate: str, config: dict[str, Any]) -> dict[str, Any]:
    original_words, candidate_words = word_tokens(original), word_tokens(candidate)
    length_ratio = len(candidate_words) / max(1, len(original_words))
    edit_ratio = changed_token_ratio(original, candidate)
    missing = sorted(protected_items(original) - protected_items(candidate))
    alpha = [character for character in candidate if character.isalpha()]
    ascii_ratio = sum(character.isascii() for character in alpha) / max(1, len(alpha))
    reasons = []
    if not candidate.strip(): reasons.append("empty")
    if ascii_ratio < 0.90: reasons.append("not_english")
    if not float(config["min_length_ratio"]) <= length_ratio <= float(config["max_length_ratio"]): reasons.append("length")
    if edit_ratio > float(config["max_changed_token_ratio"]): reasons.append("edit_budget")
    if missing: reasons.append("protected_items")
    return {"length_ratio": length_ratio, "changed_token_ratio": edit_ratio, "missing_protected": missing, "reasons": reasons}


def filter_candidates(config: dict[str, Any], run_id: str) -> None:
    from sentence_transformers import SentenceTransformer

    directory = run_directory(run_id)
    selections = {record["node_id"]: record for record in read_jsonl(directory / "selection.jsonl")}
    generations = read_jsonl(directory / "generations.jsonl")
    output = directory / "filtered.jsonl"
    done = {(record["node_id"], record["method"]) for record in read_jsonl(output)}
    pending = [record for record in generations if (record["node_id"], record["method"]) not in done]
    if pending:
        model = SentenceTransformer(resolve_model(config["semantic_model"]), device="cuda" if torch.cuda.is_available() else "cpu")
    for record in pending:
        original = selections[record["node_id"]]["original_text"]
        checked = []
        lexical_valid = []
        for candidate in record["candidates"]:
            checks = lexical_checks(original, candidate, config)
            entry = {"text": candidate, **checks, "semantic_similarity": None, "valid": False}
            checked.append(entry)
            if not checks["reasons"]:
                lexical_valid.append(entry)
        if lexical_valid:
            texts = [original] + [entry["text"] for entry in lexical_valid]
            embeddings = model.encode(
                texts, batch_size=int(config["embedding_batch_size"]), normalize_embeddings=True, convert_to_numpy=True
            )
            similarities = embeddings[1:] @ embeddings[0]
            for entry, similarity in zip(lexical_valid, similarities.tolist()):
                entry["semantic_similarity"] = float(similarity)
                if similarity < float(config["min_semantic_similarity"]):
                    entry["reasons"].append("semantic_similarity")
                entry["valid"] = not entry["reasons"]
        append_jsonl(output, {"node_id": record["node_id"], "method": record["method"], "candidates": checked})
        print(f"filtered node={record['node_id']} method={record['method']} valid={sum(x['valid'] for x in checked)}")
    all_filtered = read_jsonl(output)
    llm_methods = {method for method in config["methods"] if method != "random_edit"}
    llm_generations = [record for record in generations if record["method"] in llm_methods]
    llm_filtered = [record for record in all_filtered if record["method"] in llm_methods]
    parsing_rate = sum(bool(record["candidates"]) for record in llm_generations) / max(1, len(llm_generations))
    valid_rate = sum(any(candidate["valid"] for candidate in record["candidates"]) for record in llm_filtered) / max(1, len(llm_filtered))
    print(f"quality gate: parsing_rate={parsing_rate:.3f}, valid_record_rate={valid_rate:.3f}")
    if parsing_rate < float(config["min_parsing_rate"]) or valid_rate < float(config["min_valid_record_rate"]):
        raise RuntimeError("Generated-candidate quality gate failed; inspect generations/filtered logs before continuing")


def evaluate(config: dict[str, Any], run_id: str) -> None:
    directory = run_directory(run_id)
    selections = {record["node_id"]: record for record in read_jsonl(directory / "selection.jsonl")}
    filtered = read_jsonl(directory / "filtered.jsonl")
    output = directory / "evaluated.jsonl"
    done = {(record["node_id"], record["method"]) for record in read_jsonl(output)}
    by_node: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in filtered:
        if (record["node_id"], record["method"]) not in done:
            by_node[int(record["node_id"])].append(record)
    if not by_node:
        print("No evaluation work remains")
        return
    data, _, classes, descriptions = load_upstream_data(int(config["seed"]))
    target_records = load_target_records()
    victim = Victim(config, classes, descriptions)
    for node_id in sorted(by_node):
        base_graph = build_graph(data, target_records, node_id)
        for record in by_node[node_id]:
            valid = [candidate for candidate in record["candidates"] if candidate["valid"]][
                : int(config.get("query_budget", config["candidate_count"]))
            ]
            evaluated = []
            if valid:
                features = victim.encode_texts(
                    [candidate["text"] for candidate in valid], max_length=int(config["node_max_length"])
                )
                graphs = []
                for feature in features:
                    graph = base_graph.clone()
                    graph.x = graph.x.clone()
                    root = int(graph.root_n_index.item())
                    graph.x[root] = feature.to(graph.x.dtype)
                    graphs.append(graph)
                scores = victim.infer(graphs)
                for candidate, candidate_scores in zip(valid, scores.tolist()):
                    prediction = int(np.argmax(candidate_scores))
                    evaluated.append({**candidate, "scores": candidate_scores, "prediction": prediction, "success": prediction != selections[node_id]["label"]})
            successful = [candidate for candidate in evaluated if candidate["success"]]
            if successful:
                selected = max(successful, key=lambda candidate: candidate["semantic_similarity"])
            elif evaluated:
                label = int(selections[node_id]["label"])
                selected = min(evaluated, key=lambda candidate: classification_margin(candidate["scores"], label))
            else:
                selected = None
            append_jsonl(
                output,
                {
                    "node_id": node_id,
                    "method": record["method"],
                    "label": selections[node_id]["label"],
                    "clean_scores": selections[node_id]["clean_scores"],
                    "valid_candidate_count": len(valid),
                    "query_count": len(valid),
                    "evaluated_candidates": evaluated,
                    "selected": selected,
                    "success": bool(selected and selected["success"]),
                },
            )
            print(f"evaluated node={node_id} method={record['method']} success={bool(selected and selected['success'])}")


def feedback_generate(config: dict[str, Any], run_id: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = run_directory(run_id)
    selections = {record["node_id"]: record for record in read_jsonl(directory / "selection.jsonl")}
    base_evaluated = {(record["node_id"], record["method"]): record for record in read_jsonl(directory / "evaluated.jsonl")}
    output = directory / "feedback_generations.jsonl"
    done = {(record["node_id"], record["method"]) for record in read_jsonl(output)}
    methods = list(config.get("feedback_methods", []))
    pending = [(node_id, method) for node_id in selections for method in methods if (node_id, method) not in done]
    if not pending:
        print("No feedback generation work remains")
        return
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feedback_extension"] = {
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "feedback_methods": methods,
        "initial_queries": int(config["feedback_initial_queries"]),
        "refinement_queries": int(config["feedback_refinement_queries"]),
        "max_changed_token_ratio": float(config["max_changed_token_ratio"]),
        "min_semantic_similarity": float(config["min_semantic_similarity"]),
        "generator_model": config["generator_model"],
    }
    atomic_json(manifest_path, manifest)
    generator_source = resolve_model(config["generator_model"])
    tokenizer = AutoTokenizer.from_pretrained(generator_source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        generator_source,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    refinement_queries = int(config["feedback_refinement_queries"])
    initial_queries = int(config["feedback_initial_queries"])
    for node_id, method in pending:
        item = selections[node_id]
        base_method = feedback_base_method(method)
        base = base_evaluated.get((node_id, base_method))
        if base is None:
            raise RuntimeError(f"Missing base evaluation for node={node_id} method={base_method}")
        initial = base["evaluated_candidates"][:initial_queries]
        initial_success = any(candidate["success"] for candidate in initial)
        started = time.perf_counter()
        prompts: list[str] = []
        raw_outputs: list[str] = []
        candidates: list[str] = []
        errors: list[str] = []
        skipped_reason = "round_1_success" if initial_success else None
        if not initial_success:
            best = choose_candidate(initial, int(item["label"]))
            current_text = best["text"] if best else item["original_text"]
            for attempt in range(refinement_queries):
                prompt = build_feedback_prompt(item, method, current_text, initial, attempt, refinement_queries)
                prompts.append(prompt)
                messages = [
                    {"role": "system", "content": "You are a precise academic text editor using numerical feedback."},
                    {"role": "user", "content": prompt},
                ]
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=3072).to(device)
                kwargs = {
                    "max_new_tokens": int(config["max_new_tokens"]),
                    "do_sample": float(config["temperature"]) > 0,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if kwargs["do_sample"]:
                    kwargs["temperature"] = float(config["temperature"])
                attempt_seed = int(config["seed"]) + int(node_id) * 103 + attempt * 2017 + (1 if method == "graph_feedback" else 0)
                torch.manual_seed(attempt_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(attempt_seed)
                try:
                    with torch.inference_mode():
                        generated = model.generate(**inputs, **kwargs)
                    raw = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                    raw_outputs.append(raw)
                    edited = apply_edit_plan(
                        raw,
                        current_text,
                        float(config["max_changed_token_ratio"]) * 0.95,
                        budget_reference=item["original_text"],
                    )
                    if edited and edited not in candidates:
                        candidates.append(edited)
                except Exception as exception:
                    errors.append(f"attempt={attempt}: {type(exception).__name__}: {exception}")
        append_jsonl(
            output,
            {
                "node_id": node_id,
                "method": method,
                "base_method": base_method,
                "initial_query_count": len(initial),
                "prompt": prompts,
                "prompt_sha256": sha256_text(json.dumps(prompts, ensure_ascii=False)) if prompts else None,
                "raw_output": raw_outputs,
                "candidates": candidates,
                "generation_seconds": time.perf_counter() - started,
                "skipped_reason": skipped_reason,
                "error": "; ".join(errors) if errors else None,
            },
        )
        print(f"feedback-generated node={node_id} method={method} candidates={len(candidates)} skipped={skipped_reason}")


def feedback_filter(config: dict[str, Any], run_id: str) -> None:
    from sentence_transformers import SentenceTransformer

    directory = run_directory(run_id)
    selections = {record["node_id"]: record for record in read_jsonl(directory / "selection.jsonl")}
    generations = read_jsonl(directory / "feedback_generations.jsonl")
    output = directory / "feedback_filtered.jsonl"
    done = {(record["node_id"], record["method"]) for record in read_jsonl(output)}
    pending = [record for record in generations if (record["node_id"], record["method"]) not in done]
    model = None
    if any(record["candidates"] for record in pending):
        model = SentenceTransformer(resolve_model(config["semantic_model"]), device="cuda" if torch.cuda.is_available() else "cpu")
    for record in pending:
        original = selections[record["node_id"]]["original_text"]
        checked = []
        lexical_valid = []
        for candidate in record["candidates"]:
            checks = lexical_checks(original, candidate, config)
            entry = {"text": candidate, **checks, "semantic_similarity": None, "valid": False}
            checked.append(entry)
            if not checks["reasons"]:
                lexical_valid.append(entry)
        if lexical_valid and model is not None:
            texts = [original] + [entry["text"] for entry in lexical_valid]
            embeddings = model.encode(
                texts, batch_size=int(config["embedding_batch_size"]), normalize_embeddings=True, convert_to_numpy=True
            )
            for entry, similarity in zip(lexical_valid, (embeddings[1:] @ embeddings[0]).tolist()):
                entry["semantic_similarity"] = float(similarity)
                if similarity < float(config["min_semantic_similarity"]):
                    entry["reasons"].append("semantic_similarity")
                entry["valid"] = not entry["reasons"]
        append_jsonl(
            output,
            {"node_id": record["node_id"], "method": record["method"], "skipped_reason": record["skipped_reason"], "candidates": checked},
        )
        print(f"feedback-filtered node={record['node_id']} method={record['method']} valid={sum(x['valid'] for x in checked)}")
    active_generations = [record for record in generations if not record.get("skipped_reason")]
    all_filtered = read_jsonl(output)
    active_filtered = [record for record in all_filtered if not record.get("skipped_reason")]
    parsing_rate = sum(bool(record["candidates"]) for record in active_generations) / max(1, len(active_generations))
    valid_rate = sum(any(candidate["valid"] for candidate in record["candidates"]) for record in active_filtered) / max(1, len(active_filtered))
    print(f"feedback quality gate: parsing_rate={parsing_rate:.3f}, valid_record_rate={valid_rate:.3f}")
    if active_generations and (parsing_rate < float(config["min_parsing_rate"]) or valid_rate < float(config["min_valid_record_rate"])):
        raise RuntimeError("Feedback candidate quality gate failed; inspect feedback logs before continuing")


def feedback_evaluate(config: dict[str, Any], run_id: str) -> None:
    directory = run_directory(run_id)
    selections = {record["node_id"]: record for record in read_jsonl(directory / "selection.jsonl")}
    base_evaluated = {(record["node_id"], record["method"]): record for record in read_jsonl(directory / "evaluated.jsonl")}
    filtered = {(record["node_id"], record["method"]): record for record in read_jsonl(directory / "feedback_filtered.jsonl")}
    output = directory / "feedback_evaluated.jsonl"
    trajectory_output = directory / "feedback_trajectories.jsonl"
    done = {(record["node_id"], record["method"]) for record in read_jsonl(output)}
    pending = [(node_id, method) for node_id in selections for method in config.get("feedback_methods", []) if (node_id, method) not in done]
    if not pending:
        print("No feedback evaluation work remains")
        return
    data, _, classes, descriptions = load_upstream_data(int(config["seed"]))
    target_records = load_target_records()
    victim = Victim(config, classes, descriptions)
    initial_queries = int(config["feedback_initial_queries"])
    refinement_queries = int(config["feedback_refinement_queries"])
    for node_id, method in pending:
        item = selections[node_id]
        base_method = feedback_base_method(method)
        base = base_evaluated[(node_id, base_method)]
        initial = base["evaluated_candidates"][:initial_queries]
        refinement = []
        if not any(candidate["success"] for candidate in initial):
            valid = [candidate for candidate in filtered[(node_id, method)]["candidates"] if candidate["valid"]][:refinement_queries]
            if valid:
                base_graph = build_graph(data, target_records, node_id)
                features = victim.encode_texts([candidate["text"] for candidate in valid], max_length=int(config["node_max_length"]))
                graphs = []
                for feature in features:
                    graph = base_graph.clone()
                    graph.x = graph.x.clone()
                    graph.x[int(graph.root_n_index.item())] = feature.to(graph.x.dtype)
                    graphs.append(graph)
                scores = victim.infer(graphs)
                for candidate, candidate_scores in zip(valid, scores.tolist()):
                    prediction = int(np.argmax(candidate_scores))
                    refinement.append(
                        {**candidate, "scores": candidate_scores, "prediction": prediction, "success": prediction != int(item["label"])}
                    )
        combined = initial + refinement
        selected = choose_candidate(combined, int(item["label"]))
        clean_margin = classification_margin(item["clean_scores"], int(item["label"]))
        for query_index, candidate in enumerate(combined, 1):
            margin = classification_margin(candidate["scores"], int(item["label"]))
            append_jsonl(
                trajectory_output,
                {
                    "node_id": node_id,
                    "method": method,
                    "query_index": query_index,
                    "round": 1 if query_index <= len(initial) else 2,
                    "text": candidate["text"],
                    "scores": candidate["scores"],
                    "margin": margin,
                    "margin_change": margin - clean_margin,
                    "success": candidate["success"],
                },
            )
        result = {
            "node_id": node_id,
            "method": method,
            "base_method": base_method,
            "label": item["label"],
            "clean_scores": item["clean_scores"],
            "initial_candidates": initial,
            "refinement_candidates": refinement,
            "evaluated_candidates": combined,
            "valid_candidate_count": len(combined),
            "query_count": len(combined),
            "selected": selected,
            "success": bool(selected and selected["success"]),
        }
        append_jsonl(output, result)
        print(f"feedback-evaluated node={node_id} method={method} queries={len(combined)} success={result['success']}")


def feedback_finalize(config: dict[str, Any], run_id: str) -> None:
    directory = run_directory(run_id)
    target = directory / "evaluated.jsonl"
    done = {(record["node_id"], record["method"]) for record in read_jsonl(target)}
    added = 0
    for record in read_jsonl(directory / "feedback_evaluated.jsonl"):
        key = (record["node_id"], record["method"])
        if key not in done:
            append_jsonl(target, record)
            done.add(key)
            added += 1
    print(f"Finalized {added} feedback evaluation records")


def bootstrap_rate(values: np.ndarray, seed: int, repeats: int = 10000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    rates = values[indices].mean(axis=1)
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def aggregate(config: dict[str, Any], run_id: str) -> None:
    directory = run_directory(run_id)
    records = read_jsonl(directory / "evaluated.jsonl")
    generations = {(record["node_id"], record["method"]): record for record in read_jsonl(directory / "generations.jsonl")}
    feedback_generations = {
        (record["node_id"], record["method"]): record for record in read_jsonl(directory / "feedback_generations.jsonl")
    }
    if not records:
        raise RuntimeError("No evaluated records found")
    methods = sorted({record["method"] for record in records})
    summary_rows = []
    for method in methods:
        subset = [record for record in records if record["method"] == method]
        successes = np.array([record["success"] for record in subset], dtype=float)
        low, high = bootstrap_rate(successes, int(config["seed"]))
        selected = [record["selected"] for record in subset if record["selected"]]
        semantic = [item["semantic_similarity"] for item in selected]
        edits = [item["changed_token_ratio"] for item in selected]
        queries = [int(record.get("query_count", record["valid_candidate_count"])) for record in subset]
        successful_queries = [
            int(record.get("query_count", record["valid_candidate_count"])) for record in subset if record["success"]
        ]
        margin_reductions = []
        for record in subset:
            if record["selected"]:
                label = int(record["label"])
                margin_reductions.append(
                    classification_margin(record["clean_scores"], label)
                    - classification_margin(record["selected"]["scores"], label)
                )
        timings = []
        for record in subset:
            key = (record["node_id"], method)
            if key in generations:
                timings.append(float(generations[key]["generation_seconds"]))
            elif key in feedback_generations:
                base_key = (record["node_id"], feedback_base_method(method))
                base_seconds = float(generations.get(base_key, {}).get("generation_seconds", 0.0))
                timings.append(base_seconds + float(feedback_generations[key]["generation_seconds"]))
        summary_rows.append(
            {
                "method": method,
                "attempted_nodes": len(subset),
                "successful_nodes": int(successes.sum()),
                "asr": float(successes.mean()),
                "asr_ci_low": low,
                "asr_ci_high": high,
                "attacked_subset_accuracy": 1.0 - float(successes.mean()),
                "no_valid_candidate_rate": sum(record["valid_candidate_count"] == 0 for record in subset) / len(subset),
                "median_semantic_similarity": float(np.median(semantic)) if semantic else float("nan"),
                "median_changed_token_ratio": float(np.median(edits)) if edits else float("nan"),
                "mean_queries": float(np.mean(queries)) if queries else float("nan"),
                "mean_queries_success": float(np.mean(successful_queries)) if successful_queries else float("nan"),
                "median_margin_reduction": float(np.median(margin_reductions)) if margin_reductions else float("nan"),
                "generation_seconds": float(sum(timings)),
            }
        )
    summary_path = directory / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    by_key = {(record["node_id"], record["method"]): int(record["success"]) for record in records}
    comparisons = []
    comparison_specs = [
        ("graph_prompt_vs_non_graph", "graph_prompt_attack", "non_graph_attack"),
        ("graph_feedback_vs_feedback_non_graph", "graph_feedback", "feedback_non_graph"),
        ("graph_feedback_vs_graph_prompt", "graph_feedback", "graph_prompt_attack"),
    ]
    all_nodes = {record["node_id"] for record in records}
    for name, left, right in comparison_specs:
        common_nodes = sorted(node for node in all_nodes if (node, left) in by_key and (node, right) in by_key)
        if not common_nodes:
            continue
        left_only = sum(by_key[(node, left)] == 1 and by_key[(node, right)] == 0 for node in common_nodes)
        right_only = sum(by_key[(node, left)] == 0 and by_key[(node, right)] == 1 for node in common_nodes)
        discordant = left_only + right_only
        p_value = binomtest(min(left_only, right_only), discordant, 0.5).pvalue if discordant else 1.0
        comparisons.append(
            {
                "comparison": name,
                "left_method": left,
                "right_method": right,
                "paired_nodes": len(common_nodes),
                "left_only_success": left_only,
                "right_only_success": right_only,
                "exact_mcnemar_p": p_value,
            }
        )
    atomic_json(directory / "paired_comparisons.json", comparisons)
    with (directory / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    lines = [
        "# GraphFeedback Final Local Experiment Report",
        "",
        f"- Run: `{run_id}`",
        f"- Dataset: CiteSeer",
        f"- Clean test accuracy: {manifest['clean_accuracy']:.4f} ({manifest['clean_correct']}/{manifest['test_nodes']})",
        f"- Attacked sample: {manifest['selected_nodes']} initially correct nodes",
        f"- Selection strategy: `{manifest.get('selection_strategy', 'not recorded')}`",
        "",
        "## Main results",
        "",
        "| Method | ASR (95% bootstrap CI) | Mean queries | Median margin reduction | Median semantic similarity | Generation time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['asr']:.3f} [{row['asr_ci_low']:.3f}, {row['asr_ci_high']:.3f}] | "
            f"{row['mean_queries']:.2f} | {row['median_margin_reduction']:.4f} | "
            f"{row['median_semantic_similarity']:.3f} | {row['generation_seconds']:.1f} |"
        )
    lines += ["", "## Paired comparisons", ""]
    for comparison in comparisons:
        lines.append(
            f"- {comparison['left_method']} vs {comparison['right_method']}: left-only successes="
            f"{comparison['left_only_success']}, right-only successes={comparison['right_only_success']}, "
            f"exact paired p={comparison['exact_mcnemar_p']:.4f}."
        )
    lines += ["",
        "## Claim boundary",
        "",
        "These results describe one released GraphCLIP checkpoint, one CiteSeer sample, one local generator, and fixed budgets. They do not establish broad model insecurity or guaranteed semantic preservation.",
    ]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path} and {directory / 'report.md'}")


def validate(config: dict[str, Any], run_id: str) -> None:
    directory = run_directory(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    data, texts, classes, descriptions = load_upstream_data(int(config["seed"]))
    target_records = load_target_records()
    victim = Victim(config, classes, descriptions)
    labels = data.y.tolist()
    sample_ids = stratified_sample(list(range(len(texts))), labels, 30, int(config["seed"]))
    reconstructed = victim.encode_texts(
        [texts[node_id] for node_id in sample_ids], max_length=int(config["node_max_length"])
    )
    stored = data.x[sample_ids].float()
    cosine = torch.nn.functional.cosine_similarity(reconstructed.float(), stored, dim=1)
    test_ids = data.test_mask.nonzero(as_tuple=False).view(-1).tolist()
    graphs = [build_graph(data, target_records, int(node_id)) for node_id in test_ids]
    scores = victim.infer(graphs)
    predictions = scores.argmax(dim=1)
    test_labels = data.y[test_ids]
    clean_accuracy = float((predictions == test_labels).float().mean())
    result = {
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "reconstruction_node_ids": sample_ids,
        "median_feature_cosine": float(cosine.median()),
        "min_feature_cosine": float(cosine.min()),
        "max_absolute_difference": float((reconstructed - stored).abs().max()),
        "clean_test_nodes": len(test_ids),
        "clean_accuracy": clean_accuracy,
        "passed": float(cosine.median()) >= 0.999,
    }
    atomic_json(directory / "validation.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("Feature reconstruction gate failed")


def check_environment() -> None:
    import torch_geometric
    import transformers
    import sentence_transformers

    result = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(result, indent=2))


def download_models(config: dict[str, Any]) -> None:
    from huggingface_hub import snapshot_download

    models = [config["node_encoder_model"], config["semantic_model"], config["generator_model"]]
    ignored = ["*.onnx", "onnx/**", "openvino/**", "*.h5", "*.msgpack", "*.ot", "*.tflite"]
    for model_name in models:
        destination = local_model_path(model_name)
        print(f"Downloading/caching {model_name}", flush=True)
        path = snapshot_download(
            repo_id=model_name,
            local_dir=destination,
            local_dir_use_symlinks=False,
            ignore_patterns=ignored,
            max_workers=4,
        )
        print(f"Cached at {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable local GraphFeedback experiment pipeline")
    parser.add_argument(
        "command",
        choices=[
            "check", "download-models", "prepare-checkpoint", "validate", "select", "generate", "filter", "evaluate",
            "feedback-generate", "feedback-filter", "feedback-evaluate", "feedback-finalize", "aggregate",
        ],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default="demo30")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    if args.command == "check": check_environment()
    elif args.command == "download-models": download_models(config)
    elif args.command == "prepare-checkpoint": prepare_checkpoint()
    elif args.command == "validate": validate(config, args.run_id)
    elif args.command == "select": select_nodes(config, args.config, args.run_id, args.sample_size or int(config["sample_size"]), args.force)
    elif args.command == "generate": generate(config, args.run_id)
    elif args.command == "filter": filter_candidates(config, args.run_id)
    elif args.command == "evaluate": evaluate(config, args.run_id)
    elif args.command == "feedback-generate": feedback_generate(config, args.run_id)
    elif args.command == "feedback-filter": feedback_filter(config, args.run_id)
    elif args.command == "feedback-evaluate": feedback_evaluate(config, args.run_id)
    elif args.command == "feedback-finalize": feedback_finalize(config, args.run_id)
    elif args.command == "aggregate": aggregate(config, args.run_id)


if __name__ == "__main__":
    main()
