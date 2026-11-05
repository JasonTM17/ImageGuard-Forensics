"""EvaluationService — bọc runner, ghi experiment rows vào DB."""

from __future__ import annotations

from pathlib import Path

from imageguard.core.config import ThresholdSet
from imageguard.evaluation.comofod import CoMoFoDAdapter
from imageguard.evaluation.datasets import SynthAdapter
from imageguard.evaluation.runner import EvalReport, run_eval
from imageguard.forensics.registry import load_method
from imageguard.storage.repository import ExperimentRepository


class EvaluationService:
    def __init__(self, cfg: ThresholdSet, experiments: ExperimentRepository | None,
                 reports_dir: Path = Path("reports")) -> None:
        self.cfg = cfg
        self.experiments = experiments
        self.reports_dir = reports_dir

    def run(self, dataset: str, root: Path, split: str,
            method_specs: list[str], seed: int = 42) -> EvalReport:
        adapter = (SynthAdapter(root) if dataset == "synth"
                   else CoMoFoDAdapter(root))
        methods = [load_method(s) for s in method_specs]
        report = run_eval(adapter, methods, self.cfg, split, seed)
        report.save(self.reports_dir)
        if self.experiments is not None:
            self.experiments.save({
                "experiment_id": report.experiment_id,
                "dataset": report.dataset, "split": report.split,
                "provenance": report.provenance,
                "per_method": report.per_method,
            })
        return report
