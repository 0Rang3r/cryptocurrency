import os
import csv
import json
from typing import List, Dict
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "AUC"]


class ExperimentReporter:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.results: List[Dict] = []
        os.makedirs(results_dir, exist_ok=True)

    def add(self, result: Dict) -> None:
        self.results.append(result)

    def print_summary(self) -> None:
        if not self.results:
            logger.warning("暂无结果可展示")
            return

        header = f"{'Model':<30} | " + " | ".join(f"{m:<9}" for m in METRIC_COLUMNS)
        separator = "-" * len(header)

        logger.info(f"\n{'实验结果汇总':^{len(header)}}")
        logger.info(separator)
        logger.info(header)
        logger.info(separator)

        for res in self.results:
            model_name = res.get("model", "Unknown")
            row = f"{model_name:<30} | "
            row += " | ".join(f"{res.get(m, 0.0):.4f}   " for m in METRIC_COLUMNS)
            logger.info(row)

        logger.info(separator)

        best = max(self.results, key=lambda x: x.get("F1", 0.0))
        logger.info(
            f"最优模型: {best['model']} | F1={best['F1']:.4f} | AUC={best['AUC']:.4f}"
        )

    def save(self, tag: str = "") -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{tag}" if tag else ""

        csv_path = os.path.join(self.results_dir, f"results{suffix}_{timestamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["model"] + METRIC_COLUMNS
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in self.results:
                writer.writerow({k: res.get(k, "") for k in fieldnames})
        logger.info(f"结果已保存到 CSV: {csv_path}")

        json_path = os.path.join(self.results_dir, f"results{suffix}_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到 JSON: {json_path}")