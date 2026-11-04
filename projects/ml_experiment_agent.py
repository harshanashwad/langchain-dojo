from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import Annotated, Literal
from pydantic import BaseModel, Field
import operator

import json
from pathlib import Path

RUNS_PATH = Path(__file__).parent / "runs.json"

OPS = {
            "gt": operator.gt,
            "gte": operator.ge,
            "lt": operator.lt,
            "lte": operator.le,
        }

SUMMARY_KEYS = ("run_id", "name", "dataset", "model", "status", "metrics")


with open(RUNS_PATH) as f:
    runs = json.load(f)  # list of dicts

# Schema for metric_threshold

class MetricThreshold(BaseModel):
    metric: Literal['accuracy', 'f1', 'loss'] = Field(description='The selected metric')
    threshold: float = Field(description='Value of the metric to compare. Accuracy and f1 should be between 0 and 1')
    comparison: Literal['gt', 'gte', 'lt', 'lte'] = Field(description='The comparison operator. gt is greater than, gte is greater than or equal to, lt is less than, lte is less than or equal to')


def matches(run: dict, dataset: str | None, model: str | None, metric_thresholds: list[MetricThreshold] | None) -> bool:
    if dataset is not None and run['dataset'] != dataset:
        return False
    if model is not None and run['model'] != model:
        return False
    if metric_thresholds is not None:
        for t in metric_thresholds:
            # Get the metric and check for null values before comparison
            metric = run['metrics'].get(t.metric)
            if metric is None:
                return False
            if not OPS[t.comparison](run['metrics'][t.metric], t.threshold):
                return False
    
    return True


@tool
def search_runs(dataset: Annotated[Literal['imdb', 'cifar10', 'mnist', 'fraud-tx'] | None, 'The dataset to filter'] = None,
                model: Annotated[Literal['resnet18', 'resnet50', 'vit-small', 'logreg'] | None, 'The model to filter'] = None, 
                metric_thresholds: Annotated[list[MetricThreshold] | None, 'All threshold conditions must pass'] = None ) -> list[dict]:
    '''
    Find runs by dataset, model and a list of metric thresholds. One dataset, one model and a list of metric thresholds per call (all are optional). For multiple datasets or models, call this tool once per value. It returns run summaries without notes; use get_run for full details.
    '''
    
    results = []
    for run in runs:
        if matches(run, dataset, model, metric_thresholds):
            summary = {k: run[k] for k in SUMMARY_KEYS}
            results.append(summary)
    
    return results

results = search_runs.invoke({"metric_thresholds": [{"metric": "accuracy", "threshold": 0.9, "comparison": "lt"}]})

for r in results:
    print(r)
    print()
    print()