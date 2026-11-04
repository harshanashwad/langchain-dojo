import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

import ml_experiment_agent as agent

CLEAN_DATA = Path(__file__).parent.parent / "runs_copy.json"

@pytest.fixture
def fresh_runs(tmp_path, monkeypatch):
    path = tmp_path / "runs.json"
    shutil.copy(CLEAN_DATA, path)
    with open(path) as f:
        data = json.load(f)
    monkeypatch.setattr(agent, "runs", data)
    monkeypatch.setattr(agent, "RUNS_PATH", path)
    return path


def test_search_runs_by_dataset_returns_only_that_dataset(fresh_runs):
    results = agent.search_runs.invoke({"dataset": "cifar10"})

    assert len(results) > 0
    assert all(r["dataset"] == "cifar10" for r in results)


def test_search_runs_by_accuracy_threshold_returns_only_passing_runs(fresh_runs):
    results = agent.search_runs.invoke({
        "metric_thresholds": [{"metric": "accuracy", "threshold": 0.9, "comparison": "gt"}] # Pydantic converts dicts to BaseModel objects first underneath
    })

    assert len(results) > 0
    assert all(r["metrics"]["accuracy"] > 0.9 for r in results)


def test_search_runs_excludes_runs_with_null_metric(fresh_runs):
    null_ids = {r["run_id"] for r in agent.runs if r["metrics"].get("accuracy") is None}
    assert null_ids, "fixture should contain at least one run with null accuracy"

    results = agent.search_runs.invoke({
        "metric_thresholds": [{"metric": "accuracy", "threshold": 0.0, "comparison": "gte"}]
    })

    returned_ids = {r["run_id"] for r in results}
    assert null_ids.isdisjoint(returned_ids)


def test_search_runs_rejects_invalid_comparison(fresh_runs):
    # This test is to check pydantic validating metric_thresholds dict in search_runs correctly
    # The code inside this block must throw a ValidationError. If it does, THEN the test passes
    with pytest.raises(ValidationError):
        agent.search_runs.invoke({
            "metric_thresholds": [{"metric": "accuracy", "threshold": 0.9, "comparison": "eq"}]
        })

def test_get_run_id_not_found(fresh_runs):
    # Check if we get the error dict saying run_id not found for an non existent run id
    result = agent.get_run.invoke({'run_id': '-1'})
    assert 'error' in result and result['error'] == 'No run with run_id: -1 is found'

def test_archive_persists_to_file(fresh_runs):
    agent.archive_run.invoke({"run_id": "r01"})

    with open(fresh_runs) as f:
        saved = json.load(f)
    r01 = next(r for r in saved if r["run_id"] == "r01")
    assert r01["status"] == "archived"

def test_archive_run_already_archived(fresh_runs):
    # Check we get already archived error if we try to archive a run that's already archived
    test_run_id = 'r01'

    # This confirms r01 is archived for the first time
    result = agent.archive_run.invoke({"run_id": test_run_id})
    assert result == f'Run with run_id: {test_run_id} has been successfully archived'
    
    run = next((r for r in agent.runs if r['run_id'] == test_run_id), None)
    assert run['status'] == 'archived'
    
    # Attempt archiving a second tme    
    result = agent.archive_run.invoke({"run_id": test_run_id})
    assert result == f'Run with run_id: {test_run_id} is already archived'
