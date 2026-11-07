import json
import re
import shutil
import tempfile
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import tools
from agent import agent

HERE = Path(__file__).parent
CLEAN_DATA = HERE/"runs_copy.json"
EVAL_CASES = HERE/"eval_cases.json"
RUN_ID = re.compile(r"\br\d{2}\b")

def reset_data(tmp_dir):
    path = Path(tmp_dir) / "runs.json"
    shutil.copy(CLEAN_DATA, path)
    with open(path) as f:
        data = json.load(f)
    tools.runs = data
    tools.RUNS_PATH = path

def run_case(case):
    config = {'configurable': {'thread_id': case['id']}}

    result = agent.invoke({'messages': [{'role': 'user', 'content': case['question']}]}, config=config)

    while '__interrupt__' in result:
        # The eval cases I wrote is bound to have one archive_run call each. So no case will have multiple requests. That's the reason we dont loop through requests and accumulate the requests
        requests = result['__interrupt__'][0].value.get('action_requests')
        decision = {'type': case['approval'] or 'reject'}
        result = agent.invoke(Command(resume={'decisions': [decision] * len(requests)}), config=config)
    
    return result['messages']

def check_case(case, messages):
    failures = []

    called = [tc["name"] for m in messages if m.type == "ai" for tc in m.tool_calls]
    answer = messages[-1].content
    found_ids = set(RUN_ID.findall(answer))

    for tool in case["expected_tools"]:
        if tool not in called:
            failures.append(f"Expected tool not called: {tool}")
    
    for tool in case["forbidden_tools"]:
        if tool in called:
            failures.append(f"Forbidden tool called: {tool}")
    
    expected_ids = {x for x in case["answer_contains"] if RUN_ID.fullmatch(x)}
    other_strings = [x for x in case["answer_contains"] if not RUN_ID.fullmatch(x)]

    if expected_ids and found_ids != expected_ids:
        failures.append(f"Run IDs in answer {sorted(found_ids)} != expected {sorted(expected_ids)}")

    for s in other_strings:
        if s not in answer:
            failures.append(f"Answer missing: {s}")
    
    for run_id, status in case.get("expected_status", {}).items():
        run = next(r for r in tools.runs if r["run_id"] == run_id)
        if run["status"] != status:
            failures.append(f"{run_id} status is {run['status']}, expected {status}")

    return failures

def main():
    cases = json.loads(EVAL_CASES.read_text())
    passed = 0

    for case in cases:
        with tempfile.TemporaryDirectory() as tmp:
            reset_data(tmp)
            messages = run_case(case)
            failures = check_case(case, messages)
            
            for message in messages:
                print(message)
                print()
        if failures:
            print(f"FAIL  {case['id']}: {'; '.join(failures)}")
        else:
            passed += 1
            print(f"PASS  {case['id']}")
        

    print(f"\nScore: {passed}/{len(cases)}")


if __name__ == "__main__":
    main()
    