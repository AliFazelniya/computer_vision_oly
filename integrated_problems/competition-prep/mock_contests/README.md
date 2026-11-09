# Mock contests

`python mock_contests/template_solution.py` runs the starter template against a fresh hidden wall-edge set.

Protocol (60 min): read the task and write the metric (0-5) -> harness + baseline (5-15) -> fix the single biggest failure, one change at a time (15-40)
-> held-out check + fallbacks (40-52) -> freeze, run everything (52-58) -> verify output format (58-60).

Rules: never let an exception lose an item (`safe_method`), never tune on the set you report, log every change with its measured effect.
