# Router benchmark results

Predictions: `/Users/juanfelenis/Documents/Proyectos/personal-agent/benchmarks/results/laya-v1.jsonl`
Cases: **54** | Gold decisions: **648**

## Primary metrics

| Metric | Score |
|---|---:|
| Skill accuracy | 64.8% |
| Complexity accuracy | 29.6% |
| Model-route accuracy | 51.9% |
| Mean of 3 choice accuracies | 48.8% |
| Mean F1 for tool flags (positive class = yes) | 45.5% |
| Mean F1 for clarification/confirmation flags | 9.1% |
| Overall decision accuracy (all 12 fields) | 73.3% |
| Median per-case latency | 226.3 ms |
| P95 per-case latency | 392.1 ms |

## By language

| Language | Cases | Skill accuracy | Complexity accuracy | Model-route accuracy | Tool F1 (mean) |
|---|---:|---:|---:|---:|---:|
| de | 18 | 77.8% | 27.8% | 33.3% | 34.9% |
| en | 18 | 72.2% | 27.8% | 61.1% | 47.8% |
| es | 18 | 44.4% | 33.3% | 61.1% | 45.9% |

## By field

| Field | Accuracy | Correct / total | F1 for yes |
|---|---:|---:|---:|
| skill | 64.8% | 35/54 | n/a |
| complexity | 29.6% | 16/54 | n/a |
| model_route | 51.9% | 28/54 | n/a |
| tool_file_search | 72.2% | 39/54 | 48.3% |
| tool_file_read | 75.9% | 41/54 | 48.0% |
| tool_file_modify | 85.2% | 46/54 | 33.3% |
| tool_web_search | 72.2% | 39/54 | 54.5% |
| tool_email_draft | 88.9% | 48/54 | 40.0% |
| tool_email_send | 94.4% | 51/54 | 40.0% |
| tool_system_change | 90.7% | 49/54 | 54.5% |
| needs_clarification | 70.4% | 38/54 | 0.0% |
| requires_confirmation | 83.3% | 45/54 | 18.2% |

> Interpretation: this is a small, policy-labelled pilot, not a general model leaderboard. Inspect individual errors before selecting a backend. The `cloud_candidate` label never enables cloud use.
