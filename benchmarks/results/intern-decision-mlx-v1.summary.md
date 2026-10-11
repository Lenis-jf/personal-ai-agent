# Router benchmark results

Predictions: `/Users/juanfelenis/Documents/Proyectos/personal-agent/benchmarks/results/intern-decision-mlx-v1.jsonl`
Cases: **54** | Gold decisions: **648**

## Primary metrics

| Metric | Score |
|---|---:|
| Skill accuracy | 75.9% |
| Complexity accuracy | 75.9% |
| Model-route accuracy | 88.9% |
| Mean of 3 choice accuracies | 80.2% |
| Mean F1 for tool flags (positive class = yes) | 67.3% |
| Mean F1 for clarification/confirmation flags | 56.4% |
| Overall decision accuracy (all 12 fields) | 86.4% |
| Median per-case latency | 493.7 ms |
| P95 per-case latency | 566.1 ms |

## By language

| Language | Cases | Skill accuracy | Complexity accuracy | Model-route accuracy | Tool F1 (mean) |
|---|---:|---:|---:|---:|---:|
| de | 18 | 77.8% | 72.2% | 88.9% | 69.4% |
| en | 18 | 72.2% | 88.9% | 88.9% | 68.0% |
| es | 18 | 77.8% | 66.7% | 88.9% | 66.7% |

## By field

| Field | Accuracy | Correct / total | F1 for yes |
|---|---:|---:|---:|
| skill | 75.9% | 41/54 | n/a |
| complexity | 75.9% | 41/54 | n/a |
| model_route | 88.9% | 48/54 | n/a |
| tool_file_search | 83.3% | 45/54 | 57.1% |
| tool_file_read | 75.9% | 41/54 | 58.1% |
| tool_file_modify | 90.7% | 49/54 | 54.5% |
| tool_web_search | 88.9% | 48/54 | 80.0% |
| tool_email_draft | 94.4% | 51/54 | 80.0% |
| tool_email_send | 94.4% | 51/54 | 66.7% |
| tool_system_change | 96.3% | 52/54 | 75.0% |
| needs_clarification | 87.0% | 47/54 | 46.2% |
| requires_confirmation | 85.2% | 46/54 | 66.7% |

> Interpretation: this is a small, policy-labelled pilot, not a general model leaderboard. Inspect individual errors before selecting a backend. The `cloud_candidate` label never enables cloud use.
