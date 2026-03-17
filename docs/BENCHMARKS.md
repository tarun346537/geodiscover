# Benchmark procedures

## Results and run metadata

`benchmarks/reference.json` contains experiment metrics and units.
The commands below write measured reports for each workload. Record survey/model
IDs, field-release identifier, split definitions, hardware model, software
versions, seed, acquisition budget, particle count, cache resolution and timing
convention with each comparison. Identify the baseline implementation and state
whether accuracy differences use relative percentages or percentage points.

Evaluation reports model MSE and noiseless survey-prediction MSE. Define the
accuracy score before comparing an accuracy decrease, and keep error ratios
separate from accuracy percentages.

## Small local check

```console
make check sample demo benchmark
```

The checked configuration has four independent synthetic worlds, two
acquisitions and 16 MCTS simulations per decision. `outputs/demo.json` records
the actual trial rows and local sequential decision timings. The observed mean
model MSE was 1.0182478 for greedy information gain and 0.9925889 for MCTS.
On this linear workload, the two-budget efficiency grid matched its greedy
reference at two acquisitions.

The test suite passed 64 tests with 80% statement coverage. A warm, sequential
four-candidate API fixture checks p99 against the 140 ms alert threshold.
Use the HTTP load test below to measure deployed service capacity.

## Acquisition counts and measurement efficiency

Prepare a valid DC problem and model file using `data/README.md`. For a survey
with at least eight candidates, a larger export can use:

```console
python -m geodiscover.plans --problem path/to/problem.json --ray --ray-address auto --target-plans 60000 --max-nodes 2500000 --budget 8 --simulations 4096 --particles 128 --max-pending 16 --output outputs/plans.jsonl
```

`--target-plans 60000` counts action-state records. `--max-nodes 2500000` sets the
global expansion ceiling. Read
`outputs/plans.run.json` for actual episode/plan/node/simulation counts, worker
and wall time, forward-cache accesses and the dataset hash. Node expansion and
physical forward solving are counted separately.

```console
python -m geodiscover.benchmark --problem path/to/problem.json --models path/to/models.jsonl --split test --trials 20 --checkpoint outputs/policy.pt --budgets 2,4,6,8 --baseline greedy --baseline-budget 8 --output outputs/efficiency.json
```

The benchmark holds geological models and observation noise fixed across
methods and budgets. For the selected baseline, it finds the smallest evaluated
budget whose mean model MSE meets the baseline-budget mean. Measurement
reduction is `1 - matching_budget / baseline_budget`. A missing match is `null`.
Compare measurement efficiency with the 41% reduction using enough
held-out models and a suitable budget grid. Include confidence intervals and
geographic grouping in the experiment design.

## Search, distillation and policy timing

```console
python -m geodiscover.training --plans outputs/plans.jsonl --problem path/to/problem.json --models path/to/models.jsonl --budget 8 --particles 128 --device cuda --batch-size 256 --epochs 10 --rl-episodes 128 --output outputs/policy.pt
python -m geodiscover.demo --problem path/to/problem.json --models path/to/models.jsonl --split test --trials 20 --checkpoint outputs/policy.pt --budget 8 --particles 128 --output outputs/policy-eval.json
```

Each method receives the same observations at the same budget. The report
includes model/prediction error and p50/p99 decision time. Shared simulator
ensemble setup is recorded separately. Record survey/model IDs, hardware and
timing statistic beside comparisons with the 95 ms policy and 40 s
search values. Specify the accuracy score for the 3% trade-off comparison.

The policy export contains configuration, dataset SHA-256, distillation metrics,
refinement metrics and phase profiles. Training emits `policy.steps.jsonl` and
stores phase state under `policy-state/`. Re-run the same training command with
`--resume`. Keep data, model dimensions and optimizer/validation settings fixed.

## Redis and epoch profiling

Measure the forward-cache layer against a bypassed control:

```console
python -m geodiscover.cache_benchmark --problem path/to/problem.json --models path/to/models.jsonl --split train --repeats 3 --quantization 0.02 --output outputs/cache-benchmark.json
```

Add `--redis-url` to include Redis. Both timed passes solve the same quantized
representatives. Exact-versus-quantized prediction error is measured separately.
The report contains actual hits/misses, elapsed time, response hashes and input
fingerprints. Compare the measured speed ratio with the 6.8× result.
Record cold/warm Redis state before each run. Repeat runs and report dispersion,
accounting for the fixed uncached-first order and solver-internal factorization
reuse in both conditions. Use DC models for solver timing and the matrix fixture
for CLI checks.

Keep the model order, seeds, episode grouping and hardware fixed while comparing
cache conditions. Each condition still has a bounded local cache. A shared
Redis condition can reuse results across workers and local-cache evictions:

```console
python -m geodiscover.training --plans outputs/plans.jsonl --problem path/to/problem.json --models path/to/models.jsonl --budget 8 --device cuda --output outputs/local-policy.pt
python -m geodiscover.training --plans outputs/plans.jsonl --problem path/to/problem.json --models path/to/models.jsonl --budget 8 --device cuda --redis-url redis://127.0.0.1:6379/0 --cache-quantization 0.02 --output outputs/redis-policy.pt
```

Quantization changes forward approximation as well as reuse. Assess its
prediction error independently against exact solves. Record cold/warm cache
state and actual hit/miss/failure counts with the speed comparison.

Declare how many acquisition episodes constitute an RL epoch. Sum their
`profile.wall_seconds` for comparable training-only epoch cost and report cold
initialization separately. Validation and command startup are outside that
per-episode timer. The epoch comparison is 9 hours to 80 minutes,
with a reward-computation ratio of 6.8×.

GPU profiles include CUDA peak allocation and optional device-wide NVML samples.
Install `.[profiling]` and reserve the device for a comparable utilization check.
CPU runs leave utilization unavailable. Record sampling and device context when
assessing the 31% to 74% utilization values.

## Service load and failure behavior

```console
python -m uvicorn geodiscover.service:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```console
python -m geodiscover.loadtest --url http://127.0.0.1:8000 --payload configs/request-small.json --requests 1000 --concurrency 8 --warmup 20 --output outputs/loadtest.json
```

The included request is a four-candidate synthetic matrix. To assess a policy
or DC workload, configure its checkpoint/survey and provide its actual request
payload. The report includes attempted and successful-decision RPS, all-request
p50/p95/p99, HTTP/transport errors, modes and fallback fraction. Warmup requests
are excluded from measured throughput and latency.

Report fallback fraction beside the 310 requests/s and 140 ms p99 comparison.
Fast fallback responses can otherwise change the apparent service throughput.
The controlled tests inject worker failure, timeout and malformed primary
actions, confirming an available integer fallback with HTTP 200.

Prometheus uses `configs/prometheus.yml` and `configs/prometheus-alerts.yml`.
YAML, rule references and emitted metric names were checked locally. Use
`promtool check config configs/prometheus.yml` in a Prometheus installation to
check its parser against the deployment version.
