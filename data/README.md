# Survey and training inputs

Place local datasets, results and model checkpoints under `data/` or `outputs/`.
Both locations are ignored by Git. Generate a small linear-model fixture:

```console
python scripts/generate_sample_data.py --output outputs/samples --models 18
```

It writes `problem.json` and `models.jsonl`. The default 18-model set contains
10 training, four validation and four test records. The forward model in this
example is a linear potential-field model. The DC workflow uses the adapter below.

## Survey JSON

`Problem.from_dict` accepts `operator`, `noise_std`, `prior_mean`,
`prior_covariance`, optional `offset`, and optional `metadata`.
The linear observation equation is `y = operator @ model + offset + noise`.
The matrix has one row per candidate acquisition and one column per model
parameter. Noise standard deviations are strictly positive. The prior
covariance is symmetric positive definite.

Model files contain one JSON object per line:

```json
{"model_id":"world-1","split":"train","model":[0.1,0.2]}
```

The vector length must match the survey's model dimension. Every ID is unique, and
valid split values are `train`, `validation`, and `test`. Training and validation
may share the same input file: each reader selects its designated split.
Exact duplicate geological vectors must share a split even if IDs differ.
Training exports record explicit RL world hashes, so evaluation also rejects
those worlds if they are renamed in a separate held-out file. Use current policy
exports for these checks, and group nearby or correlated worlds geographically
during input preparation.
Choose location/stratigraphy holdouts when preparing field-derived models.

## SimPEG DC configuration

`configs/dc-small.json` is a small synthetic flat-ground DC example:

```console
python -m geodiscover.simpeg_adapter --config configs/dc-small.json --output outputs/dc-problem.json
python -m geodiscover.plans --problem outputs/dc-problem.json --episodes 5 --budget 2 --particles 8 --simulations 16 --output outputs/dc-plans.jsonl
python -m geodiscover.training --plans outputs/dc-plans.jsonl --problem outputs/dc-problem.json --budget 2 --particles 8 --epochs 2 --rl-episodes 2 --output outputs/dc-policy.pt
```

| Config field | Meaning |
| --- | --- |
| `hx`, `hz` | Positive tensor-mesh cell widths in metres |
| `origin` | Lower mesh origin `[x,z]` in metres (z is positive upward) |
| `candidates` | Dipoles with `a`,`b`,`m`,`n` electrode coordinates `[x,z]` |
| Candidate `current` | Optional source current in amperes, default 1 |
| `noise_std` | Voltage standard deviations in volts |
| `reference_conductivity` | Positive conductivity in S/m |
| `prior_std` | Standard deviation of log-conductivity parameters |
| `nky` | 2.5D quadrature setting |
| `boundary_condition` | SimPEG boundary condition, example `Robin` |

Each geological model is natural-log conductivity, ordered consistently with
the tensor mesh's `cell_centers`. The adapter uses `ExpMap`, dipole voltage
receivers and `Simulation2DCellCentered` with sparse LU. Coordinates must lie
within the intended mesh geometry. Prepare topography and measured-survey QC
before constructing the adapter inputs.

Sensitivity export uses central differences around the reference model and
costs `2*n_cells+1` forward solves. Its JSON includes `metadata.simpeg_dc`.
That metadata selects nonlinear simulator observations and particle search
under `--backend auto`. `--backend linear` selects the local Gaussian baseline.
Quantization is disabled during finite-difference construction.

## Teacher JSONL

The exporter writes version 2 records. A scenario header precedes its plans:

```json
{"schema_version":2,"record_type":"scenario","scenario_id":"scenario-42","split":"train","problem":{},"belief_backend":"particle","particle_ensemble":{"models":[],"predictions":[],"weights":[]},"provenance":{}}
{"schema_version":2,"record_type":"plan","scenario_id":"scenario-42","selected":[],"observations":[],"remaining_budget":2,"teacher_action":0,"search":{}}
```

The empty arrays/objects above describe the record shape. Actual headers carry
validated survey arrays and complete ensemble responses. Linear scenarios omit
`particle_ensemble`. The loader also accepts expanded version 1 action records.
It rejects orphan plans, duplicate headers, scenario split leakage and invalid
teacher actions. Survey/ensemble inputs remain fixed within a scenario. Plan
records cannot override header context. In-memory distillation uses the same
checks.

A plan is one acquisition decision at one belief state. An episode is the whole
budgeted sequence. The `.run.json` sidecar records actual plan/episode/node
counts and forward-cache accesses. These counts remain separate because one
solver response vector can support many search branches.

## Evaluation

```console
python -m geodiscover.demo --problem outputs/samples/problem.json --models outputs/samples/models.jsonl --split test --trials 4 --budget 2 --simulations 16 --checkpoint outputs/policy.pt --output outputs/heldout.json
```

All strategies see the same true models and per-candidate observation noise.
Without `--models`, evaluation samples independent worlds from the configured
prior. Record model IDs, survey configuration, public dataset release, hardware,
seed and baseline settings with each experiment.
