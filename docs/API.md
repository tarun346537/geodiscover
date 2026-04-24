# Planner API

Start a local service after installing the `service` profile:

```console
python -m uvicorn geodiscover.service:app --host 127.0.0.1 --port 8000
```

## Endpoints

| Method | Path | Response |
| --- | --- | --- |
| GET | `/health` | JSON service/survey/policy readiness |
| GET | `/metrics` | Prometheus text exposition |
| POST | `/plan` | JSON acquisition decision |
| GET | `/openapi.json` | FastAPI's request/response schema |
| GET | `/docs` | Interactive OpenAPI documentation |
| GET | `/redoc` | Read-only OpenAPI documentation |

## GET /health

```console
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","policy_loaded":false,"solver":"linear_gaussian","configured_survey":false}
```

`solver` is `simpeg_particle_belief` for a configured nonlinear DC prior.
`policy_loaded=false` allows search and fallback decisions to remain available.
Invalid survey configuration or a malformed existing checkpoint prevents
startup. A missing optional checkpoint leaves `policy_loaded=false` and allows
search/fallback decisions. `/health` becomes available after setup completes.

## POST /plan

```console
curl -H "Content-Type: application/json" -H "X-Request-ID: survey-step-7" --data-binary @configs/request-small.json http://127.0.0.1:8000/plan
```

The request schema is `PlanRequest` in `geodiscover/service.py`:

| Field | Type / default | Constraint |
| --- | --- | --- |
| `problem` | Object or omitted | Required unless `GEO_PROBLEM_JSON` configures a fixed survey |
| `selected` | Integer array, `[]` | Unique available indices, at most 128 |
| `observations` | Number array, `[]` | One finite observation per selection |
| `remaining_budget` | Integer, `1` | 1 to 16, bounded by remaining candidates |
| `simulations` | Integer, `128` | 1 to 4096 per primary search |
| `timeout_ms` | Integer, `500` | 10 to 2000 ms |
| `seed` | Integer, `42` | 0 to 2147483647 |
| `strategy` | String, `mcts` | `mcts` or `policy` |

An inline `problem` contains:

| Field | Shape / meaning |
| --- | --- |
| `operator` | Candidate x model-dimension matrix (up to 128 x 64) |
| `noise_std` | Positive observation standard deviation per candidate. A length-one value broadcasts |
| `prior_mean` | Model-dimension vector |
| `prior_covariance` | Symmetric positive-definite model covariance |
| `offset` | Optional candidate-length response offset |

All numerical values must be finite. The inline schema rejects additional keys
including simulator metadata. Inline requests use the Gaussian baseline.
Configured DC requests omit `problem` and condition the server's particle prior.
Observation units must match the configured forward responses.

An ordinary decision has this shape. Values shown here are a schema example:

```json
{
  "action": 1,
  "mode": "mcts",
  "fallback_reason": null,
  "expected_information_gain": 2.3,
  "statistics": {"simulations": 128, "nodes_expanded": 30},
  "provenance": "live_request"
}
```

`action` is an available integer candidate index. `mode` is `mcts`, `policy`,
or `greedy_bayesian_fallback`. Search statistics depend on the backend. Particle
search adds its particle count and observation-bin resolution. The information
gain is a predictive-variance proxy for particle beliefs.

Fallback responses remain HTTP 200 and set `fallback_reason` to `timeout`,
`workers_busy` or `planner_failure`. The last includes an unavailable policy
checkpoint or malformed primary decision. This behavior lets the survey loop
continue while exposing its degraded planning mode.

The service returns `X-Request-ID` on ordinary responses. A supplied identifier
is limited to 128 characters. Otherwise the service assigns one. JSON request
records include the identifier, method, path, status and elapsed time.

## Errors

| Status | Meaning |
| --- | --- |
| 200 | Primary or explicitly labeled fallback decision |
| 422 | Invalid request shape, nonfinite data, nonpositive noise, invalid covariance, duplicate selection or excess budget |
| 404 | Unknown path |
| 405 | Unsupported method on an existing path |
| 500 | Unexpected application failure outside the handled primary-planner path |

Validation errors use FastAPI's `detail` field. Worker failure is handled by the
fallback path rather than returned as a successful primary decision.

## GET /metrics

```console
curl http://127.0.0.1:8000/metrics
```

| Metric | Labels | Meaning |
| --- | --- | --- |
| `geodiscover_plans_total` | `mode` | Completed acquisition decisions |
| `geodiscover_fallbacks_total` | `reason` | Fallback decisions by cause |
| `geodiscover_plan_seconds` | Histogram buckets | Valid planning-request latency |

The latency histogram includes fallback handling. Validation failures happen
before its observation point and remain visible in request records.
`configs/prometheus.yml` scrapes the local service every 15 seconds and loads
three rules: target unavailable for two minutes, p99 above 140 ms for five
minutes, and fallback fraction above 5% for five minutes.

## Runtime settings

Set `.env.example` values in the process environment before startup.
`Settings.load` reads those overrides and can also load a JSON settings file
through Python.

| Variable | Use |
| --- | --- |
| `GEO_PROBLEM_JSON` | Fixed survey JSON, including optional SimPEG metadata |
| `GEO_POLICY_CHECKPOINT` | Portable acquisition-policy checkpoint |
| `GEO_PARTICLES` | Prior ensemble size, default 64 |
| `GEO_WORKERS` | Maximum occupied planning slots, default 2 |
| `GEO_RAY_ADDRESS` | Ray cluster address. Empty uses local search |
| `GEO_REDIS_URL` | Training-cache endpoint through the settings-backed CLI |
| `GEO_LOG_LEVEL` | Application JSON logging level |
| `GEO_OUTPUT_DIR` | Default local result directory |
| `GEO_API_URL` | Default load-test endpoint, overridden by `--url` |
| `GEO_CPU_THREADS` | PyTorch CPU training threads |

For a prepared DC survey, set `GEO_PROBLEM_JSON` to its exported JSON before
starting Uvicorn. The server solves its prior response ensemble during setup.
Its default 64-particle prior is independent of observations supplied by clients.
Change the survey/prior by restarting with a new configuration.

```console
python -m geodiscover.loadtest --url http://127.0.0.1:8000 --payload configs/request-small.json --requests 1000 --concurrency 8 --output outputs/loadtest.json
```

Use a request without an inline `problem` when measuring a configured nonlinear
survey. Load reports include warmup errors, all measured request latencies,
HTTP/transport errors and the fraction of successful decisions using fallback.
