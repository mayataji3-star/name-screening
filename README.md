# Bilingual Name Screening MVP

Arabic/English intelligent name screening MVP using multilingual embeddings (`intfloat/multilingual-e5-base`) and FAISS semantic retrieval.

## Setup

```bash
python -m pip install -r Name_Screening/requirements.txt
```

## Project Structure

- `src/name_screening/`: MVP package (config, normalization, indexing, screening, API, CLI)
- `data/watchlist_seed.csv`: seeded bilingual sanctions dataset
- `data/eval_cases.csv`: evaluation scenarios
- `artifacts/`: generated index, metadata, audit, and eval reports
- `tests/`: unit/integration tests

## CLI Usage

Run from repository root:

```bash
python -m name_screening.cli rebuild-index
python -m name_screening.cli screen --name "Tariq Al-Hashimi" --dob "1985-04-12" --nationality "Yemen"
python -m name_screening.cli evaluate
```

For fast local rebuild while testing very large OpenSanctions files:

```bash
python -m name_screening.cli rebuild-index --max-records 10000
```

If Python cannot find the package, set:

```bash
set PYTHONPATH=Name_Screening\src
```

## API Usage

```bash
set PYTHONPATH=Name_Screening\src
uvicorn name_screening.api:app --reload
```

Endpoints:

- `GET /health`
- `POST /screen`
- `POST /screen/batch`
- `POST /index/rebuild`

UI routes:

- `GET /` (web UI)
- `GET /web/*` (static assets)

Example request:

```json
{
  "name": "طارق الهاشمي",
  "dob": "1985-04-12",
  "nationality": "اليمن",
  "top_k": 3
}
```

## Evaluation

Evaluation reads `data/eval_cases.csv`, runs screening, and writes:

- `artifacts/evaluation_results.csv`
- `artifacts/evaluation_report.md`

Metrics include precision, recall, F1, and confusion counts.

## Web UI

Run API:

```bash
python Name_Screening/run_api.py
```

Then open:

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

The page provides:

- single screening form (`/screen`)
- batch JSON screening (`/screen/batch`)
- health check (`/health`)
- index rebuild (`/index/rebuild`)

## Dataset Scale

`load_watchlist()` expands the seed file deterministically to a larger in-memory watchlist
(default target: 500 records) so MVP testing reflects higher-volume behavior while keeping
the seed CSV easy to maintain.

## OpenSanctions Integration

- Place OpenSanctions file at: `Name_Screening/data/entities.ftm.json`
- Source: [OpenSanctions PEP datasets](https://www.opensanctions.org/datasets/peps/)
- The loader uses memory-safe streaming ingestion and does **not** read the entire file at once.
- Ingestion scope is configured to people-only by default.
- If `entities.ftm.json` is present, it is preferred over the seed CSV for indexing.
- If index artifacts are missing, API startup auto-builds from the available source.

### First-run expectation

First index build can take noticeable time depending on `--max-records` and CPU. For iterative development, use `rebuild-index --max-records <N>`.

## Audit Logging

Each screening decision is appended to:

- `artifacts/audit_log.jsonl`

The log contains timestamp, request context, top score, threshold, decision, and matches.
