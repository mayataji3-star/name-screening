# Project Brief — Bilingual Name Screening Service (NLP + LLM)

**Owner:** _(trainee)_
**Mentor:** Mohammad Yaghmour
**Suggested duration:** 3 weeks (part-time), with 3 checkpoints
**Repo folder:** `RedKeys/Name_Screening` (reference implementation) → you build in a **new folder** you own

> **Scope note, read it now:** this project has **no database**. No Oracle, no Neo4j, no Postgres, no Docker, no cloud
> anything. Files on disk only: CSV in, CSV/JSON out. The one exception you may reach for — and only if you actually
> need it — is a single **SQLite** file (stdlib `sqlite3`) for the analyst feedback queue. Nothing else.
> The point of this project is your **NLP and LLM** work. Every hour spent on infrastructure is an hour spent on the
> wrong thing.

---

## 1. Why this exists (read this first, twice)

Every regulated bank has to answer one question, thousands of times a day:

> *"Is this customer — or this counterparty on this wire — the same person as someone on a sanctions, PEP, or adverse-media list?"*

If the answer is **yes and the bank misses it**, that is a sanctions breach: fines, licence risk, regulator consent orders. If the answer is **no but the system says yes**, an analyst spends 20 minutes clearing an innocent customer — and a mid-size bank generates thousands of those a month. Industry false-positive rates on name screening routinely sit **above 90%**. That is the real business problem: not "can we match names", but **"can we match names well enough that humans only look at the cases that deserve a human"**.

Now add the part that makes it genuinely hard in our region:

| Problem | Example |
|---|---|
| Two scripts | `طارق الهاشمي` vs `Tariq Al-Hashimi` — same person, zero character overlap |
| No standard transliteration | Tariq / Tarek / Tareq / Tarik / Tarrek |
| Definite articles & particles | Al-Hashimi / AlHashimi / El Hashimi / Hashimi |
| Kunya & nicknames | `Abu Ali` is a real-world alias, not a name |
| Name order & length | 4-part Arabic names vs 2-part passport names, father/grandfather names dropped |
| Missing identifiers | DOB and nationality are often blank on the list side (lots of `UNKNOWN` in the mock data) |
| Common names | Thousands of `Mohammad Ahmad` — a name match alone means almost nothing |

Pure string matching (Levenshtein, Soundex) breaks on Arabic. Pure embeddings are fuzzy and unexplainable. Pure LLM is slow, expensive, non-deterministic, and unauditable. **The interesting engineering is in combining them and being able to defend every decision to an auditor.**

That is your project.

---

## 2. What you are building

A **name screening service** that takes a person (name + whatever identifiers exist) and returns a ranked, explained, auditable decision against a watchlist.

```
                 ┌──────────────────────────────────────────────┐
  name +  ─────► │ 1. Normalize   (deterministic, cheap)        │
  DOB +          │ 2. Retrieve    (NLP: fuzzy + phonetic + emb) │  top ~20 candidates
  country        │ 3. Adjudicate  (LLM via Groq, top-N only)    │  same-person score + reason
                 │ 4. Decide      (thresholds / bands)          │  AUTO_CLEAR | REVIEW | HIT
                 │ 5. Log         (audit trail, append-only)    │
                 └──────────────────────────────────────────────┘
                                     ▼
                         UI: screen one · screen a batch · work the alert queue
```

**Where everything lives — the storage design, decided for you so you don't have to think about it:**

| Thing | Storage | Why |
|---|---|---|
| Watchlist (a few hundred rows) | CSV → pandas DataFrame in memory | It's small. A database buys you nothing. |
| Vector index | local FAISS file — or plain numpy, since a brute-force cosine scan over a few hundred rows is genuinely fine | Rebuildable in seconds |
| Audit log | append-only JSONL file | Trivially inspectable, naturally append-only |
| Eval set & results | CSV | Diffable in git |
| Analyst dispositions | in-memory, or **one SQLite file** if you want them to survive a restart | The only place a DB is even arguably justified |

If you catch yourself designing a schema, migrations, or a connection pool — stop, you've drifted off the task.

**Non-negotiable design rule:** the LLM is an *advisor*, never the sole decider. Deterministic layers must be able to explain and reproduce the outcome, and hard contradictions (e.g. confirmed DOBs 20 years apart) must be able to override the LLM. An auditor will ask "why did this clear?" — "the model said so" is not an answer.

---

## 3. What already exists (reference only — do not modify it)

`RedKeys/Name_Screening` is an earlier MVP. **Read it, run it, learn from it, then build your own.** Treat it as a colleague's prototype you're reviewing, not as a base to patch.

- `src/name_screening/` — MVP: FAISS + `intfloat/multilingual-e5-base` embeddings, weighted field scoring, FastAPI, CLI, JSONL audit log, small web UI in `web/`
- `sota_screening/` — a second iteration: candidate generation → conflict-aware reranker → **local** Qwen2.5-1.5B LLM judge → decision bands (`AUTO_CLEAR` / `REVIEW` / `AUTO_HIT`) → calibration + benchmark modules
- `data/*.csv` — mock bilingual watchlist, alias map, eval cases
- `tests/` — 7 passing tests worth reading (`test_normalization.py` especially)

**Ignore `src/name_screening/neo4j_import.py` completely.** It's a leftover graph-database importer from a different experiment, it is not part of your task, and it needs packages that aren't installed. Side effect you will hit: `python -m name_screening.cli` imports it at startup and dies with `ModuleNotFoundError: neo4j`. Either `pip install neo4j python-dotenv` to make the reference CLI start, or skip the CLI and use the API. Do not build on it, and do not add a database to your own version.

**Deliverable 0 (day 1–2):** a one-page written critique. What does the MVP do well? Where would it produce false positives / false negatives? Which parts would you keep? Come to the first checkpoint ready to argue your answers.

Worth stealing conceptually (not copy-pasting blindly): the `[NAME] … [DOB] … [RESIDENCY]` structured passage format for embeddings, the decision-band idea, the conflict cap, the "LLM only for borderline scores" gate, the JSONL audit log.

---

## 4. Your build — phase by phase

### Phase 1 — Data & normalization (deterministic foundation)
- Watchlist: start from the mock CSVs in `data/` (`watchlist_seed.csv`, `watchlist_mock.csv`, `alias_map_mock.csv`). **Everything here is synthetic — there is no customer data in this repo and none is coming.** If you want more volume, generate it yourself, or take a **small flat-file slice** (a few hundred rows) of public [OpenSanctions](https://www.opensanctions.org/datasets/) data — still just a file on disk, still no database.
- Build a normalizer: Unicode NFKC, strip Arabic diacritics/tatweel, unify `أ إ آ → ا`, `ة → ه`, `ى → ي`, casefold Latin, strip punctuation, handle `Al-/El-/Bin/Ibn/Abu/Abd al-` particles, collapse whitespace.
- Build a transliteration/phonetic key so Arabic and Latin forms of the same name land in the same bucket (the MVP has a naive version in `normalization.py` — you can do much better; look at Soundex/Double-Metaphone/Beider-Morse and at ICU transliteration).
- **Write unit tests as you go.** Every tricky pair you discover becomes a test case. This is the artifact that proves you understand the domain.

### Phase 2 — Candidate retrieval (the NLP layer)
Recall matters more than precision here — a candidate you never retrieve can never be matched.
- At least **two independent recall channels**, then union:
  - lexical/fuzzy (e.g. RapidFuzz `token_set_ratio`, n-gram or phonetic blocking)
  - semantic (multilingual sentence embeddings — `intfloat/multilingual-e5-base` is already cached on the machine and puts Arabic and English in one vector space)
- Search **aliases as first-class entries**, not just primary names.
- Report **recall@20** on your eval set. At this data size you can afford to run a brute-force scan alongside your indexed version — do it, and understand where they differ.

### Phase 3 — LLM adjudication via **Groq** ← *this is the heart of the project*
- Get your own key at [console.groq.com](https://console.groq.com). Put it in a **`.env` file that is git-ignored**; read it with `os.environ` / `python-dotenv`. **Never hardcode a key, never commit one, never paste one into Slack or a screenshot.** If you ever leak one, revoke it immediately and say so — no drama, just speed.
- Groq is an inference provider for open models (Llama, Qwen, GPT-OSS families) and is *fast*, which matters when you're adjudicating 20 candidates per query. Check the current model list in the console and pick one that supports **JSON / structured output**; document why you chose it.
- Send the LLM **only the top-N candidates** (5–10), one pair at a time or in a small batch — never the whole watchlist. Cost and latency are part of the design.
- Prompt contract:
  - system prompt states the task (same real-world person? yes/no + confidence), the domain rules (transliteration variance is normal; missing fields are *not* evidence of mismatch; contradictiSAons reduce the score), and a strict JSON schema
  - `temperature=0` for reproducibility
  - required output: `{"same_person_score": 0.0-1.0, "verdict": "MATCH|POSSIBLE|NO_MATCH", "reason": "<one grounded sentence>", "signals": {"name": "...", "dob": "...", "geo": "..."}}`
  - validate with Pydantic; retry once on malformed JSON; on repeated failure **fall back to the deterministic score and log that the LLM was unavailable** — the service must never break because Groq had a bad minute
- Engineering hygiene: timeout, exponential backoff on 429, and a simple cache keyed by `(query_hash, candidate_id)` — a JSON file on disk or `functools.lru_cache` is plenty — so re-running the same batch is free. Log tokens and latency per call.
- **Treat the prompt as code you iterate on and measure**, not as a magic string you write once. Version it (`prompt_v1`, `v2`…), record which version produced which result in the audit log, and show in your report what each change actually bought you.
- Reference for prompt shape and robust JSON parsing: `sota_screening/qwen/judge.py`.

### Phase 4 — Decision policy, explainability, audit
- Three bands, thresholds in a config file, not hardcoded in logic: `AUTO_CLEAR` / `REVIEW` / `HIT`.
- Blend: the deterministic score is the backbone; the LLM adjusts it within a bounded range (the MVP uses a 25% blend weight, and only for mid-band scores — a defensible pattern).
- **Hard-conflict override:** a confirmed DOB or nationality mismatch caps the score regardless of what the LLM says.
- Every screening appends one JSON line to an **audit file**: timestamp, input, list version, candidates with each sub-score, model + prompt version, final band, and the human disposition if one is made. Assume a regulator reads it two years from now.
- The explanation shown to the analyst must be **field-grounded** ("same surname, DOB differs by 4 years, alias matched"), never just a number.

### Phase 5 — The UI
Keep it simple and useful. **Streamlit is already installed** on the machine and is the fastest path; FastAPI + a small HTML/JS page (like `web/`) is equally acceptable — your call, justify it in the README.

Minimum:
1. **Screen one** — form (name, DOB, nationality, gender, aliases) → ranked candidates with score breakdown, LLM reason, band, colour-coded
2. **Screen a batch** — upload a CSV, get a results table + download
3. **Alert queue** — the `REVIEW` items, where an analyst clicks **True match / False positive / Unsure** with a comment (keep it in session state, or one SQLite file if you want it to survive a restart — that's the feedback data for a stretch goal)
4. Small config panel: thresholds, top-K, LLM on/off toggle

RTL rendering for Arabic must actually look right. Test with `طارق الهاشمي`.

### Phase 6 — Evaluation (this is what makes it credible)
- Build a labelled eval set of **at least 100 pairs** you construct yourself: true matches across scripts, transliteration variants, nickname/kunya cases, common-name near-misses that must NOT match, and DOB-conflict cases. `data/eval_cases.csv` shows the shape, not the target size.
- Report, at each threshold: **precision, recall, F1, false-positive rate, and recall@K of the retrieval stage**, plus **median latency and estimated cost per 1,000 screenings**.
- Produce a **threshold curve** and recommend an operating point — then defend it: in AML a false negative is a regulatory failure and a false positive is an operations cost. They are not equally bad. Say which way you tuned and why.
- Ablation table — this is the money slide: deterministic only → +embeddings → +LLM. Show what each layer actually buys. If the LLM adds nothing, **say so**; that's a real finding, not a failure.

---

## 5. Definition of done

- [ ] `README.md` — what it is, how to run it in under 5 minutes on a clean machine, architecture diagram, design decisions **and the trade-offs you rejected**
- [ ] `requirements.txt` with **pinned versions**, and a `.env.example` (no real key)
- [ ] Runs end-to-end from a clean clone with **zero infrastructure**: `pip install` → run → screen a name → see a decision. No service to stand up, no database to create.
- [ ] Tests: normalization + scoring + decision policy + a mocked-Groq test (no test may require a live API key)
- [ ] Evaluation report with the metrics and ablation above, committed as markdown
- [ ] UI demo: single, batch, queue
- [ ] A 15-minute walkthrough to the team where you demo it and take questions

## 6. Stretch goals — all NLP/LLM, pick one or two, only after "done"
- **Prompt ablation:** zero-shot vs few-shot vs reason-then-JSON. Which wins on *your* eval set, and by how much?
- **Model comparison on Groq:** a small fast model vs a large one — measure the accuracy gap against the latency and cost gap, then make a recommendation.
- **Self-consistency:** sample the judge 3× on borderline pairs and vote. Does agreement correlate with correctness? If it does, you've built a confidence signal.
- **Embedding comparison:** `multilingual-e5-base` vs LaBSE vs something smaller — recall@20 and index build time.
- **LLM as test-case generator:** have it produce hard transliteration variants and adversarial near-misses, verify them by hand, fold the good ones into your eval set.
- **Calibration:** when the model says 0.8, is it right 80% of the time? Plot it.
- **Nickname/kunya handling** and gender inference from Arabic name morphology.
- **Fuzzy DOB logic** — year-only, off-by-one-digit, swapped day/month. Extremely common in real data.

---

## 7. Ground rules

1. **No database.** Files on disk. SQLite only for the feedback queue, only if you need persistence. If you think you need more, come and argue it — you probably don't.
2. **Do not modify `src/name_screening/` or `sota_screening/`.** Work in your own folder, on your own branch.
3. **Synthetic data only.** Everything you've been given is mock and you will not be given a client set. Nothing customer-derived goes to Groq or any third-party API — that habit matters more than this project does.
4. **No secrets in git.** Key in `.env`, `.env` in `.gitignore`, `.env.example` committed.
5. Small commits with real messages. Push daily so blockers are visible.
6. **Stuck for more than 2 hours? Ask.** Silence is the only wrong move here.
7. Write down assumptions as you make them. Half of AML engineering is documented judgement calls.

## 8. Checkpoints
- **End of week 1** — critique of the MVP, working normalizer + retrieval, eval set drafted
- **End of week 2** — Groq adjudicator + decision policy wired, first metrics
- **End of week 3** — UI, evaluation report, demo

---

## 9. Environment gotchas on this machine (save yourself half a day)

- **Use a virtual environment.** `python -m venv .venv` — do not install into the global one.
- `torch` + `sentence-transformers` are a **~2.5 GB** install. The first model download (`multilingual-e5-base`, ~1.1 GB) also takes a while — but it's already in the HuggingFace cache on this machine, so reuse it.
- Global Python here is **3.14**; some ML wheels lag new Python releases. If `pip install` fights you, build your venv on 3.11/3.12.
- **Always run from the `Name_Screening/` folder.** Config paths are relative to the working directory — running from elsewhere silently creates a *second* `data/` and `artifacts/` tree.
- Arabic output can crash the Windows console with `UnicodeEncodeError` (cp1252). Set `PYTHONUTF8=1`, or run `python -X utf8`.
- The reference `sota_screening` defaults to `use_qwen_judge: True`, which downloads **Qwen2.5-1.5B (~3 GB)** and runs it on CPU — slow. Set it to `False` while exploring; your LLM layer runs on Groq anyway.
- `sota_screening` tests need `src` on the path: `set PYTHONPATH=src` before `pytest`.
- The reference `requirements.txt` is incomplete (`neo4j` and `python-dotenv` are imported but not listed) and unpinned. Yours should not repeat that.
- `artifacts/` isn't in the repo — it's generated on first run. Note the trap: if `artifacts/` already exists, the MVP loads *that* index instead of rebuilding from your CSV. Delete it whenever you change the watchlist, or you'll spend an afternoon debugging ghosts.
- `data/watchlist_neo4j.csv` isn't in the repo either; the loader writes a fresh mock file on first run. Don't be surprised when it appears — the name is historical, there's no graph database behind it.

---

**One last thing.** This is not a toy exercise with a known answer — false-positive reduction in name screening is an open, expensive problem that vendors sell for six figures. Build it as if it will be shown to a client, because it might be. Put your name on the README.
