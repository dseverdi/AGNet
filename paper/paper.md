# paper.md — working notebook for paper.tex revisions

Temporary memory: where we are, what's on disk, what's pending, and what the
numbers say. Updated as data lands. Distinct from `REVIEW.md` (the original
reviewer-style audit) and from the plan file in `~/.claude/plans/`.

---

## Status (2026-06-02)

| Phase | Status | Notes |
|---|---|---|
| Phase A — code + text edits | ✅ done | `--disable-ptr-emb` wired; `build_classical_baselines` / `build_encoder_linear_probe` / `build_multi_seed_summary` added; paper text edits applied. |
| Phase B — GPU runs | 🔄 partial | Steps 1, 3, 5, 6, 7, 8 done. **Step 2 (multiseed_train) running in tmux now.** Step 4 (multiseed_eval) queued after Step 2. |
| Phase C — paper integration | 🔄 in progress | Starting now with data already on disk. Multi-seed bars added later when Steps 2+4 finish. |

---

## What's on disk

### Code

- `set_predictor.py` — `__init__` takes `disable_ptr_emb: bool`; `forward()` zeros `ptr_emb` when set. Architecture and parameter count are unchanged (verified: 464,001 params either way).
- `train_set_predictor.py` — `--disable-ptr-emb` CLI flag (line 82); flag is saved in checkpoint's `args` dict via `vars(args)`. Prints `[setpred] coordinate-only ablation: ptr_emb masked to zeros` when active.
- `paper/scripts/build_paper_data.py` — added `build_classical_baselines`, `build_encoder_linear_probe`, `build_multi_seed_summary`. `_load_setpredictor(device, ckpt_path)` reads `disable_ptr_emb` from checkpoint. `build_per_polygon_all` accepts `SP_CKPT` / `SP_SUFFIX` env vars for alt checkpoints.
- `paper/scripts/run_revisions.sh` — Phase B runbook. **Auto-logs every step to `paper/logs/<step>_<timestamp>.log`** (added 2026-06-02). 8 named steps + `run_step` wrapper.

### Checkpoints

| Path | Status |
|---|---|
| `checkpoints/set_predictor/standard/set_predictor_best.pt` | original full probe (seed=1234, ptr_emb on) |
| `checkpoints/set_predictor/no_encoder/set_predictor_best.pt` | **Step 1 output** — ablation (seed=1234, ptr_emb masked) |
| `checkpoints/set_predictor/seed{11,22,33}/set_predictor_best.pt` | Step 2 outputs (pending) |

### Paper data (`paper/data/*.json`)

| File | Source | Status |
|---|---|---|
| `dist_dev_test.json` (367) | full probe | ✅ pre-existing |
| `dist_test_OOD.json` (2107) | full probe | ✅ pre-existing |
| `dist_dev_test_noenc.json` (367) | Step 3 | ✅ written 2026-06-02 |
| `dist_test_OOD_noenc.json` (2107) | Step 3 | ✅ written 2026-06-02 |
| `baseline_classical.json` | Step 5 | ✅ written 2026-06-02 |
| `encoder_linear_probe.json` | Step 6 | ✅ written 2026-06-02 |
| `worked_examples.json` | Step 8, **t=0.20** | ✅ regenerated 2026-06-02 |
| `multi_seed_summary.json` | Step 7 | ⏳ empty (no seeds yet); will refill after Step 4 |
| `dist_{dev_test,test_OOD}_seed{11,22,33}.json` | Step 4 | ⏳ pending |

---

## Key numerical results

### 1. No-encoder ablation (C1 evidence — the strongest new result)

**Architecture and parameter count identical**, only `ptr_emb` is zeroed. Probe sees `(xy, in_S)` only.

Headline comparison at matched cardinality on the **OOD split** (2107 polygons):

| Variant | threshold | mean cov | min cov | \|S\|/n | \|S\|/OPT | #<0.95 | #<0.99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Policy seed (no probe) | — | 0.9568 | 0.049 | 0.181 | 1.149 | 310 | 1842 |
| Full probe | 0.20 | 0.9976 | 0.765 | 0.424 | 2.753 | **3** | 89 |
| Full probe | 0.30 | 0.9921 | 0.765 | 0.298 | 1.918 | 19 | 537 |
| **No-encoder probe** | 0.20 | 0.9765 | 0.049 | 0.283 | 1.810 | **81** | 1206 |
| **No-encoder probe** | 0.30 | 0.9632 | 0.049 | 0.194 | 1.241 | 252 | 1733 |

**Take:** at matched `|S|/n ≈ 0.28–0.30`, the full probe drops OOD failures from 19 → 3 (more aggressive op-point) while the no-encoder probe is stuck at **81 failures even at its most aggressive operating point**. The encoder closes failures by **~27× at matched threshold** and by **~4× at matched cardinality**.

On **dev_test** (367 polygons) the gap is smaller but consistent:

| Variant | threshold | mean cov | \|S\|/n | #<0.95 | #<0.99 |
|---|---:|---:|---:|---:|---:|
| Full probe | 0.20 | 0.9993 | 0.438 | 0 | 4 |
| Full probe | 0.30 | 0.9979 | 0.335 | 0 | 12 |
| No-encoder probe | 0.20 | 0.9952 | 0.398 | 0 | 57 |
| No-encoder probe | 0.30 | 0.9800 | 0.209 | 38 | 206 |

The min-cov column is the cleanest signal of representation power: full probe's worst-case OOD coverage is `0.765` regardless of threshold; no-encoder's worst-case is `0.049` (i.e. it inherits the policy seed's catastrophic-failure tail).

### 2. Encoder linear probe (m2 — replaces "visible separation")

GroupKFold(5) over 200 dev_test polygons (12,268 vertex embeddings, 1985 positives / 10283 negatives). Logistic regression on frozen 128-d encoder embeddings, binary LS-target labels.

- **ROC-AUC = 0.8419 ± 0.0242**
- **PR-AUC = 0.5798 ± 0.0475** (positives are ~16%, so PR baseline is 0.16)

Per-fold AUCs: 0.838, 0.873, 0.849, 0.799, 0.851.

**Take:** the per-vertex linear separability of LS-targets in encoder space is `0.842` AUC — well above chance, well above any rhetorical "visible separation" claim. Replaces the qualitative PCA-scatter argument in §7.

### 3. Classical baselines (M4 anchor)

| Method | Split | n | mean cov | \|S\|/n | \|S\|/OPT | #<0.95 |
|---|---|---:|---:|---:|---:|---:|
| Greedy | dev_test | 367 | 0.9994 | 0.152 | 1.009 | 0 |
| Local search | dev_test | 367 | 0.9948 | 0.137 | 0.884\* | 0 |
| Greedy | OOD (partial) | 40 | 0.9993 | 0.166 | 1.052 | 0 |
| Local search | OOD | 2107 | 0.9724 | 0.131 | 0.836\* | 188 |

\* LS `|S|/OPT < 1` because the reference OPT is a time-limited B&B lower bound, not the true optimum. Will report `|S|/n` as the primary cardinality metric for LS in the table to avoid this artifact.

**Take:** classical anchors confirm what we already say in prose — greedy/LS win at `|S|/n` (~0.15) while the probe trades that for feasibility (~0.30 at t=0.30, ~0.44 at t=0.20). The "competitive in cardinality" sentence in §6.1 / §7 now has a numerical anchor.

OOD greedy is only partial (40/2107 polygons present in cached files); honest framing: "where comparable, classical greedy reaches cov ≈ 1.0 at `|S|/n ≈ 0.17`". Full OOD greedy recomputation is `~3h CGAL` if needed; deferred.

---

## Diagnostics from Step 1 (ablation training log)

- Train loss dropped monotonically `0.69 → 0.62` over 60 epochs; no collapse.
- Best checkpoint saved at **ep 41** (`t=0.6, Δ|S|/n=0.0005`) — model plateaued.
- Validation `t=0.5` cov hovered around `0.972` for the full run — close to the policy seed's `0.970`, far from the full probe's `0.998`. **The probe-without-encoder is barely doing more than the policy seed.**
- `t=0.9` row is noisy/collapsing because at high thresholds the no-encoder probe drops too many vertices indiscriminately — exactly the failure mode predicted when the probe has nothing to differentiate vertices by geometric importance.

---

## Phase C integration — progress

### tab_headline (5 rows, derived from per-polygon JSONs)

✅ Done. `paper/scripts/build_tables.py:tab_headline` rewritten:
- Loads from `dist_dev_test.json` + `dist_dev_test_noenc.json` + `baseline_classical.json` (consistent aggregation pipeline; no more dual-source numbers).
- Two grouped blocks: \emph{Classical baselines} (greedy, LS) and \emph{Learned policy / probe} (seed, full probe, no-encoder ablation).
- Columns: Mean cov, #cov≥0.95/N (Wilson CI), Mean |S|/n, Mean |S|/OPT.
- LS `|S|/OPT < 1` is marked with `$^{\dagger}$` and explained in the caption.
- Output: [paper/tables/tab_headline.tex](paper/tables/tab_headline.tex), 5 rows + 2 group headers.

Pending: after Step 4 finishes, decorate the **SetPredictor full** row with ±std over seeds {1234, 11, 22, 33}.

Helpers added: `_agg_per_poly()` aggregates a per-polygon list at a method key; `_row()` renders one LaTeX row with Wilson CI + optional bold/dagger.

### Paper prose edits

✅ All in `paper/paper.tex`:
- **§6.1** (around tab_headline): caption rewritten to explain the 5 rows + LS dagger; opening paragraph now says "third row of Table…" and anchors "competitive in cardinality" to greedy's `|S|/n = 0.152` and LS's `0.137` vs the policy seed's `0.166`.
- **§6.1b NEW**: added `\subsection{Does the encoder carry signal beyond coordinates?}` (label `sec:res-headline-ablation`) — one paragraph reporting the no-encoder ablation result: 4 vs 57 sub-0.99 polygons on dev_test (14× gap), 3 vs 81 sub-0.95 polygons on OOD (27× gap), min-cov 0.765 vs 0.049 on OOD.
- **§7 Discussion**: replaced "visible separation" sentence with **two** quantitative measurements — (i) the ablation paragraph (referenced via `\ref{sec:res-headline-ablation}`), (ii) the linear-probe ROC-AUC `0.842 ± 0.024` (5-fold polygon-grouped CV, 200 polygons, 12,268 vertices, PR-AUC 0.580 ± 0.048 vs prior 0.16).
- **§7 fig:mechanism caption**: now says "the 2D scatter is the visualization, the AUC is the measurement" — proper rhetorical hierarchy.
- **§7 "not a better solver" paragraph**: anchored on Table~\ref{tab:headline} — greedy at `|S|/n=0.15, |S|/OPT=1.01`; probe at `|S|/n=0.44, |S|/OPT=2.90`. The "roughly three times the OPT guard count" framing is now precise and reproducible.

### Compile status (post-edits)

`latexmk -pdf` exit 0. **20 pages**, 0 undefined refs, 0 errors. Page count unchanged because the new prose displaced the deleted "visible separation" rhetoric. Headline table now has 5 rows + 2 group headers vs the previous 2 rows — height is comparable to the old display (the new compact `lrrrr` is similar width to the old `lrrr`).

### Multi-seed integration (2026-06-05)

**Step 4 (multiseed_eval) done** — 3 seeds × ~40 min OOD CGAL eval each. **Step 7 (multiseed_summary)** rebuilt with 4 seeds (1234, 11, 22, 33).

Multi-seed numbers (probe at $t = 0.20$):

| Split | metric | mean ± std | per-seed (1234, 11, 22, 33) |
|---|---|---|---|
| dev_test | cov | 0.9984 ± 0.0011 | — |
| dev_test | #<0.95 / 367 | 0.75 ± 1.30 | 0, 0, 3, 0 |
| dev_test | \|S\|/n | 0.389 ± 0.061 | — |
| dev_test | \|S\|/OPT | 2.56 ± 0.40 | — |
| OOD | cov | 0.9946 ± 0.0020 | — |
| OOD | #<0.95 / 2107 | 17.5 ± 12.5 | **3, 34, 25, 8** |
| OOD | \|S\|/n | 0.362 ± 0.045 | — |
| OOD | \|S\|/OPT | 2.34 ± 0.29 | — |
| OOD | min_cov | range 0.58–0.88 | 0.765, 0.576, 0.675, 0.884 |

**Honest read.** The displayed seed (1234) was on the aggressive end of the threshold-cardinality Pareto. At fixed $t = 0.20$ the four seeds land at different operating points; the variation in `#<0.95` is large (3–34 on OOD) because the probes calibrate slightly differently. What's stable is the qualitative ~10× reduction over the policy seed's 310 failures; what's variable is exactly where each seed lands.

### Paper claims updated to multi-seed

- **Abstract** + **C2 contribution**: "about two orders of magnitude" → "roughly an order of magnitude on average across four probe seeds". 310/3 = 103× was the single-seed claim; 310/17.5 = 17.7× is the four-seed mean.
- **§6.1b ablation paragraph**: full probe stats now quote 4-seed mean (cov $0.998 \pm 0.001$, sub-$0.99$ range 1–9 mean 4.5). The no-encoder ablation gap is now framed as "qualitative" because the absolute numbers shift across seeds.
- **§6.3 OOD subsection**: rewritten with explicit 4-seed mean (17.5 ± 12.5), per-seed enumeration {3, 8, 25, 34}, and across-seed min_cov range (0.576–0.884). Acknowledges that fixed-$t$ comparison places different seeds at different points on the Pareto.
- **§6.3 OOD failure analysis paragraph**: prefixed with "for the displayed seed (1234)" so the per-polygon analysis (rand-10-8, rand-20-20, random-20-10) is correctly scoped to one seed.
- **§6.4 cost-of-feasibility**: replaced $|S|/\OPT \approx 1.09 / 2.20 / 2.90$ single-seed numbers with $1.09 / 2.02 \pm 0.21 / 2.56 \pm 0.40$ four-seed means. Also softened "$367/367$ on every polygon" to "$0.75 \pm 1.3$ sub-$0.95$ polygons (three of four seeds reach $367/367$)".
- **§7 Discussion (cost paragraph)**: replaced $|S|/n=0.44, |S|/\OPT=2.90$ single-seed with multi-seed mean $0.389 \pm 0.061 / 2.56 \pm 0.40$.
- **§7 Discussion (encoder paragraph)**: "reduces the out-of-distribution tail to three polygons of 2107" → "reduces the out-of-distribution feasibility tail by roughly an order of magnitude on average across four probe seeds".
- **§8 Limitations item 4** ("Single random seed"): rewritten to acknowledge the probe is 4-seed evaluated and the policy alone is single-seed (because the encoder is the studied artifact).

### tab_headline (final form)

✅ Six-row table:
- Classical greedy, Local search (deterministic; LS marked with $\dagger$ for $|S|/\OPT < 1$)
- Pretrained pointer seed (deterministic; the policy is single-seed)
- **SetPredictor full** at $t=0.20$ — **mean $\pm$ std across 4 probe seeds**
- SetPredictor no-encoder ablation at $t=0.20$ (single-seed, since the ablation is a contrast)

Note: a single SetPredictor row carries the multi-seed std; the seed and no-encoder rows are deterministic at their respective seeds and don't carry std. The caption clarifies this scoping.

### Compile + verification

`latexmk -pdf` exit 0. **20 pages**, 0 undefined refs, 0 errors. No new figures; the multi-seed integration is entirely numerical (rewriting numbers in 7 paragraphs across §6, §7, §8 + the headline table).

### What's still pending (truly optional)

- **fig:distributions**: no-encoder overlay still optional; could also add multi-seed mean curve. Defer — visual clutter risk, and the numerical evidence is already comprehensive.
- **OOD classical baseline coverage**: still partial (40/2107). Footnote in caption explains. Full OOD greedy recomputation (~3h CGAL) only if a reviewer specifically asks.
- **Multi-seed for the SetPredictor at t=0.25 / t=0.30**: data exists in the per-seed JSONs but not surfaced in tab_headline (which only shows t=0.20). Could augment if needed.

### Status snapshot (2026-06-05)

| Artifact | State |
|---|---|
| `paper/data/multi_seed_summary.json` | ✅ 4 seeds, populated |
| `paper/data/dist_*_seed{11,22,33}.json` | ✅ 6 files written by Step 4 |
| `checkpoints/set_predictor/seed{11,22,33}/` | ✅ 3 best+final checkpoints |
| `paper/tables/tab_headline.tex` | ✅ 6 rows, probe row with ±std |
| `paper/paper.tex` | ✅ All multi-seed numerical updates applied |
| `paper.pdf` | ✅ 20 pages, clean compile |
| Logs | `paper/logs/multiseed_train_*.log`, `multiseed_eval_*.log` |

**Phase C is complete.** Revision pass closes: M1 (multi-seed ✓), M2 (coordinate-only ablation ✓), M3 (REINFORCE softened to citation ✓), M4 (classical anchors ✓), M5 (geo-free language audit ✓), m1 (training-curve caption ✓), m2 (linear-probe AUC ✓), m3 (OOD failure analysis ✓), mi3 (diagnostic framing ✓), mi4 (worked-example t=0.20 ✓).

## Phase D — optional polish (2026-06-05)

After Phase C compiled clean at 20 pages, an adversarial reviewer reread surfaced 30 findings. Commit `63dd220` closed the must-fix and high-value items; this Phase D commit closes the remaining substantive ones.

### Substantive finding: cross-seed OOD failure pattern (most important)

Verified directly from per-seed JSONs that the "small polygons dominate the OOD failures" claim is **specific to the displayed seed (1234)**. Across the other three seeds, failures concentrate at **large** n:

| seed | #fail | failure-n median | n ≤ 20 | n ≥ 300 |
|---|---|---|---|---|
| 1234 | 3 | 20 | **3/3** | 0 |
| 11 | 34 | 425 | 1/34 | 32/34 |
| 22 | 25 | 50 | 6/25 | 11/25 |
| 33 | 8 | 425 | 0/8 | 7/8 |

Mean probe coverage at $n \in [401, 500]$:
- seed 1234: 0.994
- seed 11: 0.972
- seed 22: 0.986
- seed 33: 0.987

The aggregate result (mean coverage 0.97–0.99 at n≥400 across seeds, four-seed mean 17.5 ± 12.5 OOD failures) stands; what varies seed-to-seed is *where* the residual failures concentrate along the n-axis. §6.3 OOD failure-analysis paragraph rewritten to disclose this honestly. The C2 claim (one-order-of-magnitude reduction) stays; the failure-mechanism interpretation now correctly distinguishes between displayed-seed (small-polygon failures) and other seeds (scale-end failures).

### Other Phase D edits

- **§4.2** "broadly successful / less successful": anchored to Table~\ref{tab:headline} numbers (mean $|S|/n = 0.166$ for policy vs $0.152$ greedy vs $0.137$ LS; 71/367 dev_test sub-0.95 polygons; 310/2107 OOD).
- **§4.3 iterative-editors claim** ("none improved in any coverage-preserving regime"): now backed by a single concrete number from `results/eval_editor_smoke.json` — the best editor variant on 30 polygons lost $0.005$ mean coverage despite reducing $|S|$ by $\sim 24\%$, and recovered only $0.27$ of the LS-target improvement.
- **§6.5 + fig:mechanism caption**: corrected a wrong claim I introduced earlier ("including the headline operating points"). The K-invariance was empirically verified at $t \in \{0.5, 0.6, 0.65, 0.7, 0.75, 0.8\}$ only; the headline operating range $\{0.20, 0.25, 0.30\}$ was NOT in fig:mechanism(a)'s sweep. Caption now scopes the empirical claim accurately and argues structurally that the fixed-point property should be most stringently tested near the decision boundary (where the high-t sweep lies).
- **tab_fixed_point caption**: explains the $t=0.65$ choice and the empirical range.

### Compile + final state

- 21 pages (up from 20; +1 page of polish prose). Still ≤22 target.
- 0 errors, 0 undefined references.
- Anti-repetition counts unchanged: 0.969 ×1, 1.09 ×2, 0.957 ×1, 0.830 ×1, 0.049 ×1, 2.20 ×1, 2.90 ×1.

### Not done (truly defer-to-submission)

- **#26–28** page-budget trims — at 21/22, fine.
- **#29** `subfigure` → `subcaption` deprecation — submission cleanup.
- **#30** `\figincl` placeholder branch removal — submission cleanup.
- Multi-seed visual envelope in `fig:distributions` — would add visual reinforcement but the numerical evidence in tab_headline and §6.3 is already complete.

### Status snapshot (2026-06-05, end of Phase D)

| Artifact | State |
|---|---|
| `paper/paper.tex` | ✅ Phase D polish applied |
| `paper.pdf` | ✅ 21 pages, clean compile |
| `paper/tables/tab_headline.tex` | ✅ Wilson CI scoped, multi-seed ±std on full probe |
| `paper/paper.md` | ✅ Phase D record added |
| All compute artifacts | gitignored (regenerable via `run_revisions.sh`) |

**Phase D closes.** The paper is in a defensible Q1 submission state. Remaining work is submission logistics (cover letter, deprecation cleanups, optional visual envelope) rather than scientific content.

---

## Open issues / decisions deferred

1. **`|S|/OPT < 1` for LS** — primary cardinality metric for classical row should be `|S|/n` to avoid this artifact. Mention OPT-reference caveat in §5 or a table footnote.
2. **OOD greedy partial coverage** — 40/2107. Either re-run greedy on all OOD (~3h CGAL) or report as "comparable subset" with footnote. Defer to integration time; lean toward the footnote.
3. **No-encoder probe in `fig:distributions`** — 5th column is at risk of clutter. Maybe add as overlay only on the coverage CDF panel.
4. **Page budget** — currently 20 pages after Phase A. Phase C will add ~1 column of prose + 1 table row. Target ≤22.
