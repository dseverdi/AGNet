# Peer Review

**Manuscript:** *Learning to Place Guards by Reinforcement: A Geo-Free Neural Policy for the Vertex-Guard Art Gallery Problem*

**Reviewed as a journal submission (single-blind, methods venue).**

---

## 1. Summary

The paper poses a narrowly scoped, well-formulated question: can a neural policy, trained from a coverage-aware reward on its own rollouts, learn to place vertex guards on simple polygons without ever seeing an explicit visibility object at inference? The authors train an LSTM pointer policy using preference optimization on Bradley–Terry pairs (PO/BT) of self-generated rollouts, evaluate it on held-out in-distribution and out-of-distribution splits of a public instance library, and find that the policy is competitive in guard cardinality but leaves a non-trivial fraction of polygons under-covered. They then train a small single-shot per-vertex classifier (the *SetPredictor*) that reads only the frozen pointer encoder's embeddings, no visibility input, and show that this probe closes the feasibility tail. The combined result is read as evidence that the reinforcement-trained encoder already carries enough geometric structure to support feasibility decisions, and that the residual tail of the policy is a decoder-calibration limit rather than a representation limit.

## 2. Overall Assessment and Recommendation

This is a careful, conceptually disciplined paper. The framing — RL as a measurement instrument, the SetPredictor as a representation probe, the geo-free constraint as a methodological restriction rather than a rhetorical flourish — is unusually rigorous for a learning-for-combinatorial-optimization manuscript. The honesty about what is and is not being claimed (no competition with greedy/LS on guard count) is refreshing and removes a category of objections that usually distract reviewers of NCO work.

The empirical claims are well-scoped to what the experiments support, with one important caveat: the central interpretation — that the encoder "carries geometric structure" — leans heavily on a single comparison (probe with vs. without seeing the encoder) that is currently asserted by argument rather than by direct ablation. A representation-probe study should include the obvious counterfactual.

**Recommendation:** *Major revision*. The contribution is genuine and the writing is unusually mature, but four points (two of them load-bearing) need to be addressed before publication.

---

## 3. Strengths

- **Question is sharp and falsifiable.** "Can RL learn to place vertex guards under a geo-free inference constraint" is a measurable claim. The paper succeeds in keeping this claim honest throughout.
- **Geo-free constraint is precisely defined (§2, last paragraph).** This is rare and important — many "learning to solve X" papers conflate "no oracle at inference" with "no oracle at training," and the authors are explicit about which they enforce.
- **The held-out methodology is correct and the alphabetical-split caveat is reported.** The 70/30 partition with a seeded random shuffle is the right call; the earlier alphabetical-split bias is noted, which is the kind of small but honest scientific disclosure I want to see.
- **The framing of the SetPredictor as a probe, not a competing solver, is methodologically clean.** It is correctly used as a diagnostic device rather than a contribution against classical solvers.
- **Honest framing of trade-offs.** The Discussion explicitly concedes that classical greedy and local search beat the SetPredictor on $|S|/\OPT$ — this is the right disclosure and removes a class of objections.
- **Limitations section is genuinely informative.** Items 1 (geo-free is at inference only, not training) and 2 (probe is supervised, not RL) are exactly the kind of caveat that a careful reader would otherwise have to derive themselves.
- **Tables and figures are data-driven (auto-generated from JSONs).** This reproducibility scaffolding is excellent practice and should be highlighted by the editors.

---

## 4. Major Concerns

### M1. The "encoder carries geometric structure" claim needs a direct ablation, not just a probe.

The paper's central interpretation — that the RL-trained encoder has internalized guard-relevant geometric structure — currently rests on the observation that a small classifier reading those embeddings recovers feasibility. This is suggestive but not conclusive. The obvious counterfactual is *missing*: a SetPredictor of the same capacity that reads only raw coordinates $x_i$ and the seed indicator $\mathbf{1}\{i \in \pi_{\mathrm{seed}}\}$, without the encoder embeddings $h_i$. If that ablated probe also closes the feasibility tail, the claim about the encoder is unsupported — the coordinates plus the seed indicator alone may already carry enough information for a small Transformer to solve the task. If it does not, the claim is strengthened.

I would expect a journal paper making this representation claim to report this ablation. It is a one-line change in the feature concatenation (Eq. (eq:setpred-feat)), a re-run of training, and a row added to Table 1.

A related ablation — using a *randomly initialized* (untrained) encoder of the same shape — would help separate "this LSTM architecture leaks coordinate signal" from "this trained encoder has learned something useful." A two-row table comparing (i) full features, (ii) no $h_i$, (iii) untrained $h_i$ would settle the representation claim cleanly.

The PCA visualisation (Figure 5) is suggestive but not load-bearing — it shows that guard and non-guard vertices are linearly separable in the trained encoder's 2D projection, but the same could be true for raw coordinates of vertices in instances where the LS target preferentially picks corners or convex turns.

### M2. The PO/BT vs. REINFORCE comparison is asserted, not measured.

§4.1, paragraph 4: "we observed that rewards saturate quickly into a bimodal regime ... where REINFORCE advantages collapse." This is the paper's only explicit defence of PO over REINFORCE for vertex-guard AGP, and it is presented as an observation without a measurement. Given that PO/BT is the methodological commitment of the paper, the reader deserves either: (a) a small experiment showing REINFORCE collapsing in this setting, or (b) a more careful presentation of this as a hypothesis rather than a finding. The current phrasing has the rhetorical weight of evidence without the corresponding measurement.

This is doubly important because Bello et al. (2016) and the broader NCO literature have used REINFORCE+baseline successfully on similar pointer-network combinatorial tasks. A reader will reasonably ask: why doesn't it work here? The paper should either show the failure or pull back the claim.

### M3. Statistical reporting beyond Wilson CIs is thin.

The paper uses Wilson 95% CIs on the coverage-feasibility proportion (good, and correct), but a single random seed is acknowledged as a limitation. For a journal venue, the bar is higher than for a workshop. At minimum, I would like to see:

- The PO/BT policy trained with **at least three random seeds** and the headline coverage and $|S|/\OPT$ reported as mean ± SD across seeds. This is cheap (the policy is small).
- The SetPredictor trained with at least three seeds on a fixed policy seed. Also cheap.
- Bootstrap CIs on $|S|/\OPT$ and on mean coverage (the Wilson CI only covers the proportion).

Without this, the residual differences between thresholds in Tables 2–3 are not reliably attributable to method changes versus seed noise. Item 4 of the Limitations section concedes this, but the right move is to do the multi-seed run rather than admit it.

### M4. The "PO/BT training dynamics" figure (Figure 1). [RESOLVED in this revision]

At the time of the first read of this manuscript Figure 1 was a "Figure pending" placeholder, which would have blocked submission. The authors have since reconstructed a coarse 4-point curve from four saved checkpoints (epochs 110, 114-best, 160, 200) evaluated on a 100-polygon dev_test sample, written by `paper/scripts/build_po_training_curve.py`. This is acceptable as a *post-hoc* reconstruction, but the figure caption should now state honestly that the curve was reconstructed after the fact from saved checkpoints rather than logged during training, and that the four points are restricted to the late-training regime (epochs 110--200). A complete training curve over all 200 epochs would still be preferable; if a future training run is available the authors should preserve per-epoch metrics in a sidecar JSON.

---

## 5. Minor Concerns

### Presentation

- **(p. 1, Abstract)** "we treat it here as a measurement instrument" — this framing is unusual and effective, but the abstract is dense and three sentences long in places. Consider one cleaving cut for readability; the message will not weaken.
- **(p. 2, ¶3)** "On a held-out in-distribution split it reaches mean coverage of roughly $0.97$" — Section 6.1 reports $0.969$, with the figure 0.97 used in §6.1's caption. Pick one rounding and apply it uniformly.
- **(p. 3, Related Work)** The label of the line of work as "Supervised post-processing of learned solvers" is fine, but a one-sentence acknowledgement of the post-hoc-improvement literature outside ML (e.g., LP rounding, local search on heuristic outputs) would round out the framing.
- **(§4.2, "We mention this only to motivate the single-shot design")** — the paragraph is one long sentence. Break.
- **(Figure 4, K-invariance caption)** Phrasing: "the across-$K$ variation is at the fourth decimal at every threshold" — the y-axis is tightly cropped (acknowledged), but a reviewer's worry is whether tight cropping inflates a real change. Add the absolute y-range explicitly, e.g., "($0.972 \le \overline{\mathrm{Cov}} \le 0.984$)."

### Specific text edits

- **(p. 4, ¶ before Eq. (eq:bt-loss))** "above a coverage gate" is undefined the first time it appears. State the gate value here (or forward-reference §4.4 where $\tau = 0.99$ appears).
- **(§4.1 "Reward and rollouts")** "$K$ rollouts" — $K$ is reused in §4.5 for the iterative-inference passes of the probe. Rename one (e.g., $K_{\mathrm{roll}}$ vs $K$) to remove the collision.
- **(§4.2)** "DAgger refresh~\cite{RossGordonBagnell2011DAgger}" — the citation is fine. The phrase "refresh" is slightly informal; "DAgger-style on-policy correction" is the conventional usage.
- **(§4.3, Eq. (eq:setpred-feat))** $z_i \in \mathbb{R}^{131}$ — this is $128 + 2 + 1 = 131$; clarify in text that the $+1$ is the binary indicator (a reader will count $128 + 2 = 130$ and wonder where the third dimension came from).
- **(§4.4, "single random seed (1234)")** Repeated in §5.2 and §8.4. Consolidate.
- **(§5.1, Table 1 caption)** "AGPIL" library — the bibliography entry `art-gallery-instances-page` refers to "AGPVG" / "Couto, de Rezende, de Souza." Standardise the abbreviation.
- **(§6.1)** "$|S|/\OPT \approx 1.09$" appears at least three times in the paper (§1, §6.1, Discussion). Consider whether all three need it.
- **(§6.5)** "approximately two orders of magnitude" — be precise: $310 \to 3$ is a factor of $103$. "Two orders of magnitude" is correct; "$\sim 100\times$" (used in the table caption) is also correct. Use one phrasing.
- **(Figure 5 caption)** "Visible separation between guard and non-guard vertices ... is direct visual evidence" — *visible separation in a 2-D projection of a 128-D space* is suggestive but not direct evidence; in particular it does not control for the possibility that the same separation appears in PCA of raw coordinates. Soften to "consistent with the hypothesis" rather than "direct evidence." (See M1.)

### Citations / bibliography

- **(`paper.bib`)** `Pan2025PO` — the bib entry has author "Pan, Yu and others" with no full author list. If the paper is forthcoming/accepted ICML 2025, a complete entry is required. Same for the venue (full proceedings title, page range when known).
- **(`paper.bib`)** `cgal:eb-25b` — edition 5.6 is from 2024 and the citation key references "25b". Either update the key or the edition for internal consistency.
- **(§3, "Reinforcement learning for combinatorial optimization")** Consider citing Kool et al. (2019) more centrally — the attention model they introduced is the standard baseline a reviewer will ask about. The Williams (1992) citation for REINFORCE is correct but should be paired with Sutton & Barto (2018) for context.
- **(§3, last paragraph on DAgger)** Ross et al. (2011) is correct. A pointer to the broader feasibility-repair-by-learning literature (e.g., constrained RL surveys, geometric set-cover learning work) would help a reader new to the area.

---

## 6. Detailed Comments by Section

### Abstract

The abstract is unusually well-written for a methods paper. The "we treat it here as a measurement instrument" framing should not be removed. The closing sentence ("we do not claim to compete with classical greedy or local-search heuristics on guard count; the contribution is a bounded answer ...") is exactly the kind of scope disclosure I want at the top of a paper. Word count is approximately 280 — slightly long for a journal abstract; one paragraph could be merged.

### §1 Introduction

Clear and well-organised. The two-contribution structure (down from three in an earlier draft, judging by the related-work compression) is right. Three small items:

- The phrase "to a measurable degree" appears twice in close proximity (¶4 and ¶5). One occurrence is enough.
- The reading of the OOD result as "the encoder generalises" needs the M1 ablation.
- Consider adding one sentence about why vertex-guard (rather than point-guard) is the right setting: the current passage in §2 about the discrete action space and the natural pointer-network fit is useful, and a one-line preview in §1 would help a reader who skips ahead.

### §2 Problem and Notation

Well-formed. The geo-free definition is precise and the train/inference distinction is correctly stated. The paragraph about "what makes 'no explicit geometric knowledge' a measurable property and not just a rhetorical one" is one of the best paragraphs in the paper and should stay verbatim.

### §3 Related Work

Three paragraphs is the right length. The Pan2025PO citation is the methodological anchor and is correctly placed. The supervised-post-processing paragraph is short — that is correct in light of the SetPredictor's role as a probe rather than a method.

### §4 Method

§4.1 (the RL method) is the most technical section. It needs:

1. M2's evidence that PO is better than REINFORCE in this setting, or a softening of the claim.
2. The training curve figure (M4) made non-placeholder.
3. The $K_{\mathrm{roll}}$ vs $K$ symbol collision resolved.

§4.2 (architectural alternative) is the right length — one paragraph framed as motivation, not a co-equal contribution. Good revision from the earlier multi-paragraph diagnostic.

§4.3–4.5 (the SetPredictor) are clear and reproducible. M1's coordinate-only ablation would slot into §4.3 cleanly.

### §5 Experiments

The held-out methodology is clean. The OOD setup is the right call. Two points:

- The dataset table is good. State whether the train/dev/test/large splits are disjoint by family (random simple, orthogonal, von Koch) or by polygon name. If the latter, the OOD signal could include a family-distribution shift in addition to a size shift.
- The disc-vis-vs-CGAL distinction is correctly stated. Mention the disc-vis approximation error empirically — what is the mean absolute difference between disc-vis ($M{=}500$) and CGAL exact coverage on a sample? This number is a load-bearing methodological assumption.

### §6 Results

The five subsections are well-organised. Specific comments:

- **§6.1 Headline.** The Wilson CI $(0.763, 0.844)$ in the seed row corresponds to a proportion of about $0.807$, which matches $296/367 = 0.806$. Good. Make sure the CI is on the *proportion*, not on the count.
- **§6.3 Pareto.** The threshold sweep is the right cardinality (three operating points + the broader dev_tune sweep for the figure). Consider also reporting a *fourth* threshold below 0.20 in the OOD panel to show the curve's behaviour when the model is asked to be even more conservative — does $|S|/\OPT$ continue to grow gracefully, or saturate?
- **§6.5 OOD.** The 100× reduction is the paper's headline. Strengthen by reporting per-size-bin numbers (e.g., $n \in [200, 500)$, $n \in [500, 1000]$): does the probe rescue uniformly across sizes, or only on the easier OOD polygons?
- **Tables 1–6.** Use of `booktabs` is good. Table 3 (Pareto) and Table 6 (OOD) have similar structure — consider combining for tighter presentation.

### §7 Discussion

Five paragraphs, well-organised. The "Framing of the comparison" paragraph is exactly right and removes a class of reviewer objections. The "Why a single-shot design" paragraph correctly leans on §4.2 rather than reasserting the diagnostic. The "Out-of-distribution behaviour" paragraph is the right place to address M1's worry: tighten the language around "the encoder carries geometric structure" to "the encoder carries information sufficient for a small classifier to recover feasibility" — pending the coordinate-only ablation.

### §8 Limitations

Seven items, comprehensive. Items 1 and 2 are the two most consequential and are correctly placed first. Item 4 (single random seed) should be promoted to a major concern if the multi-seed run is not done before submission (see M3).

### §9 Conclusion

Tight and accurate. The two future-work directions (stronger encoder, per-instance threshold) are concrete and plausible. The closing sentence is restrained and does not introduce new claims. Good.

---

## 7. Reproducibility

- Code paths, checkpoints, and trajectory pickles are referenced in the BUILD_INFO.md documentation that ships with the LaTeX source. This is excellent reproducibility scaffolding.
- The auto-generated tables (via `build_tables.py`) and figures (via `build_figures.py`) from JSON data files mean numerical claims trace back to source data — this is the right discipline.
- The two missing data files (`po_agp_training.json`, `setpred_test_OOD_per_polygon.json`) — and consequently two of the seven figures — must be present in the submission package. The `\figincl` placeholder mechanism is internally clever but is not appropriate for a journal submission; a missing figure is not a "Figure pending" box, it is a paper that does not show its work.

---

## 8. Summary of Required Changes (for revision)

**Must address (blocking):**

1. **(M1)** Add the coordinate-only / untrained-encoder ablations of the SetPredictor. Without these, the representation claim is not supported.
2. **(M2)** Either run the REINFORCE-collapses experiment or soften the language in §4.1 paragraph 4 to a hypothesis.
3. **(M3)** Multi-seed runs (≥3 seeds) for both the PO/BT policy and the SetPredictor, with mean ± SD or bootstrap CIs in Tables 1, 3, 6.
4. **(M4)** [RESOLVED in this revision] Both `fig_po_training` and `fig_cov_vs_n` are now generated. Tighten Figure 1's caption to disclose that the curve was reconstructed post-hoc from saved checkpoints (epochs 110, 114, 160, 200) rather than logged during training.

**Should address:**

5. Clarify the disc-vis approximation error empirically.
6. Resolve the $K$ symbol collision between rollout count and probe iteration count.
7. Tighten the encoder-PCA caption (Figure 5) to "consistent with" rather than "direct evidence."
8. Standardise rounding (0.969 vs 0.97), abbreviations (AGPIL vs AGPVG), and citation key conventions.

**Nice to have:**

9. Per-size-bin OOD breakdown.
10. Bib entries completed (full author lists).

I look forward to reviewing the revised manuscript. The core scientific question is well-chosen, the experimental design is honest, and the framing is unusually mature; with the ablations above, this is a publishable methods contribution.

— Reviewer
