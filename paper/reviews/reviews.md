Ref: Submission ID 5a4a8644-bc30-4439-a722-a7947efa0d3a  

Dear Dr Ševerdija,  

Your manuscript "Learning to Place Guards by Reinforcement: A Geo-Free Neural Policy for the Vertex-Guard Art Gallery Problem" has now been assessed. If there are any reviewer comments on your manuscript, you can find them at the end of this email.  

Regrettably, your manuscript has been rejected for publication in Memetic Computing.  

Editor Comments  

"This paper invesitigated the vertex-guard art gallery problem by reinforcement learning. The motivation is interesting. However, some reviewers comment that some conclusions are stated somewhat more strongly than the evidence supports and the recent references on probing in Neural Combinatorial Optimization should be considered."  

Thank you for the opportunity to review your work. I'm sorry that we cannot be more positive on this occasion and hope you will not be deterred from submitting future work to Memetic Computing.  

Kind regards,  

Zexuan Zhu  
Editor-in-Chief  
Memetic Computing  

Reviewer Comments:  

## Reviewer 1

**Recommendation:** Minor Revision

### 1. Obvious Grammatical and Typographical Errors

- On page 7, "an end-of-sequence - EOS token" should be written as "an end-of-sequence (EOS) token."
- On page 14, "Checkpoints are selected on dev; whereas test, ood, and ood-large …" uses an incorrect semicolon. It should be "Checkpoints are selected on dev, whereas test, ood, and ood-large …" or be divided into two sentences.
- On page 15, "the distribution is Section 6.2" should be revised to "the distribution is shown/discussed in Section 6.2."
- On page 18, the construction "we strip the probe of all capacity: a plain linear classifier …, asked to separate guard from non-guard vertices" is grammatically incomplete. A clear revision would be: "we replace the probe with a plain linear classifier (without hidden layers or attention) that reads only the frozen per-vertex features and separates guard from non-guard vertices."
- On page 22, "hidden states unseen at training" should read "hidden states unseen during training."

### 2. Points that Require Further Clarification

- **Meaning of "feasibility".** Equation (2) defines a feasible AGPVG solution by exact full coverage, Cov_P(S) = 1, whereas Section 5.3 calls Cov_P(S) ≥ 0.95 the "feasibility threshold." Consequently, statements such as "closes the feasibility tail" can refer either to exact feasibility or only to passing the 0.95 coverage gate. Please distinguish these two meanings consistently, for example by reserving "feasible" for full coverage and calling 0.95 a "coverage gate" or "near-feasibility threshold."
- **Scope of "geo-free".** The abstract says that the policy is trained under the geo-free constraint, but visibility computations are used to form the training reward and the probe targets. Section 2.3 explains that the restriction applies only to inference inputs, yet this is not evident from the abstract and early introduction. Please state there explicitly that "geo-free" means no visibility oracle is queried at inference, while visibility information is used during training and evaluation.
- **Threshold selection.** The manuscript repeatedly treats t = 0.20 as the headline threshold, but the exact development-set selection rule is not stated. Please explain how t = 0.20 and the sweep {0.20, 0.25, 0.30} were chosen without reference to test/OOD results. In Section 6.4, the fixed-point analysis instead uses t ∈ {0.50, …, 0.80}; please explain why this different range supports the single-pass claim at the headline thresholds.
- **Policy runs, probe seeds, and checkpoint selection.** Section 5.2 says that the policy checkpoint was "trained once from a single seed," whereas Limitation 4 says that it was the best of several pretraining runs. Please reconcile these descriptions by stating the number of policy-training runs, their seeds, the checkpoint-selection criterion, and whether "four seeds" elsewhere always refers only to SetPredictor training seeds.
- **Interpretation of Table 1.** The REINFORCE rows were evaluated under "an earlier version of the evaluation protocol," but the differences from the final protocol are not described. Please state those differences so that readers can understand which quantities remain comparable and what conclusion the table is intended to support.

Attachments:  
• https://reviewer-feedback.springernature.com/download/attachment/b71608e7-8bdc-4d5e-9584-773aadaf52b3  

## Reviewer 2

This paper studies a representation-analysis question: whether an RL-trained policy for vertex guarding encodes useful coverage-related structure beyond what its decoder expresses. The geo-free protocol, the encoder–seed ablations, and the OOD experiments are valuable aspects of the study. However, I am not yet convinced that the current evidence supports the paper’s strongest interpretation that the encoder has internalized the geometry required for feasibility and that the remaining failures are primarily decoder-calibration errors. The practical motivation for geo-free inference and several important experimental controls also remain underdeveloped.  

1. The paper frames geo-free inference primarily as a way to separate what the learned representation contains from what a visibility oracle can compute. This is a meaningful diagnostic objective, but the broader importance of the restriction remains insufficiently justified. Because polygon coordinates are available and visibility-based greedy performs substantially better in guard count, the paper should identify concrete settings in which avoiding visibility computation is beneficial and provide an end-to-end runtime and memory comparison with classical visibility-based pipelines.  

2. The principal components—pointer networks, Bradley–Terry preference optimization, frozen-representation probing, and Transformer-based per-vertex classification—are established techniques. The novelty therefore appears to lie in their combination and adaptation to a geometric covering problem rather than in a new architecture or optimization method.  

3. I do not view the use of visibility-based rewards or local-search labels as circular in itself; such supervision may legitimately shape the representation. However, successful probing establishes that the local-search target is decodable from the frozen features, not necessarily that the encoder contains all geometry required for feasibility or that residual failures arise solely from decoder calibration.  

4. The guard-count overhead substantially limits the method’s competitiveness as an AGP solver: the full probe operates at approximately two to three times the optimum, while the visibility-based greedy reference is close to the optimum. More importantly, the paper formally defines feasibility as full coverage, Cov⁡(S)=1, but subsequently uses Cov⁡(S)≥0.95 as a feasibility threshold.  

5. The evaluation covers multiple polygon families and a broad size range, but most data come from the same instance ecosystem, and the main OOD test is predominantly size extrapolation.  

6. The missing OPT values for part of the extreme-OOD split appropriately limit guard-cost conclusions at that scale, although the paper already acknowledges this issue. The fixed-point result is an empirical observation and should not be used as strong validation that the probe is necessarily a clean representation readout. The coords-only condition is discussed in the paper and clearly succeeds through over-guarding; the more important concern is that the ablations are compared at a common probability threshold despite having different calibration and guard counts.  

## Reviewer 3

This manuscript investigates whether a neural policy can learn to solve the vertex-guard Art Gallery Problem (AGP) using only vertex coordinates at inference time, without access to visibility computations. The authors train a pointer-network policy using preference optimization (PO) with Bradley-Terry ranking, then probe the frozen encoder with a small supervised classifier (SETPREDICTOR) to assess what geometric information the adopted reinforcement-learned representation has internalized.

This is a methodologically interesting paper that addresses an important question in neural combinatorial optimization: what do RL-trained policies actually learn? The representation-probing approach is creative, and the experimental design is thoughtful.

However, several significant issues need to be addressed before the paper meets the standards for publication in a top-tier venue.

1. The claim that "to our knowledge this is the first such probing study of an RL-trained policy on a geometric covering problem" (C1) is likely true but overstated. The broader framing of representation probing in NCO is not as novel as the authors state on page 7 (Probing learned representations). There is emerging work on analyzing representations in learned combinatorial solvers that warrants consideration [1, 2, 3], and the current work should be accurately positioned among the existing works.

2. The introduction would benefit from a clear, concise statement of the practical challenge: why should we care whether a geo-free learner can place guards? The applications (surveillance, sensor placement, robotics) are mentioned but not connected to the specific constraint of not using visibility during inference.

3. The ablation studies demonstrate that the learned encoder produces informative representations, as replacing it with an untrained encoder causes downstream performance to collapse. Moreover, the linear probe (ROC-AUC ≈ 0.84) provides strong evidence that a substantial amount of guard-relevant information is already linearly accessible from the frozen embeddings.

However, I believe the interpretation of these results deserves a more nuanced discussion. The proposed SetPredictor is not a lightweight probe, but rather a relatively expressive Transformer (3 self-attention layers, 8 attention heads, and approximately 464K parameters). The large performance improvement from the linear probe (ROC-AUC ≈ 0.84) to the Transformer-based SetPredictor (ROC-AUC ≈ 0.98) suggests that a significant amount of additional nonlinear computation is still required to recover the final predictions. Consequently, the current experiments do not fully determine whether this additional performance arises because the Transformer is able to extract information that is only implicitly encoded in the learned representations, or because it performs additional task-specific nonlinear reasoning.

Therefore, I believe the presented evidence supports the conclusion that the encoder embeddings contain rich and highly informative geometric features, but provides somewhat weaker evidence for the stronger claim that the encoder alone has learned the underlying geometry and that the decoder is solely responsible for the remaining errors.

This issue is particularly relevant in light of the recent literature on probing in Neural Combinatorial Optimization, which explicitly discusses the influence of probe expressiveness on the interpretation of probing results. I encourage the authors to better position the proposed SetPredictor within this literature and discuss the implications of using a relatively powerful probing model. Furthermore, if feasible, evaluating an intermediate-capacity probe (e.g., a shallow MLP) would further strengthen the conclusions by characterizing how probe expressiveness influences performance. Such an experiment would help determine whether the remaining information is readily accessible through modest nonlinear transformations or whether it specifically requires a highly expressive Transformer-based probe.

4. Several statements throughout the manuscript are phrased in terms of "what reinforcement learning" or "what neural combinatorial optimization" internalizes. Since all experiments are conducted with a single policy architecture and training pipeline, I encourage the authors to clarify the scope of these conclusions. The current evidence supports conclusions about the studied RL policy, whereas broader claims regarding reinforcement learning or neural combinatorial optimization in general would require evidence across multiple architectures and learning paradigms.

5. The current title emphasizes the reinforcement-learning policy itself ("Learning to Place Guards by Reinforcement" and "A Geo-Free Neural Policy"), whereas the manuscript's principal contribution is the analysis of the learned representations and the investigation of what the policy has internalized. The authors may wish to consider a title that better highlights the probing aspect of the work.

**References**

[1] R. Narad, L. Boussioux, and M. Wagner. Probing neural tsp representations for prescriptive decision support. arXiv preprint arXiv:2602.07216, 2026.

[2] Z. Zhang, Y. Ma, Z. Cao, and H. C. Lau. Probing neural combinatorial optimization models. Advances in Neural Information Processing Systems, 38:131824–131862, 2026.

[3] Z. Zhang, Y. Ma, J. Yang, Z. Cao, and H. C. Lau. Unveiling neural combinatorial optimization model representations through probing.

Attachments:  
• https://reviewer-feedback.springernature.com/download/attachment/5439b93b-8186-41ec-9889-7e5c9e53e319  

## Reviewer 4

The paper investigates the vertex-guard Art Gallery Problem and adopts an optimization approach based on the Bradley–Terry model to alleviate the vanishing-gradient issue in reinforcement learning. The experimental results show that preference optimization clearly outperforms the basic reinforcement-learning methods, although the quality of the resulting solutions is inferior to that achieved by heuristic methods. Another research question explored in the paper is whether the encoder has learned meaningful geometric structure, which is examined through a series of ablation studies. Although the methodological novelty of the paper is somewhat limited, it provides a useful analysis of the interpretability of neural combinatorial optimization models. Overall, the analytical perspective adopted in the paper is of certain value.  

Q1. The NeurIPS 2025 paper, “Probing Neural Combinatorial Optimization Models,” investigates the role of encoder representations in NCO models for the Traveling Salesman Problem. It may be helpful to include this work in the related-work section and clarify how the present study differs from it in terms of the problem setting, learning framework, and probing objective.  
Q2. Would it be possible to include an additional comparison using a Transformer with the same or comparable architecture trained directly from vertex coordinates? Such an experiment could help further clarify the contribution of the RL encoder. In particular, it would be informative to determine whether the RL encoder provides additional information that is difficult for a supervised model to learn directly from coordinates, or whether the LS targets and the Transformer architecture alone are already sufficient to accomplish most of the task.
