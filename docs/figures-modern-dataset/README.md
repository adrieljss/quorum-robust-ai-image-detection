# figures-modern-dataset/ — the counterfactual, kept

Built with the candidate general probe from `ERROR_ANALYSIS.md` §8.15: the
shipped recipe plus 9,784 rows of photographic 2026-generator images
(GPT-image-2 and nano-banana-pro). Everything else is identical — same tampered
branch, same face branch, same threshold rule — so the comparison isolates the
training data.

    python scripts/make_figures.py \
        --general scratchpad/general_modern2.npz \
        --out figures-modern-dataset

**It is NOT shipped.** It failed every gate criterion:

| | shipped | modern |
|---|---|---|
| So-Fake-OOD clean AUROC | **0.9265** | 0.9163 |
| accuracy | **0.8380** | 0.8235 |
| recall | **0.7588** | 0.7260 |
| **FNR** | **24.12%** | **27.40%** |
| F1 | **0.8243** | 0.8046 |
| laion holdout FPR | **18.05%** | 24.95% |
| COCO FPR | 8.76% | **8.54%** |

FNR *rising* is the verdict: that is the one metric the experiment existed to
reduce. Only COCO false positives improve, by 0.22pp, on the pool that means
least (COCO is curated and nearly watermark-free — §7.7).

Kept because a counterfactual you can look at is worth more than a paragraph
saying it did not work, and because the obvious next idea — "train on the
generators we fail on" — will occur to everyone who reads §3.
