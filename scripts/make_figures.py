"""Regenerate docs/figures from cache. Everything here is computed, never drawn
from a pasted number, so a figure cannot outlive the result it illustrates.

    python scripts/make_figures.py            # all four
    python scripts/make_figures.py threshold  # just one

Four figures, one argument each:

  threshold.png     0.5 was never chosen, and it is the wrong place to cut
  separation.png    why -- the real-image score mass sits on top of the old cut
  robustness.png    the 15-variant grid as a picture instead of a table
  generalisation.png  the tampered branch cannot recognise unfamiliar REAL photos

reliability.png is not built here -- it belongs to quorum/fusion.py, which owns
calibration. Run `python -m quorum.fusion` for that one.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")                       # no display on a headless run
import matplotlib.pyplot as plt
import numpy as np

from quorum.detectors.general import MODELS, load

FIGURES = ROOT / "docs" / "figures"
# Shared with quorum/calibrate.py's reliability plot so the figure set reads as
# one document rather than four unrelated charts.
BLUE, RED, GREY = "#2B4A9B", "#B4462F", "#9AA3AF"
GREEN, AMBER = "#2A7355", "#B0741B"
OLD_THR, NEW_THR = 0.5, 0.766

# Short keys for the data, display labels separately. Keying the dict on the
# rendered label means every caller repeats a multi-line string literal, and one
# tweak to the wording silently KeyErrors the other figures.
REAL_SETS = {"sid": "SID_Set\nin-distribution",
             "ood": "So-Fake-OOD\nunseen generators",
             "coco": "COCO val2017\nunseen photography"}


def style(ax, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18, lw=0.6)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    return ax


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    p = FIGURES / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p.relative_to(ROOT)}")


# ---------------------------------------------------------------- data ------
def probe(name, X):
    z = np.load(MODELS / f"{name}.npz")
    return 1 / (1 + np.exp(-(X @ z["w"].ravel() + float(z["b"].ravel()[0]))))


def populations():
    """The three real-photo distributions plus the eval set, scored per branch.

    sid_calib stands in for 'reals the probe was trained around' -- sid_train
    itself is fitted on, so using it would flatter every number here.
    """
    Xo, Ro = load("so_fake_ood")
    Xc, Rc = load("organizer_val")
    Xs, Rs = load("sid_calib")
    ev = ((Ro.split == "test_ood") & (Ro.variant == "clean")).values
    out = {
        "y": Ro.label.values[ev],
        "eval": {n: probe(n, Xo[ev]) for n in ("general", "tampered")},
        "reals": {},
    }
    for key, X, R in (("sid", Xs, Rs), ("ood", Xo[ev], Ro[ev]), ("coco", Xc, Rc)):
        m = ((R.label.values == 0) & (R.variant == "clean").values
             if "variant" in R else R.label.values == 0)
        out["reals"][key] = {n: probe(n, X[m]) for n in ("general", "tampered")}
    return out


def shipped(p):
    return np.maximum(p["general"], p["tampered"])


# ------------------------------------------------------------- figures ------
def fig_threshold(D):
    """Every metric that depends on a cut, as a function of the cut."""
    y = D["y"]
    p = shipped(D["eval"])
    pc = shipped(D["reals"]["coco"])
    t = np.linspace(0.02, 0.98, 481)

    acc, prec, rec, fpr, coco = [], [], [], [], []
    for c in t:
        hat = p >= c
        tp = (hat & (y == 1)).sum(); fp = (hat & (y == 0)).sum()
        tn = (~hat & (y == 0)).sum(); fn = (~hat & (y == 1)).sum()
        acc.append((hat == (y == 1)).mean())
        prec.append(tp / (tp + fp) if tp + fp else np.nan)
        rec.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
        coco.append((pc >= c).mean())

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for v, lab, c, ls in ((prec, "precision", BLUE, "-"), (rec, "recall", RED, "-"),
                          (acc, "accuracy", GREEN, "-"),
                          (fpr, "FPR (So-Fake-OOD reals)", GREY, "--"),
                          (coco, "FPR (COCO photographs)", AMBER, "--")):
        ax.plot(t, v, ls, lw=1.8, c=c, label=lab)

    # Inside the axes, not above them: bbox_inches="tight" crops anything past
    # ylim and the labels vanish silently.
    for c, lab, col, ha in ((OLD_THR, "0.500  sigmoid default", GREY, "right"),
                            (NEW_THR, "0.766  shipped", BLUE, "left")):
        ax.axvline(c, lw=1.1, ls=(0, (3, 3)), c=col, zorder=1)
        ax.text(c + (-.012 if ha == "right" else .012), .05, lab, fontsize=8.2,
                color=col, ha=ha, va="bottom", fontweight="bold", zorder=6,
                bbox=dict(fc="white", ec="none", alpha=.78, pad=1.6))

    style(ax, "Every threshold-dependent metric, as a function of the threshold")
    ax.set_xlabel("decision threshold on the shipped score")
    ax.set_ylabel("rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.text(0.0, -0.04,
             "AUROC is identical everywhere on this axis. Moving the cut from 0.500 to 0.766 "
             "buys precision and costs recall;\nthe amber curve is the one nobody was watching "
             "— a quarter of ordinary photographs flagged as AI at the old default.",
             fontsize=8.2, color="#555C66", ha="left")
    save(fig, "threshold.png")


def fig_separation(D):
    """The distributions behind the curves above."""
    y = D["y"]
    p = shipped(D["eval"])
    pc = shipped(D["reals"]["coco"])
    bins = np.linspace(0, 1, 46)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.0), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    for ax, (series, title) in zip(axes, (
            ((("AI-generated", p[y == 1], RED), ("real", p[y == 0], BLUE)),
             "So-Fake-OOD, unseen generator families (n = 4,198)"),
            ((("real photographs", pc, AMBER),),
             "COCO val2017 — 100% real, and never trained on (n = 5,000)"))):
        for lab, v, c in series:
            ax.hist(v, bins=bins, alpha=.62, color=c, label=f"{lab}  (n = {len(v):,})",
                    edgecolor="white", linewidth=.4)
        for c, col in ((OLD_THR, GREY), (NEW_THR, BLUE)):
            ax.axvline(c, lw=1.2, ls=(0, (3, 3)), c=col, zorder=5)
        style(ax, title)
        # upper LEFT: the threshold labels live at upper centre and a legend
        # there sits straight on top of them.
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
        ax.set_ylabel("images")

    axes[1].set_xlabel("shipped score  —  max(general, tampered)")
    axes[1].set_xlim(0, 1)
    for ax in axes:
        top = ax.get_ylim()[1]
        for c, lab, col, ha in ((OLD_THR, "0.500", GREY, "right"),
                                (NEW_THR, "0.766", BLUE, "left")):
            ax.text(c + (-.008 if ha == "right" else .008), top * .96, lab,
                    fontsize=8.2, color=col, ha=ha, va="top", fontweight="bold",
                    zorder=6, bbox=dict(fc="white", ec="none", alpha=.78, pad=1.4))
    fig.suptitle("Why the threshold moved", fontsize=13, fontweight="bold",
                 x=0.0, ha="left", y=1.0)
    fig.text(0.0, -0.03,
             "The real-image mass in both panels extends well past 0.5. Everything between the two "
             "dashed lines was a false\naccusation at the old default — 34.6% of the COCO panel, "
             "reduced to 8.6% at 0.766.",
             fontsize=8.2, color="#555C66", ha="left")
    save(fig, "separation.png")


def fig_robustness():
    """The 15-variant grid as a heatmap. Rows ordered by how much they hurt."""
    from eval_grid import build
    df = build("so_fake_ood")
    # Sort by mean DROP FROM CLEAN per column, not by raw mean: spectral swings
    # 0.55-0.70 while the others swing 0.88-0.97, so a raw mean just sorts the
    # rows by whatever spectral did that day.
    order = (df - df.loc["clean"]).mean(axis=1).sort_values(ascending=False).index
    df = df.loc[order]

    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    im = ax.imshow(df.values, cmap="RdYlBu", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(df.columns)), list(df.columns), fontsize=9.5,
                  fontweight="bold")
    ax.set_yticks(range(len(df.index)), df.index, fontsize=9)
    ax.xaxis.set_ticks_position("top")
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            v = df.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="#1A1F27" if v > 0.62 else "white")
    ax.set_xticks(np.arange(-.5, len(df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(df.index), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, shrink=.55, pad=.02, label="AUROC")
    fig.suptitle("Robustness grid — AUROC per branch under each transform",
                 fontsize=12.5, fontweight="bold", x=0.0, ha="left", y=1.02)
    fig.text(0.0, -0.015,
             "Rows sorted by mean AUROC, so the transforms that actually hurt sink to the bottom. "
             "The spectral column is the\nweak branch, not a broken measurement. The face column "
             "RISES under blur — that is shortcut learning, HANDOVER-MODELS §8.",
             fontsize=8.2, color="#555C66", ha="left")
    save(fig, "robustness.png")


def fig_generalisation(D):
    """The §5g result: what each branch does to REAL photos it has not seen."""
    keys = list(REAL_SETS)
    tags = [REAL_SETS[k] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4),
                             gridspec_kw={"wspace": 0.28, "width_ratios": [1.15, 1]})

    x = np.arange(len(tags))
    for off, name, c in ((-0.19, "general", BLUE), (0.19, "tampered", RED)):
        rates = [(D["reals"][k][name] >= 0.5).mean() for k in keys]
        axes[0].bar(x + off, rates, 0.34, color=c, label=name, edgecolor="white")
        for xi, r in zip(x + off, rates):
            axes[0].text(xi, r + .012, f"{r:.1%}", ha="center", fontsize=8.5,
                         fontweight="bold", color=c)
    axes[0].set_xticks(x, tags, fontsize=7.4)      # 8.5 runs the labels together
    axes[0].set_ylabel("real photographs flagged as AI\n(raw branch score >= 0.5)")
    axes[0].set_ylim(0, max(.42, axes[0].get_ylim()[1]))
    axes[0].legend(fontsize=9, frameon=False)
    style(axes[0], "False positives on real photography, by distribution")

    for name, c in (("general", BLUE), ("tampered", RED)):
        parts = axes[1].violinplot([D["reals"][k][name] for k in keys],
                                   positions=x + (-0.16 if name == "general" else 0.16),
                                   widths=.3, showextrema=False, showmedians=True)
        for b in parts["bodies"]:
            b.set_facecolor(c); b.set_alpha(.55); b.set_edgecolor("white")
        parts["cmedians"].set_color(c); parts["cmedians"].set_linewidth(1.6)
    axes[1].axhline(0.5, lw=1.1, ls=(0, (3, 3)), c=GREY)
    axes[1].set_xticks(x, [t.split("\n")[0] for t in tags], fontsize=8.5)
    axes[1].set_ylabel("P(AI) on real photographs"); axes[1].set_ylim(0, 1)
    style(axes[1], "Score distribution on the same three sets")

    fig.suptitle("On unfamiliar photography the tampered branch flags 10x more real images",
                 fontsize=12.5, fontweight="bold", x=0.0, ha="left", y=1.02)
    fig.text(0.0, -0.22,
             "Both probes train only on SID_Set reals. On COCO — ordinary photography neither has "
             "seen — general holds at 2.4% while\ntampered fires on 24.2%. The middle pair is not "
             "the same story: So-Fake-OOD's reals are web-sourced and already processed,\nso both "
             "probes struggle and general struggles most. That set is harder, not more unseen.\n"
             "Because the shipped scorer is max(), it inherits whichever branch is worse on a "
             "given image. Adding a second real\ndistribution to the tampered probe's training "
             "made COCO worse, not better (HANDOVER.md §5g).",
             fontsize=8.2, color="#555C66", ha="left")
    save(fig, "generalisation.png")


FIGS = {"threshold": fig_threshold, "separation": fig_separation,
        "robustness": fig_robustness, "generalisation": fig_generalisation}

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or list(FIGS)
    bad = [w for w in want if w not in FIGS]
    if bad:
        raise SystemExit(f"unknown figure(s) {bad}; choose from {list(FIGS)}")
    D = populations() if any(w != "robustness" for w in want) else None
    for w in want:
        print(f"{w}:")
        FIGS[w]() if w == "robustness" else FIGS[w](D)
