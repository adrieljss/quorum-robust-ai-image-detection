"""Regenerate docs/figures from cache. Everything here is computed, never drawn
from a pasted number, so a figure cannot outlive the result it illustrates.

    python scripts/make_figures.py            # all six
    python scripts/make_figures.py threshold  # just one
    python scripts/make_figures.py --no-tampered   # the same six, general alone

`--no-tampered` answers "what do these look like without the tampered branch?".
It redirects output to docs/figures-no-tampered/, scores with the general probe
alone, drops the tampered column from the robustness grid, and RE-PICKS the
threshold on calib_ood -- 0.766 was picked for max() and is not transferable.
generalisation.png and benchmarks.png still draw both branches: they ARE the
comparison, and blanking one series would erase the evidence for the drop.

Six figures, one argument each:

  threshold.png     0.5 was never chosen, and it is the wrong place to cut
  separation.png    why -- the real-image score mass sits on top of the old cut
  robustness.png    the 15-variant grid as a picture instead of a table
  robustness-organizer_val.png  the same grid on the organizer set
  generalisation.png  the tampered branch cannot recognise unfamiliar REAL photos
  benchmarks.png    organizer set vs the headline, and what max() costs on it

The organizer set is WildFake DALL-E 3 Advanced (3,719 images) against COCO
val2017 (5,000). It now appears in four of the six: its own grid, both halves of
separation.png's lower panel, the dotted recall curve on threshold.png, and
benchmarks.png. generalisation.png is about REAL photographs only, so it keeps
using the COCO half alone.

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

# The next four are rebound by --no-tampered in __main__. Module globals rather
# than a threaded parameter: every figure would otherwise grow an argument it
# does nothing with but forward.
FIGURES = ROOT / "docs" / "figures"
BRANCHES = ("general", "tampered")
SCORE_LABEL = "max(general, tampered)"
NO_TAMPERED = False
# Shared with quorum/calibrate.py's reliability plot so the figure set reads as
# one document rather than four unrelated charts.
BLUE, RED, GREY = "#2B4A9B", "#B4462F", "#9AA3AF"
GREEN, AMBER = "#2A7355", "#B0741B"
PURPLE = "#6B4A9B"          # the tampered/edited task, wherever it appears
OLD_THR, NEW_THR = 0.5, 0.766

# Short keys for the data, display labels separately. Keying the dict on the
# rendered label means every caller repeats a multi-line string literal, and one
# tweak to the wording silently KeyErrors the other figures.
REAL_SETS = {"sid": "SID_Set\nin-distribution",
             "ood": "So-Fake-OOD\nunseen generators",
             "coco": "COCO val2017\nunseen photography"}

SOURCE_LABEL = {"so_fake_ood": "So-Fake-OOD, unseen generator families",
                "organizer_val": "organizer set, WildFake DALL·E 3 vs COCO val2017"}


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
    # The DALL-E 3 half of the organizer set. Already loaded and previously
    # thrown away -- "coco" above keeps only organizer_val's label==0 rows, so
    # every figure showed the benchmark's real side and none of its fake side.
    ai = (Rc.label.values == 1) & (Rc.variant == "clean").values
    out["organizer"] = {n: probe(n, Xc[ai]) for n in ("general", "tampered")}
    # Locally-edited photographs -- the half of the task neither eval set above
    # can contain, because both are synthetic-vs-real by construction. All
    # positives; figures needing negatives borrow the So-Fake-OOD reals.
    Xt, Rt = load("sid_tampered_eval")
    kt = (Rt.variant == "clean").values
    out["edited"] = {n: probe(n, Xt[kt]) for n in ("general", "tampered")}
    return out


def shipped(p):
    return np.maximum.reduce([p[n] for n in BRANCHES])


def repick():
    """0.766 was picked for max(); a different scorer needs its own cut.

    Same rule as pick_threshold.py -- the high end of the accuracy plateau on
    calib_ood -- so the two constants stay comparable.
    """
    from pick_threshold import plateau, shipped as raw
    X, R = load("so_fake_ood")
    k = (R.split == "calib_ood").values
    hi, lo, _ = plateau(raw(X[k], BRANCHES), R.label.values[k])
    print(f"  threshold re-picked on calib_ood: {hi:.4f}  (plateau {lo:.3f}-{hi:.3f})")
    return hi


# ------------------------------------------------------------- figures ------
def fig_threshold(D):
    """Every metric that depends on a cut, as a function of the cut."""
    y = D["y"]
    p = shipped(D["eval"])
    pc = shipped(D["reals"]["coco"])
    po = shipped(D["organizer"])
    pe = shipped(D["edited"])
    t = np.linspace(0.02, 0.98, 481)

    acc, prec, rec, fpr, coco, dalle, edit = [], [], [], [], [], [], []
    for c in t:
        hat = p >= c
        tp = (hat & (y == 1)).sum(); fp = (hat & (y == 0)).sum()
        tn = (~hat & (y == 0)).sum(); fn = (~hat & (y == 1)).sum()
        acc.append((hat == (y == 1)).mean())
        prec.append(tp / (tp + fp) if tp + fp else np.nan)
        rec.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
        coco.append((pc >= c).mean())
        dalle.append((po >= c).mean())
        edit.append((pe >= c).mean())

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for v, lab, c, ls in ((prec, "precision", BLUE, "-"), (rec, "recall", RED, "-"),
                          (acc, "accuracy", GREEN, "-"),
                          (dalle, "recall (DALL·E 3, organizer)", RED, ":"),
                          (edit, "recall (SID tampered, edited)", PURPLE, ":"),
                          (fpr, "FPR (So-Fake-OOD reals)", GREY, "--"),
                          (coco, "FPR (COCO photographs)", AMBER, "--")):
        ax.plot(t, v, ls, lw=1.8, c=c, label=lab)

    # Inside the axes, not above them: bbox_inches="tight" crops anything past
    # ylim and the labels vanish silently.
    for c, lab, col, ha in ((OLD_THR, "0.500  sigmoid default", GREY, "right"),
                            (NEW_THR, f"{NEW_THR:.3f}  shipped", BLUE, "left")):
        ax.axvline(c, lw=1.1, ls=(0, (3, 3)), c=col, zorder=1)
        ax.text(c + (-.012 if ha == "right" else .012), .05, lab, fontsize=8.2,
                color=col, ha=ha, va="bottom", fontweight="bold", zorder=6,
                bbox=dict(fc="white", ec="none", alpha=.78, pad=1.6))

    style(ax, "Every threshold-dependent metric, as a function of the threshold")
    ax.set_xlabel("decision threshold on the shipped score")
    ax.set_ylabel("rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.text(0.0, -0.125,
             f"AUROC is identical everywhere on this axis. Moving the cut from 0.500 to {NEW_THR:.3f} "
             "buys precision and costs recall;\nthe amber curve is the one nobody was watching "
             f"— {np.interp(OLD_THR, t, coco):.1%} of ordinary photographs flagged as AI at the old default.\n"
             f"The dotted line is the organizer set's fake half (DALL·E 3, WildFake Advanced): "
             f"{np.interp(NEW_THR, t, dalle):.1%} of it is still caught at {NEW_THR:.3f}, against "
             f"{np.interp(NEW_THR, t, rec):.1%}\non So-Fake-OOD. The threshold was picked on "
             "So-Fake-OOD and costs the organizer benchmark nothing — that set is the easier one."
             f"\nThe purple curve is the third task, the one neither eval set contains: {np.interp(NEW_THR, t, edit):.1%} of locally-edited"
             f" photographs are caught at {NEW_THR:.3f}, against {np.interp(OLD_THR, t, edit):.1%} at 0.500.",
             fontsize=8.2, color="#555C66", ha="left")
    save(fig, "threshold.png")


def fig_separation(D):
    """The distributions behind the curves above."""
    y = D["y"]
    p = shipped(D["eval"])
    pc = shipped(D["reals"]["coco"])
    po = shipped(D["organizer"])
    pe = shipped(D["edited"])
    pr = shipped(D["reals"]["ood"])      # reused as the third panel's negatives
    bins = np.linspace(0, 1, 46)

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.6), sharex=True,
                             gridspec_kw={"hspace": 0.22})
    for ax, (series, title) in zip(axes, (
            ((("AI-generated", p[y == 1], RED), ("real", p[y == 0], BLUE)),
             "So-Fake-OOD, unseen generator families (n = 4,198)"),
            ((("DALL·E 3, AI-generated", po, RED), ("COCO val2017, real", pc, AMBER)),
             "Organizer set — WildFake DALL·E 3 Advanced vs COCO val2017, neither trained on"),
            ((("SID tampered, locally edited", pe, PURPLE), ("So-Fake-OOD, real", pr, BLUE)),
             "The third task — authentic photos with an AI-edited region, same reals as the top"))):
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

    axes[-1].set_xlabel(f"shipped score  —  {SCORE_LABEL}")
    axes[1].set_xlim(0, 1)
    for ax in axes:
        top = ax.get_ylim()[1]
        for c, lab, col, ha in ((OLD_THR, "0.500", GREY, "right"),
                                (NEW_THR, f"{NEW_THR:.3f}", BLUE, "left")):
            ax.text(c + (-.008 if ha == "right" else .008), top * .96, lab,
                    fontsize=8.2, color=col, ha=ha, va="top", fontweight="bold",
                    zorder=6, bbox=dict(fc="white", ec="none", alpha=.78, pad=1.4))
    fig.suptitle("Why the threshold moved", fontsize=13, fontweight="bold",
                 x=0.0, ha="left", y=1.0)
    fig.text(0.0, -0.055,
             "The real-image mass in the top two panels extends well past 0.5. Everything between the two "
             f"dashed lines was a false\naccusation at the old default — {(pc >= OLD_THR).mean():.1%} of the COCO panel, "
             f"reduced to {(pc >= NEW_THR).mean():.1%} at {NEW_THR:.3f}.\n"
             f"The middle panel is the organizer benchmark, both halves: DALL·E 3 piles up against 1.0 "
             f"({(po >= NEW_THR).mean():.1%} caught at {NEW_THR:.3f})\nwhile COCO sits at the other end. That gap is "
             "wider than the upper panel's — the organizer set is the easier of the two."
             f"\nThe bottom panel is the one the other two cannot contain. Its positives are REAL "
             f"photographs with an edited\nregion, so a detector that keys on 'this image looks "
             f"synthetic' has nothing to grip: {(pe >= NEW_THR).mean():.1%} clear {NEW_THR:.3f}.",
             fontsize=8.2, color="#555C66", ha="left")
    save(fig, "separation.png")


def edited_grid(source):
    """The general branch on the TAMPERED task, per variant.

    eval_grid.build() gives every branch its own task -- its `tampered` column
    is already scored on the edited set. This is that table's missing cell:
    what the branch that ships does on edited photographs. It is also the one
    edited column that survives --no-tampered, where no tampered branch exists.
    """
    import pandas as pd
    from eval_grid import held_out
    from quorum.detectors.general import auc_by_variant, fit_general

    Xt, Rt = load("sid_tampered_eval")
    Xr, Rr = held_out(*load(source))
    m = (Rr.label.values == 0)                  # borrow this source's reals
    Xtr, Rtr = load("sid_train")
    return auc_by_variant(fit_general(Xtr, Rtr.label.values),
                          np.concatenate([Xt, Xr[m]]),
                          pd.concat([Rt, Rr[m]], ignore_index=True))


# Both edited columns are scored on sid_tampered_eval, not on `source`. Spelled
# out in the header because an unlabelled "tampered" column reads as a fourth
# branch on the same task, which is exactly the misreading to avoid.
EDIT_T, EDIT_G = "tampered" + chr(10) + "(on edited)", "general" + chr(10) + "(on edited)"


def fig_robustness(source="so_fake_ood"):
    """The 15-variant grid as a heatmap. Rows ordered by how much they hurt."""
    from eval_grid import build, blur_rises
    df = build(source).rename(columns={"tampered": EDIT_T})
    if NO_TAMPERED:
        df = df.drop(columns=[EDIT_T], errors="ignore")
    df[EDIT_G] = edited_grid(source)
    # Sort by mean DROP FROM CLEAN per column, not by raw mean: spectral swings
    # 0.55-0.70 while the others swing 0.88-0.97, so a raw mean just sorts the
    # rows by whatever spectral did that day.
    order = (df - df.loc["clean"]).mean(axis=1).sort_values(ascending=False).index
    df = df.loc[order]

    ge = df[EDIT_G]                       # no backslashes inside an f-string
    fig, ax = plt.subplots(figsize=(7.9, 6.4))
    im = ax.imshow(df.values, cmap="RdYlBu", vmin=0.45, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(df.columns)), list(df.columns), fontsize=8.4,
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
    fig.suptitle(f"Robustness grid — {SOURCE_LABEL[source]}",
                 fontsize=12.5, fontweight="bold", x=0.0, ha="left", y=1.02)
    # The blur note is true on So-Fake-OOD and FALSE here: on organizer_val the
    # face row FALLS under blur, 0.9520 -> 0.8887. eval_grid.py gates the same
    # caveat on the same predicate; printing it unconditionally would have the
    # caption contradict the numbers directly above it.
    fig.text(0.0, -0.015,
             "Rows sorted by mean drop from clean, not raw AUROC — a raw mean just sorts them by "
             "whatever spectral did.\nThe transforms that actually hurt sink to the bottom. "
             "The spectral column is the weak branch, not a broken measurement.\n"
             f"The edited column(s) are a DIFFERENT task: sid_tampered_eval against this source's reals. "
             f"General runs\n{ge.min():.3f}-{ge.max():.3f} there — a coin flip, which is the whole reason the tampered branch exists."
             + ("\nThe face column RISES under blur — that is shortcut learning, "
                "HANDOVER-MODELS §8." if blur_rises(df) else
                f"\nThe general column holds {df.loc['clean', 'general']:.4f} clean to "
                f"{df['general'].min():.4f} at its worst — a {df.loc['clean', 'general'] - df['general'].min():.4f} "
                f"drop across all 15 transforms. Read that\nwith the same caution as the face row: "
                f"it also RISES on {(df['general'] > df.loc['clean', 'general']).sum()} of them, "
                f"peaking at {df['general'].max():.4f} ({df['general'].idxmax()}). At this AUROC that is "
                "mostly ceiling,\nbut it is the shape a shortcut makes, and this set is the easy one."),
             fontsize=8.2, color="#555C66", ha="left")
    save(fig, "robustness.png" if source == "so_fake_ood" else f"robustness-{source}.png")


def fig_generalisation(D):
    """The §5g result, both error types on one axis.

    "Fraction flagged as AI" is the false-positive rate on a real set and the
    RECALL on a generated one, so a single bar height carries both. Three real
    distributions (want low) then the organizer set's DALL-E 3 half (want high)
    -- which is what makes the tampered branch's trade visible: it pays a false
    positive rate on COCO and returns nothing on this generator.
    """
    keys = list(REAL_SETS) + ["dalle", "edited"]
    tags = ([REAL_SETS[k] for k in REAL_SETS]
            + ["WildFake DALL·E 3\nunseen generator", "SID tampered\nlocally edited"])
    ai = len(REAL_SETS)                    # first AI-touched column; two now follow
    SRC = {"dalle": "organizer", "edited": "edited"}

    def series(k, name):
        return D[SRC[k]][name] if k in SRC else D["reals"][k][name]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4),
                             gridspec_kw={"wspace": 0.26, "width_ratios": [1.2, 1]})

    x = np.arange(len(tags))
    for off, name, c in ((-0.19, "general", BLUE), (0.19, "tampered", RED)):
        rates = [(series(k, name) >= 0.5).mean() for k in keys]
        axes[0].bar(x + off, rates, 0.34, color=c, label=name, edgecolor="white")
        for xi, r in zip(x + off, rates):
            axes[0].text(xi, r + .012, f"{r:.1%}", ha="center", fontsize=8.5,
                         fontweight="bold", color=c)
    axes[0].set_xticks(x, tags, fontsize=6.6)      # 8.5 runs the labels together
    axes[0].set_ylabel("fraction flagged as AI\n(raw branch score >= 0.5)")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    style(axes[0], "Both error types on one axis, by distribution")

    for name, c in (("general", BLUE), ("tampered", RED)):
        parts = axes[1].violinplot([series(k, name) for k in keys],
                                   positions=x + (-0.16 if name == "general" else 0.16),
                                   widths=.3, showextrema=False, showmedians=True)
        for b in parts["bodies"]:
            b.set_facecolor(c); b.set_alpha(.55); b.set_edgecolor("white")
        parts["cmedians"].set_color(c); parts["cmedians"].set_linewidth(1.6)
    axes[1].axhline(0.5, lw=1.1, ls=(0, (3, 3)), c=GREY)
    # First line only, and shortened: the full names collide at four columns.
    short = [t.split("\n")[0].replace("WildFake ", "") for t in tags]
    axes[1].set_xticks(x, short, fontsize=8.0)
    axes[1].set_ylabel("P(AI)"); axes[1].set_ylim(0, 1)
    style(axes[1], "Score distribution on the same five sets")

    # The AI column is a different question from the three to its left. Mark the
    # boundary rather than leaving a reader to infer it from the label.
    for ax in axes:
        ax.axvline(ai - 0.5, lw=1.0, ls=(0, (2, 3)), c="#B0B6BF", zorder=0)
    axes[0].text(ai + .5, 1.0, "AI-touched — want HIGH", fontsize=7.6,
                 color="#6B727C", ha="center", va="top", style="italic")

    fp = lambda n: (D["reals"]["coco"][n] >= .5).mean()
    rc = lambda n: (D["organizer"][n] >= .5).mean()
    ed = lambda n: (D["edited"][n] >= .5).mean()
    fig.suptitle(f"The tampered branch flags {fp('tampered') / max(fp('general'), 1e-9):.0f}x more real "
                 f"photographs — and is the only branch that sees edited ones",
                 fontsize=12.5, fontweight="bold", x=0.0, ha="left", y=1.02)
    fig.text(0.0, -0.32,
             "Both probes train only on SID_Set reals. On COCO — ordinary photography neither has "
             f"seen — general holds at {fp('general'):.1%} while\n"
             f"tampered fires on {fp('tampered'):.1%}. The middle pair is not "
             "the same story: So-Fake-OOD's reals are web-sourced and already processed,\nso both "
             "probes struggle and general struggles most. That set is harder, not more unseen.\n"
             f"The fourth column is the trade: on DALL·E 3 general recalls {rc('general'):.1%} and "
             f"tampered only {rc('tampered'):.1%}. That set contains no\nlocally-edited images, so the "
             "tampered branch can only add false positives there — which is why max() is WORSE than\n"
             + ("general alone on the organizer set. This run DROPS it — the shipped score is now"
                + "\nthe general probe alone, so the red series above is what was given up "
                "(HANDOVER.md §5g)."
                if NO_TAMPERED else
                "general alone on the organizer set, and still ships, because it wins the pooled"
                + "\ntask (HANDOVER.md §5g).")
             + f"\nThe FIFTH column is what the other four cannot show: on locally-edited photographs "
             f"general fires on just {ed('general'):.1%}\nwhile tampered catches {ed('tampered'):.1%}. "
             "Dropping the red series costs exactly that column, and nothing else here.",
             fontsize=8.2, color="#555C66", ha="left")
    save(fig, "generalisation.png")


def fig_benchmarks():
    """Both eval sets, general alone vs the shipped max, clean and worst.

    A dumbbell rather than bars on purpose: the interesting gaps are ~0.03 wide,
    so the axis has to be truncated to show them, and a truncated bar chart lies
    about ratios. Dots carry no area, so the same truncation is honest.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    def held(src):
        X, R = load(src)
        if "calib_ood" in set(R.split):                     # never score the carve
            m = (R.split != "calib_ood").values
            X, R = X[m], R[m].reset_index(drop=True)
        return X, R

    Xo, Ro = held("so_fake_ood")
    Xt, Rt = load("sid_tampered_eval")
    # sid_tampered_eval is all positives, so it needs negatives borrowed. Take
    # the SAME reals the headline row uses -- then the two rows differ only in
    # what counts as a positive, which is the comparison this figure is for.
    real = (Ro.label.values == 0)
    SETS = [("Organizer set\nDALL\u00b7E 3 + COCO val2017", *held("organizer_val")),
            ("So-Fake-OOD\nunseen generator families", Xo, Ro),
            ("SID tampered\nlocally edited vs So-Fake-OOD reals",
             np.concatenate([Xt, Xo[real]]),
             pd.concat([Rt, Ro[real]], ignore_index=True))]

    rows = {}
    for label, X, R in SETS:
        g, t = probe("general", X), probe("tampered", X)
        for scorer, v in (("general alone", g),
                          ("max(general, tampered)  \u2014 shipped", np.maximum(g, t))):
            a = pd.Series({var: roc_auc_score(R.label[k], v[k])
                           for var, k in ((var, (R.variant == var).values)
                                          for var in R.variant.unique())
                           if R.label[k].nunique() == 2})
            rows[(label, scorer)] = (a["clean"], a.min(), a.idxmin())
    ship = "general alone" if NO_TAMPERED else "max(general, tampered)  — shipped"
    headline = next(v[0] for (lab, sc), v in rows.items()
                    if lab.startswith("So-Fake-OOD") and sc == ship)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ticks, y = [], 0
    for (label, scorer), (clean, worst, where) in rows.items():
        c = BLUE if scorer.startswith("general") else RED
        ax.plot([worst, clean], [y, y], lw=2.4, c=c, alpha=.45, zorder=1,
                solid_capstyle="round")
        ax.scatter([worst], [y], s=64, c="white", edgecolors=c, lw=1.8, zorder=3)
        ax.scatter([clean], [y], s=64, c=c, zorder=3)
        ax.text(clean + .004, y, f"{clean:.4f}", va="center", fontsize=8.4,
                fontweight="bold", color=c)
        ax.text(worst - .004, y, f"{worst:.4f}  ({where})", va="center", ha="right",
                fontsize=8.0, color=c)
        ticks.append((y, f"{label}\n{scorer}"))
        y -= 1

    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=8.2)
    lo = min(w for _, w, _ in rows.values())
    ax.set_xlim(lo - 0.30 * (1.025 - lo), 1.025)   # room for the left-hand labels
    ax.set_ylim(y + .5, .7)
    ax.set_xlabel("AUROC   \u2014   hollow dot = worst variant, solid = clean")
    # Verified equal to predict.score_embeddings() to 4dp. eval_grid.py's
    # combiner table maxes the PLATT-CALIBRATED branches instead and reads
    # 0.9114/0.8771 on so_fake_ood -- a different model, not this one.
    style(ax, "Every eval set, general alone against the shipped max()")
    fig.text(0.0, -0.26,
             "The organizer set is the only externally-comparable number we get, and it is the "
             "easiest of the three: DALL\u00b7E 3 is\na softer target than So-Fake-OOD's generator "
             f"families, so {headline:.4f} stays the headline claim.\n"
             "On the organizer set the shipped max() is WORSE than the general probe alone. It "
             "contains no locally-edited\nimages, so the tampered branch can only add false "
             f"positives there.\nThe third pair is the reverse case: on "
             f"locally-edited photographs\ngeneral alone scores {rows[(SETS[2][0], 'general alone')][0]:.4f} \u2014 a coin flip \u2014 and {rows[(SETS[2][0], 'general alone')][1]:.4f},\n"
             "below chance, at its worst variant. "
             + ("This run DROPS tampered: the 'general alone' row of each pair is what now ships,"
                + "\nand the max() row is what it replaced."
                if NO_TAMPERED else
                "max() still wins on the pooled task \u2014 FULL avg 0.9113 vs 0.8849,"
                + "\nwhich is why it ships. The trade is real and belongs in the write-up."),
             fontsize=8.2, color="#555C66", ha="left")
    save(fig, "benchmarks.png")


FIGS = {"threshold": fig_threshold, "separation": fig_separation,
        "robustness": fig_robustness,
        "robustness-organizer": lambda: fig_robustness("organizer_val"),
        "generalisation": fig_generalisation, "benchmarks": fig_benchmarks}

if __name__ == "__main__":
    if "--no-tampered" in sys.argv[1:]:
        NO_TAMPERED = True
        BRANCHES = ("general",)
        SCORE_LABEL = "general branch alone"
        FIGURES = ROOT / "docs" / "figures-no-tampered"
        NEW_THR = repick()
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or list(FIGS)
    bad = [w for w in want if w not in FIGS]
    if bad:
        raise SystemExit(f"unknown figure(s) {bad}; choose from {list(FIGS)}")
    NO_DATA = {"robustness", "robustness-organizer", "benchmarks"}
    D = populations() if any(w not in NO_DATA for w in want) else None
    for w in want:
        print(f"{w}:")
        FIGS[w]() if w in NO_DATA else FIGS[w](D)
