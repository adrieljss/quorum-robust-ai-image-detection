"""Data-flow diagram -> docs/figures/datasets.drawio.

    app.diagrams.net, File > Open From > Device

Answers three questions a judge will ask about the numbers:
  which dataset trains which branch, what the degradation grid does to each,
  and which sets were never trained on.

Counts are read from data/manifests/main.csv at generation time, and the KEEP
variant list from quorum.detectors.general, so the diagram cannot drift from the
manifest the way a hand-typed table would.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quorum.degrade import TRANSFORMS  # noqa: E402
from quorum.detectors.general import KEEP  # noqa: E402

OUT = ROOT / "docs" / "figures" / "datasets.drawio"

BOX = "rounded=1;whiteSpace=wrap;html=1;arcSize=10;"
TRAIN = "fillColor=#dae8fc;strokeColor=#6c8ebf;"
CALIB = "fillColor=#ffe6cc;strokeColor=#d79b00;"
EVAL = "fillColor=#d5e8d4;strokeColor=#82b366;"
LOCK = "fillColor=#f8cecc;strokeColor=#b85450;"
GREY = "fillColor=#f5f5f5;strokeColor=#999999;"
YELLOW = "fillColor=#fff2cc;strokeColor=#d6b656;"
BRANCH = "fillColor=#e1d5e7;strokeColor=#9673a6;"
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#666666;"
        "strokeWidth=2;endArrow=block;")
SOFT = EDGE + "dashed=1;strokeWidth=1;"
EVAL_EDGE = EDGE + "dashed=1;strokeColor=#82b366;"

nodes, edges = [], []


def box(i, label, x, y, w, h, style=BOX + GREY, font=12):
    nodes.append((i, label, x, y, w, h, style + f"fontSize={font};"))
    return i


def txt(i, label, x, y, w, h, style="align=left;", font=11):
    nodes.append((i, label, x, y, w, h,
                  "text;html=1;whiteSpace=wrap;verticalAlign=middle;"
                  f"fontSize={font};" + style))
    return i


def arrow(a, b, label="", style=EDGE):
    edges.append((a, b, label, style))


# --- facts, from the manifest --------------------------------------------
M = pd.read_csv(ROOT / "data" / "manifests" / "main.csv")
C = M[M.variant == "clean"]


def n(source, split=None):
    g = C[C.source == source]
    if split:
        g = g[g.split == split]
    return len(g)


CACHE = ROOT / "data" / "cache" / "embeddings" / "vitl14_v1"
_SIZES = {}
for _f in CACHE.glob("*.npy"):
    _src = _f.stem.rsplit("_", 1)[0]
    for _pre in ("face_", "spec_"):
        if _src.startswith(_pre):
            _src = _src[len(_pre):]
    _SIZES[_src] = _SIZES.get(_src, 0) + _f.stat().st_size


# Raw image bytes. MEASURED where we kept the pixels; for the two streamed
# corpora there is nothing on disk to measure, so the figure is the source
# footprint those images represent and is marked with a tilde. The gap between
# these numbers and the cache column is the point: 768 floats replace a photo.
# SOURCE size -- what you would download -- not what sits in data/raw.
#
# The distinction bit once already. data/raw/images/sid_train looks like the
# dataset but is `--save-images` output: normalised q95 JPEGs at 239 KB, our own
# re-encodes, and only 12,000 of the 16,000 images. Reporting that as SID_Set's
# size understated it 7x. Source images are sampled from the repo instead:
# SID_Set 1.21 MB/image (30 sampled), So-Fake-OOD 2.68 MB/image (40 sampled),
# both as PNG, which is how the parquet stores them.
#
# (True, False) = measured from local originals; (True,) with est=True = image
# count x a sampled mean.
SID_MB, SOFAKE_MB = 1.21, 2.68
RAW_MB = {
    "sid_train": (16000 * SID_MB, True),
    "sid_tampered": (3949 * SID_MB, True),
    "sid_tampered_eval": (1499 * SID_MB, True),
    "so_fake_ood": (6242 * SOFAKE_MB, True),
    "so_fake_tampered_eval": (3000 * SOFAKE_MB, True),
    # These we downloaded as files, so the bytes on disk ARE the source bytes.
    "wildfake_midjourney": (2343, False),
    "coco_train_reals": (818, False),
    "real_holdout_laion": (195, False),
    "organizer_val": (2715, False),
}


def size(source):
    """raw bytes / manifest rows / embedding cache, as one label line.

    Rows, not just images, because rows are what the branches actually see --
    the 15-variant grid multiplies every source, and a reader comparing "16,000"
    against "1,500" would miss that the grid is where the volume comes from.
    """
    rows = len(M[M.source == source])
    mb = _SIZES.get(source, 0) / 1e6
    raw, est = RAW_MB.get(source, (0, False))
    r = (f"~{raw / 1000:.1f} GB" if raw >= 1000 else f"~{raw:.0f} MB") if est else         (f"{raw / 1000:.1f} GB" if raw >= 1000 else f"{raw:.0f} MB")
    return f"{r} raw &#8594; {mb:.0f} MB cached &#183; {rows:,} rows"


def mix(source, split=None):
    g = C[C.source == source]
    if split:
        g = g[g.split == split]
    v = g.label.value_counts().to_dict()
    return v.get(0, 0), v.get(1, 0)


n_variants = M.variant.nunique()
fams = sorted(C[(C.split == "calib_ood") & (C.label == 1)].generator.unique())
n_test_gen = C[(C.split == "test_ood") & (C.source == "so_fake_ood")
               & (C.label == 1)].generator.nunique()

box("title", "<b>Quorum &#8212; data, training and evaluation</b>"
             "<br><font style='font-size:13px;color:#666666'>"
             f"every image is scored under all {n_variants} degradation settings "
             "&#183; nothing evaluated was trained on</font>",
    40, 18, 780, 56,
    "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontStyle=1;",
    font=21)

# =========================================================================
# TRAINING SOURCES
# =========================================================================
box("h_train", "<b>TRAINING DATA</b>", 40, 100, 300, 26,
    "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontStyle=1;"
    "fontColor=#6c8ebf;", font=14)

r0, a0 = mix("sid_train", "train")
box("sid", f"<b>SID_Set</b> &#183; train<br>{n('sid_train', 'train'):,} images &#183; {r0:,} real / {a0:,} AI"
           f"<br><font style='font-size:10px'>{size('sid_train')}</font>",
    40, 128, 205, 78, BOX + TRAIN)
r1, a1 = mix("sid_tampered", "train")
box("sidtamp", f"<b>SID_Set tampered</b><br>{n('sid_tampered', 'train'):,} AI-inpainted real photos"
               f"<br><font style='font-size:10px'>{size('sid_tampered')}</font>",
    40, 218, 205, 78, BOX + TRAIN)
box("mj", f"<b>WildFake Midjourney</b><br>{n('wildfake_midjourney', 'train'):,} images &#183; all AI"
          f"<br><font style='font-size:10px'>{size('wildfake_midjourney')}</font>",
    40, 308, 205, 66, BOX + TRAIN)
box("coco", f"<b>COCO train2017</b><br>{n('coco_train_reals', 'train'):,} real photos &#183; "
            f"<font style='font-size:10px'>{size('coco_train_reals')}</font>"
            "<br><i>tried as negatives, reverted</i>",
    40, 386, 205, 70, BOX + GREY)

# --- the degradation grid ------------------------------------------------
grid_txt = " &#183; ".join(f"{k}({len(v)})" for k, v in TRANSFORMS.items())
box("grid", f"<b>Degradation grid</b> &#8212; {n_variants} settings<br>"
            f"clean + {grid_txt}<br><br>"
            "<b>TRAIN: clean + 3 RANDOM per image</b><br>"
            "seeded on image_id, so a rerun reproduces it. Every setting "
            f"appears across the source (~3,400x in SID_Set), no single image "
            "sees them all.<br><br>"
            f"<b>EVAL: all {n_variants}, every image</b>",
    272, 150, 268, 190, BOX + YELLOW, font=11)
txt("n_keep", f"<b>KEEP</b> = {', '.join(KEEP)} is a different thing: a "
              "training-time filter on ADDED sources only, so a source with a "
              "different variant density<br>cannot outvote SID_Set. Midjourney "
              "keeps 1.6 rows/image through it, calib_ood 4.0. SID_Set itself "
              "is not filtered &#8212; it uses all 4 of its sampled rows.",
    40, 458, 920, 42)

# =========================================================================
# BRANCHES
# =========================================================================
box("h_branch", "<b>WHAT EACH BRANCH TRAINS ON</b>", 590, 100, 400, 26,
    "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontStyle=1;"
    "fontColor=#9673a6;", font=14)

box("b_gen", "<b>general</b><br>SID_Set + Midjourney + 4 of 5 calib families"
             "<br><i>real vs AI</i>",
    590, 132, 250, 72, BOX + BRANCH)
box("b_tam", "<b>tampered</b><br>SID_Set reals vs SID_Set tampered"
             "<br><i>never sees a fully synthetic image</i>",
    590, 218, 250, 72, BOX + BRANCH)
box("b_face", "<b>face</b><br>face crops from SID_Set<br><i>one probe, 769-d</i>",
    590, 304, 250, 62, BOX + BRANCH)
box("b_disp", "<b>spectral</b> &#183; <b>text</b><br>SID_Set (FFT / OCR crops)"
              "<br><i>display only, not in pred</i>",
    590, 380, 250, 66, BOX + GREY)

for a, b in [("sid", "b_gen"), ("mj", "b_gen"), ("sid", "b_tam"),
             ("sidtamp", "b_tam"), ("sid", "b_face"), ("sid", "b_disp")]:
    arrow(a, b, "", SOFT)

# =========================================================================
# THE CALIBRATION CARVE
# =========================================================================
box("h_cal", "<b>THE ONE HELD-OUT SET WE ARE ALLOWED TO FIT ON</b>",
    40, 506, 700, 26,
    "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontStyle=1;"
    "fontColor=#d79b00;", font=14)

r2, a2 = mix("so_fake_ood", "calib_ood")
box("calib", f"<b>So-Fake-OOD &#183; calib_ood</b><br>"
             f"{n('so_fake_ood', 'calib_ood'):,} images &#183; {r2:,} real / "
             f"{a2:,} AI<br>{len(fams)} generator families:<br>"
             + ", ".join(fams),
    40, 538, 260, 92, BOX + CALIB)
box("rot", f"<b>{len(fams)}-fold rotation</b><br>train on {len(fams) - 1} "
           "families, calibrate<br>on the 5th, rotate, average the<br>"
           f"{len(fams)} calibrated weight vectors",
    340, 538, 250, 92, BOX + CALIB)

txt("n_cal", "Split by <b>family</b>, not by generator: Ideogram2 and Ideogram3 "
             "on opposite sides would call a sibling model &#8220;unseen&#8221;. "
             "Reals carry no generator,<br>so they fold by <b>image</b> instead "
             "&#8212; otherwise one photo lands in train and calibration at "
             "once. Platt is folded into the weights, so nothing extra ships.",
    40, 640, 900, 40)

arrow("calib", "rot", "", SOFT)
arrow("rot", "b_gen", "", SOFT)

# =========================================================================
# EVALUATION
# =========================================================================
box("h_eval", "<b>EVALUATION &#8212; never trained on</b>", 990, 100, 460, 26,
    "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontStyle=1;"
    "fontColor=#82b366;", font=14)

r3, a3 = mix("so_fake_ood", "test_ood")
box("e_ood", f"&#127942; <b>So-Fake-OOD</b> &#183; test_ood &#8212; HEADLINE<br>"
             f"{n('so_fake_ood', 'test_ood'):,} images &#183; {r3:,} real / "
             f"{a3:,} AI<br><b>{n_test_gen} generator families, none in training"
             "</b>",
    990, 132, 300, 72, BOX + EVAL)
box("e_org", f"&#128274; <b>Organizer set</b> &#183; test_organizer<br>"
             f"{n('organizer_val', 'test_organizer'):,} images &#8212; COCO val2017 + WildFake DALL&#183;E"
             f"<br><font style='font-size:10px'>{size('organizer_val')}</font>"
             "<br><b>the brief forbids training on this</b>",
    990, 214, 310, 82, BOX + LOCK)
box("e_tamp", f"<b>SID_Set tampered</b> &#183; eval<br>"
              f"{n('sid_tampered_eval', 'test_ood'):,} edited photos, same corpus as training"
              f"<br><font style='font-size:10px'>{size('sid_tampered_eval')}</font>",
    990, 306, 310, 68, BOX + EVAL)
box("e_ftamp", f"<b>So-Fake tampered</b> &#183; FOREIGN edits<br>"
               f"{n('so_fake_tampered_eval', 'test_ood'):,} images &#8212; a different corpus"
               f"<br><font style='font-size:10px'>{size('so_fake_tampered_eval')}</font>"
               "<br><i>the honest cross-corpus number</i>",
    990, 384, 310, 76, BOX + EVAL)
box("e_laion", f"<b>LAION real holdout</b><br>"
               f"{n('real_holdout_laion', 'test_holdout'):,} real web photos, corpus-disjoint"
               f"<br><font style='font-size:10px'>{size('real_holdout_laion')}</font>"
               "<br><i>false positives on unfamiliar reals</i>",
    990, 470, 310, 76, BOX + EVAL)

for e in ("e_ood", "e_org", "e_tamp", "e_ftamp", "e_laion"):
    arrow("b_gen", e, "", EVAL_EDGE)

txt("n_eval", "&#127942; the number we quote &#183; &#128274; quarantined by "
              "the problem statement &#183; every eval set is scored under all "
              f"{n_variants} settings, not just clean",
    990, 556, 470, 40)

_raw_gb = sum(v for v, _ in RAW_MB.values()) / 1000
box("total", f"<b>~{_raw_gb:.0f} GB of images &#8594; "
             f"{sum(_SIZES.values()) / 1e9:.1f} GB of embeddings</b><br>"
             f"{len(M):,} rows across {C.source.nunique()} sources. Each photo "
             "becomes 768 floats,<br>so the corpora are read <b>once</b> and "
             "never stored again &#8212; the<br>whole project fits on a laptop. "
             "<i>~ = streamed, never on disk.</i>",
    272, 350, 268, 96, BOX + YELLOW, font=10)

box("rule", "<b>The rule that makes the numbers mean something.</b> No image and "
            "no generator family crosses the train/eval line &#8212; "
            "build_manifest.py asserts both.<br>"
            f"The {n_test_gen} generators in the headline set appear nowhere in "
            "training, so the score is what happens on a model nobody has seen. "
            "The LAION pool<br>goes further: a different <b>corpus</b>, not just "
            "different images, because sharing a corpus flatters a "
            "false-positive rate.",
    40, 692, 1410, 74, BOX + YELLOW, font=12)


def render():
    cells = []
    for i, label, x, y, w, h, style in nodes:
        cells.append(
            f'        <mxCell id="{i}" value={quoteattr(label)} style="{style}" '
            f'vertex="1" parent="1">\n'
            f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry" />\n        </mxCell>')
    for k, (a, b, label, style) in enumerate(edges):
        cells.append(
            f'        <mxCell id="e{k}" value={quoteattr(label)} style="{style}" '
            f'edge="1" parent="1" source="{a}" target="{b}">\n'
            f'          <mxGeometry relative="1" as="geometry" />\n        </mxCell>')
    body = "\n".join(cells)
    return f'''<mxfile host="app.diagrams.net">
  <diagram name="Quorum data">
    <mxGraphModel dx="1600" dy="900" grid="0" gridSize="10" guides="1" \
tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" \
pageWidth="1500" pageHeight="800" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
{body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}  ({len(nodes)} nodes, {len(edges)} edges)")
    print(f"  {n_variants} variants, KEEP={KEEP}")
    print(f"  calib families: {fams}")
    print(f"  headline eval: {n('so_fake_ood', 'test_ood'):,} images, "
          f"{n_test_gen} unseen generators")
