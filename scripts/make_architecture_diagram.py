"""Poster diagram of the whole system -> docs/figures/architecture.drawio.

Open it at app.diagrams.net (File > Open From > Device) and it lays out as saved.

Two lanes:

  SHIPPED   everything that runs on a request. Split by a divider into the
            scoring spine (which produces `pred`) and the band below it
            (reported beside the verdict, never fed back).
  BUILT, NOT WIRED   branches that exist, were measured, and lost. Each carries
            the number that killed it, because "we tried it" is worth nothing
            without "and here is what it scored".

Provenance forks off the INPUT, in parallel with face_crop, and dead-ends at a
report. It is deliberately NOT drawn in series between the image and CLIP:
in-series would say the score depends on it, and the entire argument of
ERROR_ANALYSIS section 7.9 is that it must not.

Every constant is IMPORTED from predict.py or read off the .npz files, for the
same reason the figures import OPERATING_POINT rather than pasting it: a poster
that outlives the model it draws is worse than no poster. Re-run after any
change to the scorer.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import predict  # noqa: E402
from quorum.embed import BACKBONE, PRETRAINED  # noqa: E402

OUT = ROOT / "docs" / "figures" / "architecture.drawio"

BLUE = "fillColor=#dae8fc;strokeColor=#6c8ebf;"
GREY = "fillColor=#f5f5f5;strokeColor=#999999;dashed=1;"
SOLID_GREY = "fillColor=#f5f5f5;strokeColor=#999999;"
ORANGE = "fillColor=#ffe6cc;strokeColor=#d79b00;"
GREEN = "fillColor=#d5e8d4;strokeColor=#82b366;"
YELLOW = "fillColor=#fff2cc;strokeColor=#d6b656;"
RED = "fillColor=#f8cecc;strokeColor=#b85450;"
PURPLE = "fillColor=#e1d5e7;strokeColor=#9673a6;"
BOX = "rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
NOTE = ("text;html=1;whiteSpace=wrap;align=left;verticalAlign=top;"
        "fontSize=10;fontColor=#555555;")
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#666666;"
EDGE_PROV = EDGE + "dashed=1;strokeColor=#9673a6;fontColor=#9673a6;"
LANE = ("swimlane;html=1;rounded=1;arcSize=3;startSize=44;collapsible=0;"
        "fontSize=15;fontStyle=1;align=left;spacingLeft=14;verticalAlign=middle;")

nodes, edges = [], []


def box(i, label, x, y, w, h, style=BOX + BLUE, font=12, parent="1"):
    nodes.append((i, label, x, y, w, h, style + f"fontSize={font};", parent))
    return i


def note(i, label, x, y, w, h, parent="1"):
    nodes.append((i, label, x, y, w, h, NOTE, parent))
    return i


def lane(i, label, x, y, w, h, style):
    nodes.append((i, label, x, y, w, h, LANE + style, "1"))
    return i


def arrow(a, b, label="", style=EDGE):
    edges.append((a, b, label, style))


def params(name):
    """(stored numbers, file KB) straight off the shipped .npz."""
    p = ROOT / "data" / "models" / f"{name}.npz"
    d = np.load(p)
    return sum(int(np.asarray(d[k]).size) for k in d.files), p.stat().st_size / 1024


g_n, g_kb = params("general")
t_n, t_kb = params("tampered")
f_n, f_kb = params("face")
total = g_n + t_n + f_n
shift, op, alpha = predict.SHIFT, predict.OPERATING_POINT, predict.TAMPERED_SCALE

box("title", "Quorum &#8212; how one score is produced, and what else is in the box"
             "<br><font style='font-size:14px;color:#666666'>frozen CLIP &#183; "
             "three linear probes &#183; one max() &#183; "
             f"{total:,} trained numbers</font>",
    40, 30, 1200, 70,
    "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontStyle=1;",
    font=24)

# =========================================================================
# ONE LANE: the scoring spine, a divider, then everything else the request
# computes. Same lane on purpose -- it all runs on the same upload.
# =========================================================================
S = lane("shipped",
         "ON EVERY REQUEST &#8212; above the line produces <b>pred</b>; "
         "below the line is reported and never scored",
         40, 125, 1870, 1010,
         "fillColor=#f7fbf7;strokeColor=#82b366;strokeWidth=2;"
         "swimlaneFillColor=#fbfefb;")

# --- the scoring spine ---------------------------------------------------
box("img", "<b>Input image</b><br>any size<br>jpg / png / webp / bmp",
    20, 190, 180, 80, BOX + GREEN, parent=S)
box("clip", f"<b>CLIP {BACKBONE}</b><br>{PRETRAINED} weights, <b>FROZEN</b>"
            "<br>304M params, fp16<br><i>never fine-tuned</i>",
    260, 165, 230, 100, BOX + GREY, parent=S)
box("v", "<b>v</b> &#8212; 768-d<br>L2-normalised", 560, 190, 150, 70,
    BOX + SOLID_GREY, parent=S)

box("gen", f"<b>general probe</b><br>z<sub>g</sub> = w&#183;v + b<br>"
           f"{g_n:,} numbers &#183; {g_kb:.1f} KB",
    790, 80, 210, 85, parent=S)
box("tam", f"<b>tampered probe</b><br>z<sub>t</sub> = {alpha} &#215; (w&#183;v + b)<br>"
           f"{t_n:,} numbers &#183; {t_kb:.1f} KB",
    790, 215, 210, 85, parent=S)
box("zmax", "<b>z = max(z<sub>g</sub>, z<sub>t</sub>)</b><br>max in <i>logit</i> space",
    1070, 135, 200, 80, BOX + ORANGE, parent=S)
box("sig", f"<b>p = &#963;(z &#8722; {shift:.4f})</b><br>puts the operating<br>"
           f"point {op} at 0.5",
    1350, 130, 210, 90, BOX + ORANGE, parent=S)

box("facedet", "<b>face_crops()</b><br>detect + align <b>every</b><br>"
               "face &#8805; 64px, largest<br>first, capped at 8",
    250, 428, 200, 92, BOX + GREY, parent=S)
box("faceemb", "<b>the same frozen CLIP</b>, one batch<br>"
               "f<sub>1</sub>, f<sub>2</sub>, &#8230; f<sub>N</sub> &#8212; each "
               "<b>769-d</b><br>768 &#8853; standardised log&#8322;(face_px)",
    480, 433, 215, 82, BOX + SOLID_GREY, parent=S)
box("faceprobe", f"<b>face probe</b><br>z<sub>f,i</sub> = w&#183;f<sub>i</sub> + b<br>"
                 f"<i>ONE probe, {f_n:,} numbers,</i><br><i>applied to each crop</i>",
    725, 428, 195, 92, parent=S)
box("facescores", f"<b>p<sub>face,1</sub>, p<sub>face,2</sub>, &#8230;, "
                  f"p<sub>face,N</sub></b><br>&#963;(z<sub>f,i</sub> &#8722; "
                  f"{shift:.4f})<br>one score per face",
    950, 430, 200, 88, BOX + ORANGE, parent=S)
box("facesig", "<b>p<sub>face</sub> = max(p<sub>face,1</sub> &#8230; "
               "p<sub>face,N</sub>)</b><br><b>0.0</b> if N = 0",
    1180, 435, 190, 78, BOX + ORANGE, parent=S)

box("pred", "<b>pred = max(p, p<sub>face</sub>)</b>", 1350, 290, 210, 60,
    BOX + ORANGE, font=14, parent=S)
box("verdict", '<b>verdict</b><br>"AI" if pred &#8805; 0.5<br>else "real"',
    1620, 130, 200, 90, BOX + GREEN, parent=S)
box("json", "<b>predictions.json</b><br>[{image_path, pred}]",
    1620, 290, 200, 70, BOX + GREEN, parent=S)

# --- provenance: forks off the INPUT, in parallel, and dead-ends ---------
box("prov", "<b>provenance.inspect()</b><br>C2PA &#183; EXIF &#183; XMP &#183; "
            "PNG text<br><i>reads the FILE, never a pixel</i>",
    260, 295, 230, 90, BOX + PURPLE, parent=S)
box("prov_out", "<b>provenance report</b><br>a chip beside the verdict<br>"
                "<b>&#9642; dead end &#8212; no path to pred</b>",
    560, 300, 230, 85, BOX + PURPLE, parent=S)

note("n_clip", "<b>Frozen on<br>purpose.</b> Detection<br>costs 769 trained<br>"
               "numbers per branch<br>on a backbone you<br>already run &#8212; and a<br>"
               "2023 backbone has<br>never seen a 2026<br>generator. That is<br>"
               "our largest failure.",
     20, 295, 180, 130, parent=S)
note("n_gen", "trained on SID_Set + WildFake-Midjourney + 4 of 5 calib_ood<br>"
              "generator families; calibration rotated over the 5th and the<br>"
              "five weight vectors averaged",
     780, 20, 340, 55, parent=S)
note("n_tam", f"<b>{alpha} is a policy dial, not a fitted parameter.</b> max() is not"
               "<br>invariant to rescaling one arm, so this alone sets false<br>"
               "positives against edit recall (8.5% / 74.6%). Trained on<br>"
               "tampered-vs-real only &#8212; never sees a synthetic image.",
     840, 310, 345, 75, parent=S)
note("n_max", "<b>max(), not a learned combiner.</b> The task is<br>"
              "disjunctive: either arm firing means AI touched it.<br>"
              "Fusion LR scored lower on both eval sets.",
     1230, 232, 290, 55, parent=S)
note("n_face", "<b>The only branch with DIFFERENT features, and the only one that ever "
               "won a place.</b><br>"
               "Six 768-d branches have lost in max(): spectral, per-generator "
               "specialists, an MLP head,<br>face-tampered, general-noise, "
               "modern-general. This one leaves the shared path at the PIXELS.<br>"
               "Absent &#8594; 0.0, never 0.5: a branch that did not fire must not be "
               "able to raise a max().<br>"
               "<b>Every face is scored, not just the largest</b> &#8212; same "
               "disjunctive argument, one level down. Capped at 8 because max over "
               "N draws rises with N. Measured cost on 4,000 real images: FPR "
               "<b>+0.00%</b>. 7.8.",
     470, 528, 900, 88, parent=S)

for a, b in [("img", "clip"), ("clip", "v"), ("v", "gen"), ("v", "tam"),
             ("gen", "zmax"), ("tam", "zmax"), ("zmax", "sig"), ("sig", "pred"),
             ("facedet", "faceemb"), ("faceemb", "faceprobe"),
             ("faceprobe", "facescores"), ("facescores", "facesig"),
             ("facesig", "pred"),
             ("pred", "verdict"), ("pred", "json")]:
    arrow(a, b)
arrow("img", "facedet", "pixels, not the<br>shared embedding", EDGE + "dashed=1;")
arrow("img", "prov", "the ORIGINAL bytes,<br>before normalise()", EDGE_PROV)
arrow("prov", "prov_out", "", EDGE_PROV)

# --- the divider ---------------------------------------------------------
box("divider", "", 20, 625, 1820, 10,
    "line;html=1;strokeColor=#9673a6;strokeWidth=2;dashed=1;", parent=S)
box("subhead", "REPORTED, NEVER SCORED &#8212; computed on the same request and shown "
               "beside the verdict. Nothing below this line can move <b>pred</b>.",
    20, 640, 1300, 26,
    "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontStyle=1;"
    "fontColor=#9673a6;", font=13, parent=S)

# --- everything the request computes but does not score ------------------
box("d_cal", "<b>Platt-calibrated display scores</b><br>general &#183; tampered "
             "&#183; face<br><i>a DIFFERENT scale from pred</i>",
    20, 680, 250, 90, BOX + PURPLE, parent=S)
box("d_content", "<b>content_type</b><br>CLIP zero-shot over<br>5 prompt vectors",
    295, 680, 210, 90, BOX + PURPLE, parent=S)
box("d_rel", "<b>reliability</b><br>0.6&#183;decisiveness + 0.4&#183;agreement<br>"
             "high | medium | low",
    530, 680, 230, 90, BOX + PURPLE, parent=S)
box("d_regions", "<b>regions[]</b><br>face bounding box<br>drawn over the upload",
    785, 680, 210, 90, BOX + PURPLE, parent=S)
box("d_patch", "<b>patch self-consistency heat map</b> &#8212; built, not yet wired<br>"
               "argmax over a 3&#215;3 grid finds the edited cell <b>73.5%</b> of the "
               "time (chance 11.1%, top-3 86.1%)<br>"
               "Zero new parameters, 9&#215; inference. The explainability we "
               "otherwise do not have.",
    1020, 680, 820, 90, BOX + YELLOW, parent=S)

note("n_prov", "<b>Provenance is evidence rather than inference, and still not in the "
               "score.</b> On test-images/ all 7 GPT-image-2 files carry a signed C2PA "
               "manifest asserting IPTC digitalSourceType = trainedAlgorithmicMedia "
               "&#8212; and the pixel model misses <b>4 of those 7</b>. The 4 "
               "open-weights Janus-Pro files and every real photo carry nothing.<br>"
               "So it finds <i>policy-compliant commercial generators</i>, not AI; one "
               "re-save strips it; and it is null for <b>100% of every eval set we own</b>, "
               "because normalise() destroys exactly what it reads. Unmeasurable, so "
               "unscored. ERROR_ANALYSIS section 7.9.",
     20, 785, 1300, 85, parent=S)
note("n_demo", "<b>Why the line is drawn at all.</b> The displayed per-branch numbers are "
               "Platt-calibrated for readability; <b>pred</b> is not, and its branches "
               "are on the shifted scale. Quoting a signal as if it were the score is the "
               "exact drift that has bitten this repo five times &#8212; one definition, "
               "<code>predict.score_embeddings</code>, and everything else is decoration.",
     20, 880, 1820, 60, parent=S)

arrow("v", "d_content", "the same embedding, reused<br>&#8212; no second CLIP pass",
      EDGE_PROV)

# =========================================================================
# BUILT, MEASURED, NOT WIRED
# =========================================================================
C = lane("cut",
         "BUILT, MEASURED, NOT WIRED &#8212; each carries the number that killed it",
         40, 1175, 1870, 205,
         "fillColor=#fdf8f8;strokeColor=#b85450;strokeWidth=2;"
         "swimlaneFillColor=#fefbfb;")

box("c_spec", "<b>spectral / &#8220;regularity&#8221;</b><br>8 hand-crafted frequency "
              "features<br>clean <b>0.7365</b> &#183; worst <b>0.5596</b> (blur10)<br>"
              "<i>the demo returns null</i>",
    20, 60, 400, 95, BOX + RED, parent=C)
box("c_text", "<b>text</b> &#8212; cut TWICE, on measurement<br>"
              "OCR stats transfer at <b>0.4627</b>, below chance &#183; CLIP on warped "
              "crops works but is worth <b>+0.0022</b><br>"
              "under the 15-variant grid 0.8284 &#8594; <b>0.5229</b>, and the loss is "
              "label-correlated 3.63:1<br><i>the demo returns null</i>",
    445, 60, 690, 95, BOX + RED, parent=C)
box("c_fusion", "<b>fusion &#8212; logistic regression over 14 columns</b><br>"
                "<b>0.8511</b> against max()'s <b>0.8597</b> held out. Reaches parity "
                "only by becoming the general probe:<br>it can match the headline OR "
                "detect tampering, never both.<br>"
                "<i>Still the source of the demo's Platt constants &#8212; its only "
                "remaining job</i>",
    1160, 60, 680, 95, BOX + RED, parent=C)

arrow("c_fusion", "d_cal", "fusion.npz's Platt<br>constants only", EDGE_PROV)

# =========================================================================
# footer
# =========================================================================
box("stats", "<b>The whole trained model</b> &#160;&#8226;&#160; "
             f"{total:,} numbers &#183; {(g_kb + t_kb + f_kb):.1f} KB of .npz "
             "&#183; 3 linear probes &#183; no fine-tuning<br><br>"
             "<b>So-Fake-OOD, generator families never seen in training</b> "
             "&#160;&#8226;&#160; AUROC 0.9265 &#183; ACC 0.8380 &#183; F1 0.8243 "
             "&#183; FPR 8.25% &#183; FNR 24.12%",
    40, 1420, 900, 110, BOX + YELLOW, font=12)
box("grid", "<b>Robustness is the evaluation, not a pipeline step</b><br>"
            "Every eval image is scored under all 15 settings: clean + "
            "jpeg(90/70/50/30) + blur(0.5/1/2) + resize(0.5/0.25) + "
            "noise(.02/.05/.10) + jitter + crop<br>"
            "AUROC <b>0.9265 clean &#8594; 0.8911 worst</b> (noise01). Nothing above "
            "changes; the claim is that the score does not either.",
    980, 1420, 930, 110, BOX + YELLOW, font=12)


def render():
    cells = []
    for i, label, x, y, w, h, style, parent in nodes:
        cells.append(
            f'        <mxCell id="{i}" value={quoteattr(label)} style="{style}" '
            f'vertex="1" parent="{parent}">\n'
            f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry" />\n        </mxCell>')
    for n, (a, b, label, style) in enumerate(edges):
        cells.append(
            f'        <mxCell id="e{n}" value={quoteattr(label)} style="{style}" '
            f'edge="1" parent="1" source="{a}" target="{b}">\n'
            f'          <mxGeometry relative="1" as="geometry" />\n        </mxCell>')
    body = "\n".join(cells)
    return f'''<mxfile host="app.diagrams.net">
  <diagram name="Quorum architecture">
    <mxGraphModel dx="1900" dy="1100" grid="0" gridSize="10" guides="1" \
tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" \
pageWidth="1960" pageHeight="1570" math="0" shadow="0">
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
    print(f"  {total:,} trained numbers, shift {shift:.4f}, tampered scale {alpha}")
