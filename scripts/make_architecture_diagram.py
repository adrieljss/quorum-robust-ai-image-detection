"""Poster diagram of the whole system -> docs/figures/architecture.drawio.

Open it at app.diagrams.net (File > Open From > Device) and it lays out as saved.

Three containers, and the containers ARE the argument:

  SHIPPED           the only thing that produces `pred`. Three branches, one max().
  DEMO ONLY         computed per request, displayed beside the verdict, never fed
                    back into the score.
  BUILT, NOT WIRED  branches that exist, were measured, and lost. Each carries the
                    number that killed it, because "we tried it" is worth nothing
                    without "and here is what it scored".

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
EDGE_DASH = EDGE + "dashed=1;strokeColor=#9673a6;fontColor=#9673a6;"
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
# 1. SHIPPED
# =========================================================================
S = lane("shipped",
         "SHIPPED &#8212; this, and nothing else, produces <b>pred</b>",
         40, 125, 1870, 690,
         "fillColor=#f7fbf7;strokeColor=#82b366;strokeWidth=2;"
         "swimlaneFillColor=#fbfefb;")

box("img", "<b>Input image</b><br>any size<br>jpg / png / webp / bmp",
    20, 195, 180, 80, BOX + GREEN, parent=S)
box("clip", f"<b>CLIP {BACKBONE}</b><br>{PRETRAINED} weights, <b>FROZEN</b>"
            "<br>304M params, fp16<br><i>never fine-tuned</i>",
    250, 170, 230, 100, BOX + GREY, parent=S)
box("v", "<b>v</b> &#8212; 768-d<br>L2-normalised", 550, 195, 150, 70,
    BOX + SOLID_GREY, parent=S)

box("gen", f"<b>general probe</b><br>z<sub>g</sub> = w&#183;v + b<br>"
           f"{g_n:,} numbers &#183; {g_kb:.1f} KB",
    780, 85, 210, 85, parent=S)
box("tam", f"<b>tampered probe</b><br>z<sub>t</sub> = {alpha} &#215; (w&#183;v + b)<br>"
           f"{t_n:,} numbers &#183; {t_kb:.1f} KB",
    780, 220, 210, 85, parent=S)
box("zmax", "<b>z = max(z<sub>g</sub>, z<sub>t</sub>)</b><br>max in <i>logit</i> space",
    1060, 140, 200, 80, BOX + ORANGE, parent=S)
box("sig", f"<b>p = &#963;(z &#8722; {shift:.4f})</b><br>puts the operating<br>"
           f"point {op} at 0.5",
    1340, 135, 210, 90, BOX + ORANGE, parent=S)

box("facedet", "<b>face_crop()</b><br>detect + crop<br>the largest face",
    250, 425, 230, 85, BOX + GREY, parent=S)
box("faceemb", "<b>the same frozen CLIP</b>, on the crop<br>"
               "&#8853; standardised log&#8322;(face_px)",
    550, 430, 240, 75, BOX + SOLID_GREY, parent=S)
box("faceprobe", f"<b>face probe</b> &#183; <b>769-d</b><br>z<sub>f</sub> = w&#183;f + b"
                 f"<br>{f_n:,} numbers &#183; {f_kb:.1f} KB",
    840, 425, 220, 85, parent=S)
box("facesig", f"<b>p<sub>face</sub> = &#963;(z<sub>f</sub> &#8722; {shift:.4f})</b><br>"
               "<b>0.0</b> if no face found",
    1120, 430, 200, 75, BOX + ORANGE, parent=S)

box("pred", "<b>pred = max(p, p<sub>face</sub>)</b>", 1340, 295, 210, 60,
    BOX + ORANGE, font=14, parent=S)
box("verdict", '<b>verdict</b><br>"AI" if pred &#8805; 0.5<br>else "real"',
    1620, 135, 200, 90, BOX + GREEN, parent=S)
box("json", "<b>predictions.json</b><br>[{image_path, pred}]",
    1620, 295, 200, 70, BOX + GREEN, parent=S)

note("n_clip", "<b>Frozen on purpose.</b> Detection costs 769 trained numbers<br>"
               "per branch on a backbone you already run. Stated with its<br>"
               "limitation: a 2023 backbone has never seen a 2026<br>"
               "generator's artifacts &#8212; and that is our largest failure.",
     250, 280, 300, 80, parent=S)
note("n_gen", "trained on SID_Set + WildFake-Midjourney + 4 of 5 calib_ood<br>"
              "generator families; calibration rotated over the 5th and the<br>"
              "five weight vectors averaged",
     770, 25, 340, 55, parent=S)
note("n_tam", f"<b>{alpha} is a policy dial, not a fitted parameter.</b> max() is not"
               "<br>invariant to rescaling one arm, so this alone sets false<br>"
               "positives against edit recall (8.5% / 74.6%). Trained on<br>"
               "tampered-vs-real only &#8212; never sees a synthetic image.",
     770, 310, 345, 80, parent=S)
note("n_max", "<b>max(), not a learned combiner.</b> The task is<br>"
              "disjunctive: either arm firing means AI touched it.<br>"
              "Fusion LR scored lower on both eval sets.",
     1050, 230, 290, 60, parent=S)
note("n_face", "<b>The only branch with DIFFERENT features, and the only one that ever "
               "won a place.</b><br>"
               "Six 768-d branches have lost in max(): spectral, per-generator "
               "specialists, an MLP head,<br>face-tampered, general-noise, "
               "modern-general. This one leaves the shared path at the PIXELS.<br>"
               "Absent &#8594; 0.0, never 0.5: a branch that did not fire must not be "
               "able to raise a max().",
     560, 520, 780, 75, parent=S)

for a, b in [("img", "clip"), ("clip", "v"), ("v", "gen"), ("v", "tam"),
             ("gen", "zmax"), ("tam", "zmax"), ("zmax", "sig"), ("sig", "pred"),
             ("facedet", "faceemb"), ("faceemb", "faceprobe"),
             ("faceprobe", "facesig"), ("facesig", "pred"),
             ("pred", "verdict"), ("pred", "json")]:
    arrow(a, b)
arrow("img", "facedet", "pixels, not the<br>shared embedding", EDGE + "dashed=1;")

# =========================================================================
# 2. DEMO ONLY
# =========================================================================
D = lane("demo",
         "DEMO ONLY &#8212; computed per request, shown beside the verdict, "
         "<b>never fed back into pred</b>",
         40, 845, 1170, 450,
         "fillColor=#f7f9fd;strokeColor=#6c8ebf;strokeWidth=2;"
         "swimlaneFillColor=#fbfcfe;")

box("d_cal", "<b>Platt-calibrated display scores</b><br>general &#183; tampered "
             "&#183; face<br><i>a DIFFERENT scale from pred</i>",
    20, 65, 250, 90, BOX + PURPLE, parent=D)
box("d_content", "<b>content_type</b><br>CLIP zero-shot over<br>5 prompt vectors",
    295, 65, 210, 90, BOX + PURPLE, parent=D)
box("d_rel", "<b>reliability</b><br>0.6&#183;decisiveness + 0.4&#183;agreement<br>"
             "high | medium | low",
    530, 65, 230, 90, BOX + PURPLE, parent=D)
box("d_regions", "<b>regions[]</b><br>face bounding box<br>drawn over the upload",
    785, 65, 210, 90, BOX + PURPLE, parent=D)
box("d_prov", "<b>provenance</b> &#8212; C2PA &#183; EXIF &#183; XMP &#183; PNG text<br>"
              "read from the ORIGINAL bytes, before normalise()<br>"
              "<b>evidence, not inference</b> &#8212; never enters pred",
    20, 180, 250, 90, BOX + PURPLE, parent=D)
box("d_patch", "<b>patch self-consistency heat map</b> &#8212; built, not yet wired<br>"
               "argmax over a 3&#215;3 grid finds the edited cell <b>73.5%</b> of the "
               "time (chance 11.1%, top-3 86.1%)<br>"
               "Zero new parameters, 9&#215; inference. It is the explainability we "
               "otherwise do not have.",
    295, 180, 700, 60, BOX + YELLOW, parent=D)
note("n_prov", "<b>On our own test set the asymmetry is stark.</b> All 7 GPT-image-2 "
               "files carry a signed C2PA manifest asserting IPTC<br>"
               "digitalSourceType = trainedAlgorithmicMedia &#8212; and the pixel model "
               "misses <b>4 of those 7</b>. The 4 open-weights Janus-Pro<br>"
               "files and every real photo carry nothing. So it detects "
               "<i>policy-compliant commercial generators</i>, not AI, and one<br>"
               "re-save strips it. A best case, not a general one &#8212; which is "
               "exactly why it is reported and not scored.",
     295, 245, 830, 60, parent=D)
note("n_demo", "<b>Why the separation matters.</b> The displayed per-branch numbers are "
               "Platt-calibrated for readability; <b>pred</b> is not. Quoting a signal "
               "as if it were the score is the exact drift that has bitten this repo "
               "five times.<br>One definition &#8212; <code>predict.score_embeddings"
               "</code> &#8212; and everything else is decoration.",
     20, 320, 1120, 70, parent=D)

# =========================================================================
# 3. BUILT, MEASURED, NOT WIRED
# =========================================================================
C = lane("cut",
         "BUILT, MEASURED, NOT WIRED &#8212; each carries the number that killed it",
         1225, 845, 685, 425,
         "fillColor=#fdf8f8;strokeColor=#b85450;strokeWidth=2;"
         "swimlaneFillColor=#fefbfb;")

box("c_spec", "<b>spectral / &#8220;regularity&#8221;</b><br>8 hand-crafted frequency "
              "features<br>clean <b>0.7365</b> &#183; worst <b>0.5596</b> (blur10)<br>"
              "<i>demo returns null</i>",
    20, 65, 310, 100, BOX + RED, parent=C)
box("c_text", "<b>text</b> &#8212; cut TWICE, on measurement<br>"
              "OCR stats: transfer <b>0.4627</b>, below chance<br>"
              "CLIP on warped crops: works, worth <b>+0.0022</b><br>"
              "under the grid: 0.8284 &#8594; <b>0.5229</b>, and the loss<br>"
              "is label-correlated 3.63:1<br><i>demo returns null</i>",
    350, 65, 315, 130, BOX + RED, parent=C)
box("c_fusion", "<b>fusion &#8212; logistic regression, 14 columns</b><br>"
                "<b>0.8511</b> against max()'s <b>0.8597</b> held out<br>"
                "Reaches parity only by becoming the general probe:<br>"
                "it can match the headline OR detect tampering,<br>never both",
    20, 180, 310, 110, BOX + RED, parent=C)
note("n_cut", "All still in the repo, and in the fusion vector at neutral fill, so any "
              "one of them is a line away from being wired back. None is worth the "
              "line today.",
     20, 305, 645, 45, parent=C)

arrow("c_fusion", "d_cal", "fusion.npz's Platt<br>constants only", EDGE_DASH)
arrow("v", "d_content", "the same embedding, reused<br>&#8212; no second CLIP pass",
      EDGE_DASH)

# =========================================================================
# footer
# =========================================================================
box("stats", "<b>The whole trained model</b> &#160;&#8226;&#160; "
             f"{total:,} numbers &#183; {(g_kb + t_kb + f_kb):.1f} KB of .npz "
             "&#183; 3 linear probes &#183; no fine-tuning<br><br>"
             "<b>So-Fake-OOD, generator families never seen in training</b> "
             "&#160;&#8226;&#160; AUROC 0.9265 &#183; ACC 0.8380 &#183; F1 0.8243 "
             "&#183; FPR 8.25% &#183; FNR 24.12%",
    40, 1295, 900, 110, BOX + YELLOW, font=12)
box("grid", "<b>Robustness is the evaluation, not a pipeline step</b><br>"
            "Every eval image is scored under all 15 settings: clean + "
            "jpeg(90/70/50/30) + blur(0.5/1/2) + resize(0.5/0.25) + "
            "noise(.02/.05/.10) + jitter + crop<br>"
            "AUROC <b>0.9265 clean &#8594; 0.8911 worst</b> (noise01). Nothing above "
            "changes; the claim is that the score does not either.",
    980, 1295, 930, 110, BOX + YELLOW, font=12)


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
pageWidth="1960" pageHeight="1450" math="0" shadow="0">
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
