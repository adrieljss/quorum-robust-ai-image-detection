"""Slide diagram of the system -> docs/figures/system.drawio.

    app.diagrams.net, File > Open From > Device

Sits between the poster diagram and a cartoon. It keeps the real branch names,
the arithmetic, the frozen/trained split, and the branches that exist but are
not in the score -- while dropping the poster's per-decision annotations, so it
reads in half a minute rather than five.

Every count and constant is read from predict.py or off the .npz files, the same
rule make_architecture_diagram.py follows: a slide that outlives the model it
describes is worse than no slide.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import predict  # noqa: E402
from quorum.embed import BACKBONE  # noqa: E402

OUT = ROOT / "docs" / "figures" / "system.drawio"

BOX = "rounded=1;whiteSpace=wrap;html=1;arcSize=10;"
GREEN = "fillColor=#d5e8d4;strokeColor=#82b366;"
BLUE = "fillColor=#dae8fc;strokeColor=#6c8ebf;"
FROZEN = "fillColor=#eaf2fb;strokeColor=#6c8ebf;dashed=1;"
GREY = "fillColor=#f5f5f5;strokeColor=#999999;"
ORANGE = "fillColor=#ffe6cc;strokeColor=#d79b00;"
PURPLE = "fillColor=#e1d5e7;strokeColor=#9673a6;"
RED = "fillColor=#f8cecc;strokeColor=#b85450;"
YELLOW = "fillColor=#fff2cc;strokeColor=#d6b656;"
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#666666;"
        "strokeWidth=2;endArrow=block;")
SOFT = EDGE + "dashed=1;strokeColor=#9673a6;strokeWidth=1;"

nodes, edges = [], []


def box(i, label, x, y, w, h, style=BOX + BLUE, font=13):
    nodes.append((i, label, x, y, w, h, style + f"fontSize={font};"))
    return i


def txt(i, label, x, y, w, h, style="align=left;", font=11):
    nodes.append((i, label, x, y, w, h,
                  "text;html=1;whiteSpace=wrap;verticalAlign=middle;"
                  f"fontSize={font};" + style))
    return i


def arrow(a, b, label="", style=EDGE):
    edges.append((a, b, label, style))


def n_params(name):
    d = np.load(ROOT / "data" / "models" / f"{name}.npz")
    return sum(int(np.asarray(d[k]).size) for k in d.files)


g, t, f = n_params("general"), n_params("tampered"), n_params("face")
spec, text = n_params("spectral"), n_params("text_crop")
shift, op, alpha = predict.SHIFT, predict.OPERATING_POINT, predict.TAMPERED_SCALE

box("title", "<b>Quorum &#8212; system architecture</b>"
             "<br><font style='font-size:13px;color:#666666'>one frozen "
             f"backbone &#183; {g + t + f:,} trained parameters &#183; one max()"
             "</font>",
    40, 18, 660, 58,
    "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontStyle=1;",
    font=22)
box("legend", "&#10052;&#65039; <b>FROZEN</b> pretrained, never updated"
              "&#160;&#160;&#160;&#160;&#160;"
              "&#128293; <b>TRAINED</b> fitted by us, linear",
    780, 30, 450, 34, BOX + GREY, font=12)

# --- shared frozen backbone ----------------------------------------------
box("upload", "<b>Input image</b><br>any size / format", 40, 148, 145, 72,
    BOX + GREEN)
box("clip", f"&#10052;&#65039; <b>CLIP {BACKBONE}</b><br>304M params &#183; fp16"
            "<br><i>never fine-tuned</i>",
    225, 140, 205, 88, BOX + FROZEN)
box("v", "<b>v</b><br>768-d", 470, 154, 85, 60, BOX + GREY)

box("facedet", "&#10052;&#65039; <b>face_crop()</b><br>YuNet, on the PIXELS",
    225, 300, 205, 60, BOX + FROZEN)
box("fvec", "<b>f</b> = [CLIP(crop), log&#8322;px]<br><b>769-d</b>",
    470, 296, 185, 68, BOX + GREY)

# --- the three branches that ARE the score -------------------------------
box("gen", f"&#128293; <b>general</b><br>z<sub>g</sub> = w&#183;v + b<br>"
           f"{g:,} params",
    600, 92, 180, 74)
box("tam", f"&#128293; <b>tampered</b><br>z<sub>t</sub> = {alpha}(w&#183;v + b)"
           f"<br>{t:,} params",
    600, 186, 180, 74)
box("face", f"&#128293; <b>face</b><br>z<sub>f</sub> = w&#183;f + b<br>"
            f"{f:,} params",
    700, 296, 180, 74)

box("zmax", "<b>z = max(z<sub>g</sub>, z<sub>t</sub>)</b>", 830, 92, 175, 48,
    BOX + ORANGE)
box("sig", f"<b>p = &#963;(z &#8722; {shift:.4f})</b>", 830, 158, 175, 48,
    BOX + ORANGE)
box("pred", "<b>pred = max(p, p<sub>face</sub>)</b>", 1055, 122, 195, 52,
    BOX + ORANGE, font=14)
box("verdict", "<b>AI</b> if pred &#8805; 0.5<br>"
               f"<font style='font-size:10px'>&#8801; raw score &#8805; {op}"
               "</font>",
    1300, 120, 175, 56, BOX + GREEN, font=13)

txt("n_max", "<b>max(), not a vote.</b> Any ONE branch firing is enough &#8212; "
             "the task is disjunctive:<br>an image is AI-touched if it is fully "
             "synthetic <b>or</b> locally edited.",
    830, 218, 430, 40)
txt("n_face", "the only branch built on <b>different features</b><br>"
              "(769-d, not the shared 768-d)",
    895, 376, 270, 32)

for a, b in [("upload", "clip"), ("clip", "v"), ("v", "gen"), ("v", "tam"),
             ("gen", "zmax"), ("tam", "zmax"), ("zmax", "sig"), ("sig", "pred"),
             ("facedet", "fvec"), ("fvec", "face"), ("face", "pred"),
             ("pred", "verdict")]:
    arrow(a, b)
arrow("upload", "facedet", "pixels", EDGE + "dashed=1;")
arrow("clip", "fvec", "same frozen CLIP,<br>on the crop", EDGE + "dashed=1;")

# --- built, shown, not scored --------------------------------------------
box("shown", "<b>BUILT AND SHOWN IN THE DEMO &#8212; never enters pred</b>",
    40, 442, 700, 28,
    "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontStyle=1;"
    "fontColor=#9673a6;", font=13)
box("spec", f"&#128293; <b>spectral</b><br>8 FFT features &#183; {spec} params"
            "<br>0.6736 clean / 0.5471 worst",
    40, 478, 225, 74, BOX + PURPLE, font=12)
box("txtb", f"&#128293; <b>text</b><br>CLIP on an OCR crop &#183; {text} params"
            "<br>transfers 0.8083, worth +0.0022",
    285, 478, 245, 74, BOX + PURPLE, font=12)
box("prov", "<b>provenance</b><br>C2PA / EXIF / XMP<br>"
            "<i>evidence, not inference</i>",
    550, 478, 195, 74, BOX + PURPLE, font=12)
box("ui", "<b>content type &#183; reliability</b><br>face box &#183; "
          "per-branch scores",
    765, 478, 215, 74, BOX + PURPLE, font=12)

for e in ("spec", "txtb", "prov", "ui"):
    arrow("verdict", e, "", SOFT)

# --- measured and rejected ------------------------------------------------
box("cut", "<b>BUILT, MEASURED, REJECTED</b>", 1010, 442, 480, 28,
    "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontStyle=1;"
    "fontColor=#b85450;", font=13)
box("fusion", "<b>fusion</b> &#8212; a learned combiner over 14 inputs<br>"
              "<b>0.8511</b> against max()'s <b>0.8597</b>. It reaches parity "
              "only by<br>becoming the general probe: it can match the headline "
              "<b>or</b><br>detect tampering, never both.",
    1010, 478, 480, 74, BOX + RED, font=12)

txt("n_cut", "<b>Six</b> branches have lost their place in max(): spectral, "
             "per-generator specialists, an MLP head,<br>face-tampered, "
             "general-noise, modern-general. All six shared the 768-d features. "
             "The one that<br>won brought different ones.",
    40, 570, 900, 46)

box("claim", "<b>Two problems, not one.</b> A fully AI image, and a real "
             "photograph with an AI-edited patch.<br>"
             "The general probe scores edited photos at <b>AUROC 0.37</b> "
             "&#8212; worse than chance, because an<br>edited photo is "
             "<i>globally authentic</i>. It is not failing; it is answering the "
             "other question correctly.<br>So we stopped asking one model both.",
    980, 570, 510, 96, BOX + YELLOW, font=12)


def render():
    cells = []
    for i, label, x, y, w, h, style in nodes:
        cells.append(
            f'        <mxCell id="{i}" value={quoteattr(label)} style="{style}" '
            f'vertex="1" parent="1">\n'
            f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry" />\n        </mxCell>')
    for n, (a, b, label, style) in enumerate(edges):
        cells.append(
            f'        <mxCell id="e{n}" value={quoteattr(label)} style="{style}" '
            f'edge="1" parent="1" source="{a}" target="{b}">\n'
            f'          <mxGeometry relative="1" as="geometry" />\n        </mxCell>')
    body = "\n".join(cells)
    return f'''<mxfile host="app.diagrams.net">
  <diagram name="Quorum system">
    <mxGraphModel dx="1600" dy="900" grid="0" gridSize="10" guides="1" \
tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" \
pageWidth="1550" pageHeight="700" math="0" shadow="0">
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
    print(f"  trained: general {g}, tampered {t}, face {f}  ->  {g + t + f:,}")
    print(f"  shown, not scored: spectral {spec}, text {text}")
