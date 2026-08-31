"""Slide diagram of the SYSTEM as the demo presents it.

    docs/figures/system.drawio  ->  app.diagrams.net, File > Open From > Device

Deliberately NOT make_architecture_diagram.py. That one is the poster: probes,
logit-space max(), the shift, parameter counts. This one is a slide, read in
about ten seconds by someone who does not know what a probe is -- so it says
"three detectors" and "highest score wins" and shows what a user actually gets
back.

The one structural idea it must carry is the split: ONE line produces the
verdict, and everything else the demo shows is beside it, never inside it.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "figures" / "system.drawio"

BOX = "rounded=1;whiteSpace=wrap;html=1;arcSize=10;"
GREEN = "fillColor=#d5e8d4;strokeColor=#82b366;"
BLUE = "fillColor=#dae8fc;strokeColor=#6c8ebf;"
GREY = "fillColor=#f5f5f5;strokeColor=#999999;"
ORANGE = "fillColor=#ffe6cc;strokeColor=#d79b00;"
PURPLE = "fillColor=#e1d5e7;strokeColor=#9673a6;"
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#666666;"
        "strokeWidth=2;endArrow=block;")
SOFT = EDGE + "dashed=1;strokeColor=#9673a6;strokeWidth=1;"

nodes, edges = [], []


def box(i, label, x, y, w, h, style=BOX + BLUE, font=14):
    nodes.append((i, label, x, y, w, h, style + f"fontSize={font};"))
    return i


def txt(i, label, x, y, w, h, style="", font=12):
    nodes.append((i, label, x, y, w, h,
                  "text;html=1;whiteSpace=wrap;align=center;"
                  f"verticalAlign=middle;fontSize={font};" + style))
    return i


def arrow(a, b, label="", style=EDGE):
    edges.append((a, b, label, style))


box("title", "<b>How Quorum decides</b>", 40, 24, 400, 44,
    "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontStyle=1;",
    font=26)

# --- the spine ------------------------------------------------------------
box("upload", "📤<br><b>Upload</b><br>any image", 60, 120, 150, 90, BOX + GREEN)
box("web", "<b>Web app</b><br>one server,<br>models already loaded",
    260, 120, 180, 90, BOX + GREY)
box("reader", "<b>Shared image reader</b><br>a frozen, general-purpose<br>"
              "vision model &#8212; used once,<br>feeding all three detectors",
    490, 112, 250, 106, BOX + GREY)

# --- the three detectors --------------------------------------------------
box("d1", "<b>Detector 1</b><br>the whole image<br>"
          "<i>&#8220;was this generated?&#8221;</i>", 830, 20, 200, 92)
box("d2", "<b>Detector 2</b><br>an edited patch<br>"
          "<i>&#8220;was part of it changed?&#8221;</i>", 830, 130, 200, 92)
box("d3", "<b>Detector 3</b><br>faces<br>"
          "<i>&#8220;is this person real?&#8221;</i>", 830, 240, 200, 92)

box("max", "<b>Highest<br>score wins</b>", 1090, 128, 150, 96, BOX + ORANGE,
    font=15)
box("verdict", "<b>AI or Real</b><br>one number,<br>one answer",
    1300, 122, 170, 108, BOX + GREEN, font=15)

txt("n_max", "Any <b>one</b> detector is enough.<br>They never have to agree "
             "&#8212; that is<br>the point, and the name.",
    1055, 240, 230, 60)

# --- what else the demo shows --------------------------------------------
box("extras", "<b>Also shown, never counted</b>", 490, 340, 750, 32,
    "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontStyle=1;"
    "fontColor=#9673a6;", font=14)

box("e1", "each detector's<br>own score", 490, 380, 170, 74, BOX + PURPLE, font=12)
box("e2", "a box drawn<br>around the face", 680, 380, 170, 74, BOX + PURPLE, font=12)
box("e3", "what the file says<br>about itself<br><i>(C2PA / camera)</i>",
    870, 380, 180, 74, BOX + PURPLE, font=12)
box("e4", "texture and text<br>checks", 1070, 380, 170, 74, BOX + PURPLE, font=12)

txt("n_extra", "These help a person <b>judge</b> the verdict.<br>"
               "None of them can <b>change</b> it.",
    490, 468, 300, 46, "align=left;fontColor=#9673a6;")

for a, b in [("upload", "web"), ("web", "reader"),
             ("reader", "d1"), ("reader", "d2"), ("reader", "d3"),
             ("d1", "max"), ("d2", "max"), ("d3", "max"), ("max", "verdict")]:
    arrow(a, b)
for e in ("e1", "e2", "e3", "e4"):
    arrow("verdict", e, "", SOFT)

# --- the two claims a judge should leave with -----------------------------
box("claim1", "<b>Two problems, not one</b><br>"
              "A fully AI image, and a real photo with an AI-edited patch.<br>"
              "A single detector scored <b>worse than a coin flip</b> on the "
              "second &#8212; an edited photo is <i>mostly real</i>.<br>"
              "So we stopped asking one model both questions.",
    60, 560, 590, 130, BOX + "fillColor=#fff2cc;strokeColor=#d6b656;", font=13)
box("claim2", "<b>Cheap to run, cheap to extend</b><br>"
              "The reader is shared and never retrained; each detector is a "
              "tiny layer on top.<br>"
              "Whole trained system: <b>305 KB</b>. A new detector costs "
              "almost nothing to add.",
    690, 560, 550, 130, BOX + "fillColor=#fff2cc;strokeColor=#d6b656;", font=13)


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
    <mxGraphModel dx="1400" dy="800" grid="0" gridSize="10" guides="1" \
tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" \
pageWidth="1560" pageHeight="740" math="0" shadow="0">
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
