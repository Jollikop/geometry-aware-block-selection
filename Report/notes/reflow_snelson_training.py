"""Reflow the frozen PDF's vector loss paths; no training or numeric replay.

The horizontal plotting coordinate is changed, while the logarithmic vertical
coordinate and all curve vertices are preserved. The output is a layout asset,
not a replacement for the frozen source PDF or its provenance manifest.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pdfplumber
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

REPORT = Path(__file__).resolve().parents[2] / "Report"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('fontdir', type=Path, help='directory containing DejaVuSans.ttf')
parser.add_argument('--source', type=Path, default=REPORT / 'figures/snelson_sgpr_t_rbt_sbt.pdf')
parser.add_argument('--output', type=Path, default=REPORT / 'figures/snelson_training_wide.pdf')
args = parser.parse_args()
sys.path.insert(0, str(REPORT.parent / "Simulations"))
from common.paper_palette import METHOD_COLORS
font_dir = args.fontdir
pdfmetrics.registerFont(TTFont("PlotSans", str(font_dir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("PlotSansBold", str(font_dir / "DejaVuSans-Bold.ttf")))
source = args.source
manifest = json.loads((REPORT / "Results/snelson_sgpr_t_rbt_sbt_manifest.json").read_text())
digest = hashlib.sha256(source.read_bytes()).hexdigest()
assert "sha256:" + digest == manifest["artifacts"][source.name]

with pdfplumber.open(source) as pdf:
    page = pdf.pages[0]
    height = float(page.height)
    # Bounds of the original axes, measured in PDF points (top-origin y).
    left, right = 65.671875, 261.196504
    top, bottom = 7.396875, 238.676875
    curves = [c for c in page.curves if c["x1"] < 270 and len(c["pts"]) > 20
              and c["stroke"] and not c["fill"]]
    assert [len(c["pts"]) for c in curves] == [53, 40, 58, 61]
    reference = [l for l in page.lines if abs(l["x0"]-left) < 1e-5
                 and abs(l["x1"]-right) < 1e-5 and 220 < l["top"] < 235]
    assert len(reference) == 1

# Match the width, height, and axis rectangle of the cropped prediction panel.
width = float(page.width) - 270
height = 205.0
new_top, new_bottom = 7.4, 172.4
new_left, new_right = 37.671875, width - 6.468947
out = args.output
c = canvas.Canvas(str(out), pagesize=(width, height), initialFontName="PlotSans")
c.setTitle("Snelson training bound - layout-only vector reflow")


def xpos(x):
    return new_left + (x-left) * (new_right-new_left)/(right-left)


def ypos(y):
    return height-(new_top+(y-top)*(new_bottom-new_top)/(bottom-top))


# Check the rendered endpoints against the recorded final objectives.
names = ["SGPR", "T-SGPR", "RBT-SGPR", "SBT-SGPR"]
y_one, y_point3 = 50.935085, 230.474202
log_scale = (y_point3-y_one)/(-math.log(.3))
for name, curve in zip(names, curves):
    final = math.exp((y_one-curve["pts"][-1][1])/log_scale)
    assert abs(final-manifest["final_negative_bound_per_datum"][name]) < 1e-7
    c.setStrokeColor(HexColor(METHOD_COLORS[name]))
    c.setLineWidth(curve["linewidth"])
    p = c.beginPath()
    p.moveTo(xpos(curve["pts"][0][0]), ypos(curve["pts"][0][1]))
    for x, y in curve["pts"][1:]:
        p.lineTo(xpos(x), ypos(y))
    c.drawPath(p)

c.setStrokeColorRGB(0, 0, 0)
c.setFillColorRGB(0, 0, 0)
c.setLineWidth(1)
c.setDash(3.7, 1.6)
ref_y = ypos(reference[0]["top"])
c.line(new_left, ref_y, new_right, ref_y)
c.setDash()
c.setFont("PlotSansBold", 11)
c.drawString(new_left+4, ref_y+4, "E-SGPR")
c.setLineWidth(.8)
c.rect(new_left, ypos(bottom), new_right-new_left, new_bottom-new_top)
c.setFont("PlotSans", 12)
for epoch, old_x in zip([0, 20, 40, 60], [74.559358, 133.809246, 193.059134, 252.309021]):
    x = xpos(old_x)
    c.line(x, ypos(bottom), x, ypos(bottom)-3.5)
    c.drawCentredString(x, ypos(bottom)-15, str(epoch))
for value in [.3, .4, .6, 1.0]:
    y = ypos(y_one-log_scale*math.log(value))
    c.line(new_left-3.5, y, new_left, y)
    c.drawRightString(new_left-6, y-3, f"{value:g}")
c.drawCentredString((new_left+new_right)/2, 4, "iteration")
c.saveState()
c.translate(9, (ypos(bottom)+ypos(top))/2)
c.rotate(90)
c.setFont("PlotSans", 11)
c.drawCentredString(0, 0, "negative lower bound / N")
c.restoreState()

# Official method colours; compact legend matching the original panel.
legend_x, legend_y = new_right-112, ypos(top)-4
c.setFillColorRGB(1, 1, 1)
c.setStrokeColorRGB(.8, .8, .8)
c.setLineWidth(1)
c.roundRect(legend_x, legend_y-64, 108, 64, 1.6, fill=1, stroke=1)
c.setFont("PlotSans", 12)
for i, (name, curve) in enumerate(zip(names, curves)):
    y = legend_y-11-i*15
    c.setStrokeColor(HexColor(METHOD_COLORS[name]))
    c.setLineWidth(1.5)
    c.line(legend_x+4, y, legend_x+20, y)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(legend_x+26, y-2.6, name)
c.save()
print(f"Saved {out}; verified four curve endpoints; frozen source SHA256={digest}")
