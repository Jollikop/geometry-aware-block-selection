"""Reflow frozen prediction vectors and enlarge their parameter legend.

No numerical inference: removes only PDF text, preserves all painted paths,
and retypesets axes and the five recorded parameter triples below the plot.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import pdfplumber
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ContentStream, RectangleObject, FloatObject
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from normalize_snelson_bands import normalize

REPORT=Path(__file__).resolve().parents[2] / 'Report'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('fontdir', type=Path, help='directory containing DejaVuSans.ttf')
parser.add_argument('--source', type=Path, default=REPORT/'figures/snelson_sgpr_t_rbt_sbt.pdf')
parser.add_argument('--output', type=Path, default=REPORT/'figures/snelson_prediction_readable.pdf')
parser.add_argument('--hard-record', type=Path,
                    help='prediction-only record for a re-fitted hard-block figure')
parser.add_argument('--summary', type=Path,
                    help='summary beside the hard-block figure; original default unchanged')
cli=parser.parse_args()
sys.path.insert(0, str(REPORT.parent / 'Simulations'))
from common.paper_palette import METHOD_COLORS
source=cli.source
digest=hashlib.sha256(source.read_bytes()).hexdigest()
if cli.hard_record:
    hard_record=json.loads(cli.hard_record.read_text())
    assert 'sha256:'+digest == hard_record['hard_figure_sha256']
    summary_path=cli.summary or source.parent/'snelson_summary_sgpr_t_bt_sbt.csv'
else:
    manifest=json.loads((REPORT/'Results/snelson_sgpr_t_rbt_sbt_manifest.json').read_text())
    assert 'sha256:'+digest == manifest['artifacts'][source.name]
    summary_path=cli.summary or REPORT/'Results/snelson_summary_sgpr_t_rbt_sbt.csv'
fontdir=cli.fontdir
pdfmetrics.registerFont(TTFont('PlotSans',str(fontdir/'DejaVuSans.ttf')))
with summary_path.open() as f:
    summary={r['method']:r for r in csv.DictReader(f)}
with pdfplumber.open(source) as pdf:
    page=pdf.pages[0]
    text=page.extract_text()
    true_line=next(t for t in text.splitlines() if t.startswith('True GP:'))
    true_values=[float(v) for v in re.findall(r'=([0-9.]+)',true_line)]
    assert len(true_values)==3
    colors=[c['non_stroking_color'] for c in page.chars
            if c['text']=='S' and abs(c['x0']-466.211454)<.01]
names=['SGPR','T-SGPR','RBT-SGPR','SBT-SGPR','True GP']
summary_name={'RBT-SGPR':'BT-SGPR'} if cli.hard_record else {}
old_colors=[(.55686275,.77254902,.90980392),(.56078431,.83921569,.58039216),
        (.90588235,.54117647,.76470588),(.89019608,.10196078,.10980392),(0,0,0)]
colors=[HexColor(METHOD_COLORS[n]).rgb() for n in names]
values=[[float(summary[summary_name.get(n,n)][k]) for k in ['kernel_variance','lengthscale','noise']]
        for n in names[:-1]]+[true_values]
reader=PdfReader(source); page=reader.pages[0]
content=ContentStream(page.get_contents(),reader)
original_paths=[(args,op) for args,op in content.operations
                if op in [b'm',b'l',b'c',b're',b'S',b'f',b'B']]
filtered=[]; inside=False
for args,op in content.operations:
    if op==b'BT': inside=True
    if not inside:
        if op in (b'RG', b'rg'):
            for old,new in zip(old_colors,colors):
                if len(args)==3 and all(abs(float(a)-b)<1e-7 for a,b in zip(args,old)):
                    args=[FloatObject(v) for v in new]
                    break
        filtered.append((args,op))
    if op==b'ET': inside=False
assert not inside
assert original_paths==[(a,o) for a,o in filtered if o in [b'm',b'l',b'c',b're',b'S',b'f',b'B']]
content.operations=filtered; page.replace_contents(content)
width=float(page.mediabox.width)-270; height=303.
page.cropbox=RectangleObject([270,0,float(page.mediabox.width),float(page.mediabox.height)])
writer=PdfWriter(); out=writer.add_blank_page(width,height)
scale=165/231.28
offset=130-(272.620625-238.676875)*scale
out.merge_transformed_page(page,Transformation((1,0,0,scale,-270,offset)))
overlay=io.BytesIO(); c=canvas.Canvas(overlay,pagesize=(width,height),initialFontName='PlotSans')
c.setFont('PlotSans',12)
left,right=37.671875,width-6.468947
for v in [-2,0,2,4,6,8]:
    x=left+(v+2)/10*(right-left)
    c.drawCentredString(x,113,str(v))
for v in [-3,-2,-1,0,1,2]:
    y=130+(v+3)/5*165
    c.drawRightString(left-6,y-4,str(v))
c.drawCentredString((left+right)/2,96,'x')
c.saveState(); c.translate(10,212.5); c.rotate(90); c.drawCentredString(0,0,'y'); c.restoreState()
# Black text is readable at column width; colored swatches link rows to curves.
xs=[190,282,375]
c.setFont('PlotSans',13)
for x,base,sub in [(190,'σ²','f'),(282,'l','f'),(375,'σ²','')]:
    c.drawCentredString(x,78,base)
    if sub:
        c.setFont('PlotSans',9); c.drawString(x+6,75,sub); c.setFont('PlotSans',13)
for i,(name,color,vals) in enumerate(zip(names,colors,values)):
    y=61-i*14
    c.setStrokeColorRGB(*color); c.setLineWidth(1.5); c.line(left,y+4,left+15,y+4)
    c.setFillColorRGB(0,0,0); c.drawString(left+21,y,name)
    for x,value in zip(xs,vals): c.drawCentredString(x,y,f'{value:.3f}')
c.save(); overlay.seek(0)
out.merge_page(PdfReader(overlay).pages[0])
target=cli.output
with target.open('wb') as f: writer.write(f)
normalize(target, target)
print(f'{target}: unchanged prediction paths; five parameter rows at 13 pt; source {digest}')
