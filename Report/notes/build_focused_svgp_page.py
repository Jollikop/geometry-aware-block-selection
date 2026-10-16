"""Read-only aggregation for the RBF/Amplitude full-versus-refined paper page."""
import argparse
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
REPORT = PROJECT / 'Report'
sys.path.insert(0, str(PROJECT / 'Simulations'))
RESULTS = PROJECT.parent / 'Correlation_Block_SGPR_Results'
COMPLETE = RESULTS / 'svgp_completed_20260906_001421'
AGG = COMPLETE / 'figures_accuracy_20260906_003844'
tp_source = PROJECT / 'output/launchers/plot_transplant300_final.py'
if not tp_source.is_file():
    tp_source = PROJECT / 'transplant300_tools/plot_transplant300_final.py'
spec = importlib.util.spec_from_file_location('tp300', tp_source)
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)
from analysis.paper_labels import paper_label
from common.plot_style import method_color, SAME_BATCH_COLOR

KERNELS = ['rbf', 'amplitude-rbf']
DATASETS = ['pol', 'superconductivity', 'kin40k', 'protein']
METHODS = list(tp.METHODS)
MS = [64, 128, 256, 512]

def campaign(*, from_summary=False, output=None, table_override=None, wide=False):
    csv_path = REPORT / 'Results/svgp_focused_final_summary.csv'
    if table_override is not None:
        if from_summary:
            raise ValueError('table_override and from_summary are exclusive')
        table = table_override
    elif from_summary:
        provenance = json.loads((REPORT / 'Results/svgp_focused_final_manifest.json').read_text())
        assert tp.digest(csv_path) == provenance['outputs'][csv_path.name]
        table = {}
        with csv_path.open() as stream:
            for row in csv.DictReader(stream):
                key = (row['variant'], row['kernel'], row['dataset'], int(row['M']),
                       row['method'], row['metric'])
                assert key not in table and int(row['n']) == 10
                table[key] = float(row['mean']), float(row['sem'])
        assert len(table) == 512
        assert np.isfinite(list(table.values())).all()
    else:
        provenance = json.loads((AGG / 'ACCURACY_MANIFEST.json').read_text())
        source = AGG / 'aggregate_accuracy.csv'
        assert provenance['validated_accuracy_rows'] == 10240
        assert tp.digest(source) == provenance['outputs'][source.name]
        table = {}
        with source.open() as f:
            for row in csv.DictReader(f):
                if row['kind'] == 'complete' and row['kernel'] in KERNELS and row['dataset'] in DATASETS:
                    assert int(row['n']) == 10
                    key = ('complete', row['kernel'], row['dataset'], int(row['M']), row['method'], row['metric'])
                    table[key] = (float(row['mean']), float(row['sem']))
        print('Checking the 300-epoch prediction bundle and lineage...', flush=True)
        lookup, hashes, inputs = tp.load_and_certify(
            PROJECT / 'output/data/transplant300_final_20260909', COMPLETE / 'training')
        for k, d, m, method, metric in itertools.product(KERNELS, DATASETS, MS, METHODS, ['rmse', 'nll']):
            values = [float(lookup[k,d,m,method,s][metric]) for s in range(10)]
            table['sparse300',k,d,m,method,metric] = (np.mean(values), np.std(values,ddof=1)/np.sqrt(10))
        assert len(table) == 512
        assert np.isfinite(list(table.values())).all()
        rows = [dict(variant=v,kernel=k,dataset=d,M=m,method=method,metric=metric,
                     n=10,mean=float(mean),sem=float(sem))
                for (v,k,d,m,method,metric),(mean,sem) in table.items()]
        with csv_path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,
                         'ytick.labelsize':7,'pdf.fonttype':42})
    fig = plt.figure(figsize=(8.2, 6.0) if wide else (6.75, 7.95))
    # Reserve space for the rotated RMSE/NLL labels and their tick labels;
    # the previous left edge clipped the first letter in the paper PDF.
    outer=fig.add_gridspec(2,1,left=.09 if wide else .108,right=.985,
                          top=.885 if wide else .916,
                          bottom=.14 if wide else .105,hspace=.25 if wide else .24)
    axes={}
    groups=[('rbf','RBF'), ('amplitude-rbf','Amplitude-RBF')]
    for gi,(k,title) in enumerate(groups):
        inner=outer[gi].subgridspec(2,4,hspace=.15 if wide else .14,wspace=.25)
        for row,metric in enumerate(['rmse','nll']):
            for col,d in enumerate(DATASETS):
                ax=fig.add_subplot(inner[row,col]); axes[k,d,metric]=ax
                for variant in ['complete','sparse300']:
                    for method in METHODS:
                        points=[table[variant,k,d,m,method,metric] for m in MS]
                        ax.errorbar(MS,[p[0] for p in points],yerr=[p[1] for p in points],
                            color=tp.method_color(method),marker=tp.method_marker(method),
                            linestyle='-' if variant=='complete' else (0,(4,2.5)),
                            markerfacecolor=tp.method_color(method) if variant=='complete' else 'white',
                            markeredgewidth=.65 if variant=='complete' else .45,
                            lw=1 if variant=='complete' else .65,
                            markersize=3,elinewidth=.5 if variant=='complete' else .35,capsize=1.2)
                ax.set_xlim(40,536); ax.set_xticks(MS)
                ax.get_xticklabels()[0].set_horizontalalignment('right')
                ax.tick_params(length=2,pad=1)
                if row==0: ax.tick_params(labelbottom=False)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.grid(alpha=.18,lw=.4)
                if row==0:
                    ax.set_title({'pol':'Pol','superconductivity':'Superconductivity',
                                  'kin40k':'Kin40k','protein':'Protein'}[d],fontsize=7.5,pad=5)
                if col==0:
                    ax.set_ylabel(metric.upper(),labelpad=2)
                    ax.annotate('',xy=(.17,.10),xytext=(.06,.27),
                                xycoords='axes fraction',
                                arrowprops=dict(arrowstyle='-|>',color='.3',lw=.8))
                    ax.text(.21,.12,'lower is better',transform=ax.transAxes,
                            fontsize=6.0,va='center',
                            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.3))
                for spine in ax.spines.values(): spine.set_linewidth(.5)
        box=outer[gi].get_position(fig)
        fig.text(box.x0,box.y1+(.033 if wide else .047),title,fontsize=10,fontweight='bold')
    handles=[Line2D([],[],color=tp.method_color(m),marker=tp.method_marker(m),lw=1,
                    markersize=3,label=paper_label(m)) for m in METHODS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.008 if wide else .006),ncol=2,
               frameon=False,fontsize=8,handlelength=1.6,columnspacing=2)
    styles=[Line2D([],[],color='.2',lw=1.2,ls='-',marker='o',markersize=3,
                   label='BLOCK DIAG. PREDICTOR'),
            Line2D([],[],color='.2',lw=.65,ls=(0,(4,2.5)),marker='o',markersize=3,
                   markerfacecolor='white',label='Sparse SVGP after 300-epoch refinement')]
    fig.legend(handles=styles,loc='upper center',bbox_to_anchor=(.54,1.001),ncol=2,
               frameon=False,fontsize=7.6,handlelength=2.6,columnspacing=1.4)
    fig.text(.54,.105 if wide else .066,r'Inducing points $M$ (linear scale)',ha='center',fontsize=8)
    output=output or REPORT/'figures/svgp_rbf_amplitude_final_page.pdf'
    fig.savefig(output); plt.close(fig)
    if from_summary or table_override is not None:
        print(output, flush=True)
        return
    (REPORT/'Results/svgp_focused_final_manifest.json').write_text(json.dumps(dict(
        scope='RBF and Amplitude; four main datasets; predictive accuracy only',
        seeds_per_cell=10,full_donor_runs=1280,sparse300_runs=1280,
        complete_aggregate_sha256=tp.digest(source),
        complete_accuracy_manifest_sha256=tp.digest(AGG/'ACCURACY_MANIFEST.json'),
        transplant_validation_inputs=inputs,transplant_predictions_checked=len(hashes),
        outputs={output.name:tp.digest(output),csv_path.name:tp.digest(csv_path)},
        note='Existing numerical outputs only. No new training, convergence claim, or timing comparison.'),indent=2))
    print(output,flush=True)


def diagnostic():
    with (REPORT/'Results/epochs_rbf_pol_M128_summary.csv').open() as f: rows=list(csv.DictReader(f))
    curves=['SVGP','T-SVGP','BT-SVGP','SBT-SVGP','SBT-SVGP-same-batch']
    names=['SVGP','T-SVGP','(R)BT-SVGP',
           '(S)BT-SVGP','(S)BT-SVGP: same batch']
    colors=[method_color(c) for c in curves[:-1]]+[SAME_BATCH_COLOR]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
    # The epochwise NLL was evaluated with the historical predictor. The
    # ablation did not save per-epoch parameters, so it cannot be re-evaluated
    # under the paper's block-diagonal predictor. RMSE depends only on the mean and
    # remains comparable; give it the same single-panel format as Figure 5.
    for filename,metrics,size in [('epochs_rbf_pol_M128_objective.pdf',['objective'],(3.25,2.05)),
                                  ('epochs_rbf_pol_M128_test.pdf',['rmse'],(3.25,2.05))]:
        # Snelson uses 12 pt in a 435.665451 pt source, scaled to a 234 pt column.
        font_size=12*234/435.665451 if len(metrics)==1 else 8
        label_size=11*234/435.665451 if len(metrics)==1 else 8
        fig,axs=plt.subplots(1,len(metrics),figsize=size,squeeze=False)
        for ax,metric in zip(axs[0],metrics):
            for curve,name,color in zip(curves,names,colors):
                data=sorted([r for r in rows if r['curve']==curve and r['metric']==metric],key=lambda r:int(r['epoch']))
                assert len(data)==18 and all(int(r['n_seeds'])==10 for r in data)
                x=np.array([int(r['epoch']) for r in data]); y=np.array([float(r['mean']) for r in data])
                se=np.array([float(r['sem']) for r in data])
                if metric=='objective': y=-y
                ax.plot(x,y,color=color,lw=1.15,ls='--' if curve==curves[-1] else '-',label=name)
                ax.fill_between(x,y-se,y+se,color=color,alpha=.13,lw=0)
            ax.set_xlabel('Epoch',fontsize=font_size); ax.set_xlim(0,350); ax.set_xticks([0,100,200,300])
            ax.set_ylabel({'objective':'Negative stochastic\nobjective / N','rmse':'Test RMSE','nll':'Test NLL'}[metric],fontsize=label_size)
            ax.set_ylim({'objective':(-.43,.25),'rmse':(.125,.23),'nll':(-.53,.18)}[metric])
            ax.grid(alpha=.2,lw=.4); ax.tick_params(labelsize=font_size,length=2,pad=2)
            ax.text(.97,.90,r'$\downarrow$ lower is better',ha='right',transform=ax.transAxes,fontsize=label_size)
        handles,labels=axs[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=2 if len(metrics)==1 else 3,
                   fontsize=font_size if len(metrics)==1 else 6.2,frameon=False,handlelength=1.7)
        fig.subplots_adjust(left=.16 if len(metrics)==1 else .09,right=.98,top=.98,bottom=.40 if len(metrics)==1 else .30,wspace=.3)
        fig.savefig(REPORT/'figures'/filename); plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-summary', action='store_true',
                        help='render only the focused page from its certified summary')
    parser.add_argument('--output', type=Path, help='PDF destination')
    parser.add_argument('--diagnostic-only', action='store_true',
                        help='regenerate only the epochwise ablation figures')
    args = parser.parse_args()
    if not args.from_summary:
        diagnostic()
    if not args.diagnostic_only:
        campaign(from_summary=args.from_summary, output=args.output)
