"""Replot the complete frozen SVGP campaign without changing any run files."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

PROJECT = Path(__file__).resolve().parents[2]
REPORT = PROJECT / 'Report'
SOURCE = PROJECT / 'output/pdf/svgp_official_final_20260909'
ANCHOR_SOURCE = PROJECT / 'output/results/gibbs_anchor_20260926'
ANCHOR_TRANSPLANT_SOURCE = PROJECT / 'output/results/gibbs_anchor_transplant_20260930'
sys.path.insert(0, str(PROJECT / 'Simulations'))
from common.plot_style import method_color, method_marker
from analysis.paper_labels import paper_label

KERNELS = ['rbf', 'matern52', 'amplitude-rbf', 'gibbs']
DATASETS = ['pol', 'superconductivity', 'kin40k', 'protein',
            'elevators', 'traffic', 'electricity', '3droad']
METHODS = ['SVGP', 'T-SVGP', 'RBT-SVGP', 'SBT-SVGP']
MS = [64, 128, 256, 512]
NAMES = {'rbf': 'RBF', 'matern52': 'Matérn-5/2',
         'amplitude-rbf': 'Amplitude-RBF', 'gibbs': 'Gibbs',
         'pol': 'Pol', 'superconductivity': 'Superconductivity',
         'kin40k': 'Kin40k', 'protein': 'Protein', 'elevators': 'Elevators',
         'traffic': 'Traffic', 'electricity': 'Electricity', '3droad': '3Droad'}


def digest(path):
    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()


def apply_gibbs_anchor(table, provenance):
    """Replace only Gibbs SBT complete-predictor cells with certified anchor runs.

    The negative objective comes from the matching new training records. The
    the paired 300-epoch sparse SBT cells use the same inducing-anchor lineage.
    """
    pred_root = ANCHOR_SOURCE / 'prediction'
    certificate = pred_root / 'GIBBS_ANCHOR_PREDICTION_CERTIFICATE.json'
    meta = json.loads(certificate.read_text())
    assert meta['release_certified'] and meta['complete_grid'] and meta['training_unchanged']
    assert meta['training_params'] == meta['predictions'] == 320 and meta['predictor'] == 'block'
    plan_file = pred_root / 'prediction_plan.json'
    assert digest(plan_file) == meta['prediction_plan_sha256']
    aggregate = pred_root / 'gibbs_anchor_aggregate.csv'
    assert digest(aggregate) == meta['outputs'][aggregate.name]
    groups = defaultdict(list)
    checked_training = {}
    seen = set()
    assert len(meta['prediction_files']) == 320
    for relative, expected_hash in meta['prediction_files'].items():
        prediction_file = pred_root / relative
        assert digest(prediction_file) == expected_hash, prediction_file
        row = json.loads(prediction_file.read_text())
        cfg = row['training_config']
        key = cfg['dataset'], int(cfg['M']), int(cfg['seed'])
        assert key not in seen
        seen.add(key)
        assert cfg['kernel'] == 'gibbs' and cfg['method'] == 'SBT-SVGP'
        assert cfg['sbt_geometry'] == 'anchor' and cfg['init_from_kernel'] == 'rbf'
        assert cfg['epochs'] == 300 and row['identity']['predictor'] == 'block'
        training_relative = str(Path(relative).with_name('training.json'))
        training_file = (ANCHOR_SOURCE / 'training' / cfg['dataset'] / f"M{cfg['M']}"
                         / 'SBT-SVGP' / f"seed{cfg['seed']}" / 'training.json')
        assert digest(training_file) == meta['training_inputs'][training_relative], training_file
        training = json.loads(training_file.read_text())
        assert training['config'] == cfg and training['training_complete']
        assert training['config_id'] == row['identity']['config_id']
        checked_training[str(training_file.relative_to(ANCHOR_SOURCE))] = digest(training_file)
        values = dict(rmse=row['rmse'], nll=row['nll'], negative_objective=
                      -training['last_epoch_stochastic_objective_estimate_per_datum'])
        for metric, value in values.items():
            assert math.isfinite(value)
            groups[cfg['dataset'], int(cfg['M']), metric].append(value)
    assert seen == set(itertools.product(DATASETS, MS, range(10)))
    replacement = {}
    for (dataset, m, metric), values in groups.items():
        assert len(values) == 10
        replacement['complete', 'gibbs', dataset, m, 'SBT-SVGP', metric] = (
            statistics.mean(values), statistics.stdev(values) / math.sqrt(10))
    aggregate_seen = set()
    with aggregate.open() as stream:
        for row in csv.DictReader(stream):
            assert row['kernel'] == 'gibbs' and row['method'] == 'SBT-SVGP' and int(row['n']) == 10
            key = 'complete', 'gibbs', row['dataset'], int(row['M']), 'SBT-SVGP', row['metric']
            assert key not in aggregate_seen
            aggregate_seen.add(key)
            mean, sem = replacement[key]
            assert math.isclose(mean, float(row['mean']), rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(sem, float(row['sem']), rel_tol=1e-12, abs_tol=1e-12)
    assert len(aggregate_seen) == 64 and len(replacement) == 96
    table.update(replacement)
    provenance['gibbs_sbt_anchor'] = dict(certificate=str(certificate), certificate_sha256=digest(certificate),
        aggregate=str(aggregate), aggregate_sha256=digest(aggregate), training_record_hashes=checked_training,
        prediction_files_verified=320, training_records_verified=320, replaced_cells=96,
        scope='Gibbs SBT-SVGP complete RMSE/NLL and negative final objective; RBF warm start + 300 epochs',
        sparse300='Replaced by the matched 300-epoch anchor transplant aggregate')


def apply_gibbs_anchor_transplant(table, provenance):
    """Replace historical Gibbs SBT sparse cells with paired anchor transplants."""
    aggregate = ANCHOR_TRANSPLANT_SOURCE / 'gibbs_anchor_transplant300_aggregate.csv'
    if not aggregate.is_file():
        raise FileNotFoundError(aggregate)
    rows = list(csv.DictReader(aggregate.open()))
    assert len(rows) == 320
    grouped = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        assert row['kernel'] == 'gibbs'
        assert row['donor_method'] == 'SBT-SVGP'
        assert int(row['transplant_epoch']) == 300
        key = row['dataset'], int(row['M']), int(row['seed'])
        assert key not in seen
        seen.add(key)
        grouped[row['dataset'], int(row['M'])]['rmse'].append(float(row['rmse']))
        grouped[row['dataset'], int(row['M'])]['nll'].append(float(row['nll']))
    assert seen == set(itertools.product(DATASETS, MS, range(10)))
    replaced = 0
    for (dataset, m), metrics in grouped.items():
        for metric, values in metrics.items():
            table['sparse300', 'gibbs', dataset, m, 'SBT-SVGP', metric] = (
                statistics.mean(values), statistics.stdev(values) / math.sqrt(10))
            replaced += 1
    provenance['gibbs_sbt_anchor_transplant'] = dict(
        aggregate=str(aggregate), aggregate_sha256=digest(aggregate),
        prediction_rows=len(rows), replaced_cells=replaced,
        scope='Gibbs SBT-SVGP inducing-anchor donor, SVGP transplant at 300 epochs',
        donor_root='gibbs_anchor_warm300_20260917_115548',
        certificate_sha256='sha256:271f5c760eaff786c45089f6980cbd8c8abd362403547d59b29a409930481bce',
    )


def load():
    table, provenance = {}, {}
    specs = [('complete', 'donor', 'SVGP_FULL_MANIFEST.json', 'svgp_full_aggregate.csv',
              ['rmse', 'nll', 'negative_objective'], 'method'),
             ('sparse300', 'transplant300', 'SVGP_TRANSPLANT300_CERTIFICATE.json',
              'svgp_transplant300_aggregate.csv', ['rmse', 'nll'], 'donor')]
    for variant, folder, manifest, filename, metrics, method_key in specs:
        source = SOURCE / folder / filename
        certificate = SOURCE / folder / manifest
        meta = json.loads(certificate.read_text())
        assert digest(source) == meta['outputs'][filename], source
        if variant == 'complete':
            assert meta['full_paper_freeze_certified'] and meta['complete_runs'] == 5120
        else:
            assert meta['release_certified'] and meta['complete_grid']
            assert meta['predictions'] == meta['training_params'] == 5120
        with source.open() as stream:
            rows = list(csv.DictReader(stream))
        actual = set()
        for row in rows:
            key = (row['kernel'], row['dataset'], int(row['M']), row[method_key], row['metric'])
            assert key not in actual
            actual.add(key)
            assert int(row['n']) == 10
            value, sem = float(row['mean']), float(row['sem'])
            assert math.isfinite(value) and math.isfinite(sem) and sem >= 0
            table[(variant,) + key] = (value, sem)
        assert actual == set(itertools.product(KERNELS, DATASETS, MS, METHODS, metrics))
        provenance[variant] = dict(aggregate=str(source), aggregate_sha256=digest(source),
                                   certificate=str(certificate), certificate_sha256=digest(certificate),
                                   runs=5120, aggregate_cells=len(rows))
    # The expanded appendix must reproduce every matching main-page value.
    with (REPORT / 'Results/svgp_focused_final_summary.csv').open() as stream:
        for row in csv.DictReader(stream):
            key = (row['variant'], row['kernel'], row['dataset'], int(row['M']),
                   row['method'], row['metric'])
            mean, sem = table[key]
            assert math.isclose(mean, float(row['mean']), rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(sem, float(row['sem']), rel_tol=1e-12, abs_tol=1e-12)
    apply_gibbs_anchor(table, provenance)
    apply_gibbs_anchor_transplant(table, provenance)
    return table, provenance


def draw(table, kernel, datasets, path):
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(9.6, 6.15))
    grid = fig.add_gridspec(4, 5, left=.092, right=.992,
                          top=.858, bottom=.125, hspace=.33, wspace=.38)
    axes = {}
    columns=[('complete','rmse'),('complete','nll'),('complete','negative_objective'),
             ('sparse300','rmse'),('sparse300','nll')]
    for row, dataset in enumerate(datasets):
        for col,(variant,metric) in enumerate(columns):
                ax = fig.add_subplot(grid[row, col]); axes[variant, dataset, metric] = ax
                for method in METHODS:
                    values = [table[variant, kernel, dataset, m, method, metric] for m in MS]
                    ax.errorbar(MS, [v[0] for v in values], yerr=[v[1] for v in values],
                                color=method_color(method), marker=method_marker(method),
                                lw=1, markersize=3, elinewidth=.55, capsize=1.3,
                                linestyle='-' if variant=='complete' else (0,(4,2)),
                                markerfacecolor=method_color(method) if variant=='complete' else 'white')
                ax.set_xlim(40, 536); ax.set_xticks(MS)
                ax.get_xticklabels()[0].set_horizontalalignment('right')
                ax.tick_params(length=2, pad=1)
                if row < len(datasets)-1: ax.tick_params(labelbottom=False)
                ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.grid(alpha=.18, lw=.4)
                if row == 0:
                    ax.set_title({'rmse': 'RMSE $\\downarrow$', 'nll': 'NLL $\\downarrow$',
                                  'negative_objective': 'Negative objective / N $\\downarrow$'}[metric],
                                 fontsize=8,pad=6)
                for spine in ax.spines.values(): spine.set_linewidth(.5)
    for dataset, metric in itertools.product(datasets, ['rmse', 'nll']):
        lohi = [axes[v, dataset, metric].get_ylim() for v in ['complete', 'sparse300']]
        limits = min(v[0] for v in lohi), max(v[1] for v in lohi)
        for variant in ['complete', 'sparse300']:
            axes[variant, dataset, metric].set_ylim(limits)
    for dataset in datasets:
        pos = axes['complete', dataset, 'rmse'].get_position()
        fig.text(.018, (pos.y0+pos.y1)/2, NAMES[dataset],
                 fontsize=8, ha='center',va='center',rotation=90,fontweight='bold')
    boundary=axes['complete',datasets[0],'negative_objective'].get_position().x1+.012
    fig.add_artist(Line2D([boundary,boundary],[.11,.928],transform=fig.transFigure,color='.65',lw=.8))
    fig.text(.34,.933,f'{NAMES[kernel]} · BLOCK DIAG. PREDICTOR',
             ha='center',fontsize=10,fontweight='bold')
    fig.text(.815,.933,'SPARSE SVGP · 300-EPOCH REFINEMENT',ha='center',fontsize=9,fontweight='bold')
    if kernel == 'gibbs':
        fig.text(.34,.899,'(S)BT: inducing-anchor donor',ha='center',fontsize=8,color='.3')
        fig.text(.815,.899,'(S)BT: matched 300-epoch anchor transplant',ha='center',fontsize=8,color='.3')
    handles = [Line2D([], [], color=method_color(m), marker=method_marker(m),
                      lw=1, markersize=3, label=paper_label(m)) for m in METHODS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .002),
               ncol=4, fontsize=8, frameon=False, columnspacing=1.5, handlelength=1.6)
    fig.text(.53, .046, r'Inducing points $M$     $\downarrow$ lower is better',
             ha='center', fontsize=8)
    fig.savefig(path); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernels', nargs='+', choices=KERNELS, default=KERNELS,
                        help='redraw only these kernels; retain all pages and their source checks')
    parser.add_argument('--from-summary', action='store_true',
                        help='render the unchanged plots from the shipped certified summary')
    parser.add_argument('--output-dir', type=Path,
                        help='figure destination (default: Report/figures/svgp_frozen_campaign)')
    args = parser.parse_args()
    if args.from_summary:
        summary = REPORT / 'Results/svgp_frozen_campaign_summary.csv'
        manifest = json.loads((REPORT / 'Results/svgp_frozen_campaign_manifest.json').read_text())
        assert digest(summary) == manifest['outputs']['Results/svgp_frozen_campaign_summary.csv']
        table = {}
        with summary.open() as stream:
            for row in csv.DictReader(stream):
                key = (row['variant'], row['kernel'], row['dataset'], int(row['M']),
                       row['method'], row['metric'])
                assert key not in table and int(row['n']) == 10
                table[key] = float(row['mean']), float(row['sem'])
        assert len(table) == 2560
        target = args.output_dir or REPORT / 'figures/svgp_frozen_campaign'
        target.mkdir(parents=True, exist_ok=True)
        for kernel in args.kernels:
            for group in range(2):
                draw(table, kernel, DATASETS[group*4:(group+1)*4],
                     target / f'{kernel}_{group+1}.pdf')
        print(f'Rendered {2 * len(args.kernels)} pages from certified summary; no run data changed.')
        return
    table, provenance = load()
    target = args.output_dir or REPORT / 'figures/svgp_frozen_campaign'
    target.mkdir(parents=True, exist_ok=True)
    files = []
    tex = [r'\section{Complete Frozen SVGP Campaign}', r'\label{app:svgp-frozen}',
           r'This appendix reports all 5120 donor runs and 5120 matched sparse SVGP '
           r'transplants after 300-epoch refinement from the frozen campaign. '
           r'The grid comprises four kernels (RBF, Mat\'ern-5/2, Amplitude-RBF, Gibbs), '
           r'eight datasets, four inducing budgets \(M\in\{64,128,256,512\}\), '
           r'four methods, and ten seeds per configuration. Each kernel is shown '
           r'on two pages, covering the four main datasets and the four additional datasets.',
           '',
           r'Top panels report the block-diagonal predictor and its negative last-epoch '
           r'stochastic objective per training datum; bottom panels report the standard '
           r'sparse SVGP predictor after the 300-epoch transplant protocol. '
           r'The latter is not epoch-zero sparsification. Curves show means and bars '
           r'one standard error over ten seeds. Inducing budgets use a linear axis, '
           r'and matched block-diagonal/sparse accuracy panels share vertical limits. '
           r'Targets are standardized.',
           '',
           r'The recorded training objectives are method-specific stochastic estimates, '
           r'not a common exact marginal likelihood or a convergence certificate. '
           r'These figures compare predictive accuracy, not equal training cost or '
           r'hardware-normalized runtime. Numerical aggregates are reused without '
           r'modification after checking their stored certificate hashes and complete '
           r'ten-seed grids. No fitting or prediction is performed for this appendix.', '']
    # Preserve the author's introduction/citations when updating figure layout.
    existing = REPORT / 'appendix_svgp_frozen.tex'
    if existing.exists():
        intro = existing.read_text().split(r'\clearpage',1)[0]
        intro = intro.replace('Top panels report', 'The left three columns report')
        intro = intro.replace('; bottom panels report', '; the right two columns report')
        intro = intro.split(r'\paragraph{Gibbs inducing-anchor update.}')[0].rstrip()
        intro = intro.replace('all 5120 sparse SVGP transplants',
                              '5120 matched sparse SVGP transplants')
        intro += ('\n' + r'\paragraph{Gibbs inducing-anchor update.} '
                  r'The Gibbs (S)BT-SVGP donor curves and final stochastic objectives '
                  r'use the certified 320-run inducing-anchor campaign, initialized '
                  r'from the corresponding RBF checkpoints and trained for 300 additional Gibbs epochs. '
                  r'the matched sparse SVGP transplant uses the same donor lineage '
                  r'and is trained for a further 150 epochs with fixed kernel parameters.' + '\n')
        tex = [intro.rstrip(), '']
    for kernel in KERNELS:
        for group in range(2):
            datasets = DATASETS[group*4:(group+1)*4]
            file = target / f'{kernel}_{group+1}.pdf'
            if kernel in args.kernels:
                draw(table, kernel, datasets, file)
            assert file.is_file(), file
            files.append(file)
            title = NAMES[kernel].replace('Matérn', r"Mat\'ern")
            gibbs_note = (r' The (S)BT donor and sparse curve use the certified '
                          r'inducing-anchor lineage; the sparse curve is the matched '
                          r'300-epoch SVGP transplant.'
                          if kernel == 'gibbs' else '')
            tex += [r'\clearpage', r'\begin{landscape}',r'\begin{figure}[p]', r'\centering',
                    r'\captionsetup{width=\linewidth}',
                    r'\includegraphics[width=\linewidth]{figures/svgp_frozen_campaign/' + file.name + '}',
                    r'\caption{Frozen ' + title + ' campaign: ' + ', '.join(NAMES[d] for d in datasets) + '. '
                    r'Left: complete-predictor RMSE/NLL and negative final stochastic objective (solid). '
                    r'Right: sparse SVGP RMSE/NLL after 300-epoch transplant refinement (dashed). '
                    r'Mean and one standard error, ten seeds; matched accuracy scales.' + gibbs_note + '}',
                    r'\label{fig:svgp-frozen-' + kernel + '-' + str(group+1) + '}', r'\end{figure}',r'\end{landscape}', '']
    appendix = REPORT / 'appendix_svgp_frozen.tex'
    appendix.write_text('\n'.join(tex))
    summary = REPORT / 'Results/svgp_frozen_campaign_summary.csv'
    with summary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variant','kernel','dataset','M','method','metric','n','mean','sem'])
        for key, values in sorted(table.items()): writer.writerow([*key,10,*values])
    (REPORT / 'Results/svgp_frozen_campaign_manifest.json').write_text(json.dumps(dict(
        scope='Full donor grid with 320 Gibbs SBT anchor replacements; 5120 matched sparse runs shown',
        inputs=provenance, main_page_values_verified=True,
        outputs={str(p.relative_to(REPORT)): digest(p) for p in [*files, appendix, summary]},
        note='Verifies certified aggregates and coverage; does not rerun numerical certification.'), indent=2))
    print(f'Built {len(files)} pages; {len(table)} aggregate cells; 5120 donors + 5120 matched sparse runs.')


if __name__ == '__main__':
    main()
