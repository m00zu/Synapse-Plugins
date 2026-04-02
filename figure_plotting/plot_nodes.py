"""
nodes/plot_nodes.py
===================
Visualization and plotting nodes.
"""
import copy
import json
import re
import NodeGraphQt
from PySide6 import QtCore, QtWidgets, QtGui
from data_models import TableData, FigureData, StatData
import pandas as pd
import numpy as np
from nodes.base import BaseExecutionNode, PORT_COLORS, ColorPickerButtonWidget, NodeToolBoxWidget

# Set Arial as default font for consistent rendering between matplotlib and SVG
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['svg.fonttype'] = 'none'  # output real <text> elements


# ── Figure-parameter helpers ──────────────────────────────────────────────────

def _extract_params(fig):
    """Read the current aesthetic state of a matplotlib Figure into a dict."""
    from matplotlib.colors import to_rgba

    def _c(color):
        try:
            return list(to_rgba(color))
        except Exception:
            return [0.0, 0.0, 0.0, 1.0]

    _LS_NORM = {
        'solid': 'solid', '-': 'solid',
        'dashed': 'dashed', '--': 'dashed',
        'dotted': 'dotted', ':': 'dotted',
        'dashdot': 'dashdot', '-.': 'dashdot',
        'None': 'None', 'none': 'None', '': 'None',
    }

    if not fig.axes:
        return {}
    ax = fig.axes[0]
    params = {}

    # Text elements
    for key, obj, axis_obj in [('title',  ax.title,        None),
                                ('xaxis',  ax.xaxis.label,  ax.xaxis),
                                ('yaxis',  ax.yaxis.label,  ax.yaxis)]:
        entry = {
            'text':       obj.get_text(),
            'fontsize':   float(obj.get_fontsize()),
            'fontweight': str(obj.get_fontweight()),
            'color':      _c(obj.get_color()),
        }
        if axis_obj is not None:
            entry['labelpad'] = float(axis_obj.labelpad)
        if key == 'title':
            entry['pad'] = 6.0   # matplotlib default; user-editable in dialog
        params[key] = entry

    # Ticks
    for axis_key, axis in [('xtick', ax.xaxis), ('ytick', ax.yaxis)]:
        major_ticks = axis.get_major_ticks()
        minor_ticks = axis.get_minor_ticks()
        if major_ticks:
            t = major_ticks[0]
            try:   direction = t.get_tickdir()
            except Exception: direction = 'out'
            params[axis_key] = {
                'labelsize':     float(t.label1.get_size()),
                'labelrotation': float(t.label1.get_rotation()),
                'direction':     direction,
                'length':        float(t.tick1line.get_markersize()),
                'major_visible': bool(t.tick1line.get_visible()),
            }
        else:
            params[axis_key] = {'labelsize': 10.0, 'labelrotation': 0.0,
                                 'direction': 'out', 'length': 3.5,
                                 'major_visible': True}
        if minor_ticks:
            mt = minor_ticks[0]
            params[axis_key]['minor_visible'] = bool(mt.tick1line.get_visible())
            params[axis_key]['minor_length']  = float(mt.tick1line.get_markersize())
        else:
            params[axis_key]['minor_visible'] = False
            params[axis_key]['minor_length']  = 2.0

    # Grid
    for grid_key, axis in [('xgrid', ax.xaxis), ('ygrid', ax.yaxis)]:
        ticks = axis.get_major_ticks()
        if ticks:
            g = ticks[0].gridline
            ls = g.get_linestyle()
            params[grid_key] = {
                'visible':   bool(g.get_visible()),
                'linestyle': _LS_NORM.get(ls, 'solid') if isinstance(ls, str) else 'solid',
                'linewidth': float(g.get_linewidth()),
                'color':     _c(g.get_color()),
            }
        else:
            params[grid_key] = {'visible': False, 'linestyle': 'dashed',
                                 'linewidth': 0.8, 'color': [0.5, 0.5, 0.5, 1.0]}

    # Axis limits & figure size
    params['limit']   = {'x_lim': list(ax.get_xlim()), 'y_lim': list(ax.get_ylim())}
    params['figsize'] = list(fig.get_size_inches())
    params['dpi']     = float(fig.get_dpi())

    # Colours
    params['background'] = {
        'axes_bg': _c(ax.get_facecolor()),
        'fig_bg':  _c(fig.get_facecolor()),
    }

    # Padding
    sp = fig.subplotpars
    params['padding'] = {k: round(getattr(sp, k), 4)
                         for k in ('left', 'right', 'top', 'bottom')}

    # Spines
    params['spines'] = {}
    for pos in ('left', 'right', 'top', 'bottom'):
        spine = ax.spines[pos]
        params['spines'][pos] = {
            'visible':   bool(spine.get_visible()),
            'linewidth': float(spine.get_linewidth()),
            'color':     _c(spine.get_edgecolor()),
        }

    # Lines
    # Include unnamed/private lines too (eg single grayscale intensity profile)
    # and assign stable synthetic display names so they are editable in UI.
    all_lines = list(ax.get_lines())
    if all_lines:
        params['lines'] = {}
        seen_labels: dict = {}
        for i, line in enumerate(all_lines):
            raw_label = str(line.get_label() or '')
            if raw_label and not raw_label.startswith('_'):
                label = raw_label
            else:
                label = f'Line {i + 1}'
            if label in seen_labels:
                seen_labels[label] += 1
                label = f'{label} ({seen_labels[label]})'
            else:
                seen_labels[label] = 0
            ls = line.get_linestyle()
            marker = line.get_marker()
            params['lines'][label] = {
                '_idx':            i,
                '_xdata':          [float(v) for v in line.get_xdata()],
                '_ydata':          [float(v) for v in line.get_ydata()],
                'x_offset':        0.0,
                'y_offset':        0.0,
                'linestyle':       _LS_NORM.get(ls, 'solid') if isinstance(ls, str) else 'solid',
                'linewidth':       float(line.get_linewidth()),
                'color':           _c(line.get_color()),
                'marker':          str(marker) if marker not in (None, 'None', 'none') else 'None',
                'markersize':      float(line.get_markersize()),
                'markerfacecolor': _c(line.get_markerfacecolor()),
                'markeredgecolor': _c(line.get_markeredgecolor()),
                'markeredgewidth': float(line.get_markeredgewidth()),
                'alpha':           float(line.get_alpha() or 1.0),
                'visible':         bool(line.get_visible()),
            }

    # Collections (PathCollection from scatter / swarm; PolyCollection from violin / boxplot)
    import numpy as _np
    from matplotlib.markers import MarkerStyle as _MS
    from matplotlib.collections import PathCollection as _PathColl

    def _detect_marker(coll):
        """Try to identify the marker string by comparing paths to known styles."""
        paths = coll.get_paths()
        if not paths:
            return 'o'
        test_verts = paths[0].vertices
        for m_str in ['o', '.', 's', '^', 'v', '<', '>', 'D', 'P', 'X',
                      'x', '+', '*', 'p', 'h', 'H', '|', '_']:
            try:
                ms_obj = _MS(m_str)
                ref = ms_obj.get_path().transformed(ms_obj.get_transform())
                rv = ref.vertices
                if len(test_verts) == len(rv) and _np.allclose(test_verts, rv, atol=0.05):
                    return m_str
            except Exception:
                pass
        return 'o'

    # Build legend-handle → label map for true group names
    _legend_labels: dict = {}
    try:
        handles, labels = ax.get_legend_handles_labels()
        for h, lbl in zip(handles, labels):
            _legend_labels[id(h)] = lbl
    except Exception:
        pass

    seen: dict = {}
    coll_list = []
    for i, c in enumerate(ax.collections):
        # Priority: legend label > artist label > fallback
        display = _legend_labels.get(id(c), '')
        if not display:
            try:
                raw_label = c.get_label() or ''
            except Exception:
                raw_label = ''
            display = raw_label if (raw_label and not raw_label.startswith('_')) else f'Group {i}'
        # Deduplicate display names
        if display in seen:
            seen[display] += 1
            display = f'{display} ({seen[display]})'
        else:
            seen[display] = 0
        fc  = c.get_facecolor()
        ec  = c.get_edgecolor()
        # Only PathCollections (scatter dots) use point sizes meaningfully;
        # PolyCollections (violin / box bodies) use polygon vertices instead.
        _is_path = isinstance(c, _PathColl)
        szs = c.get_sizes() if _is_path else []
        lws = c.get_linewidth()
        # For PolyCollections the per-vertex alpha is embedded in fc; c.get_alpha()
        # returns None, so we read alpha from the first face-color entry.
        if _is_path:
            alpha_val = float(c.get_alpha() if c.get_alpha() is not None else 1.0)
        else:
            alpha_val = float(fc[0][3] if len(fc) > 0 and len(fc[0]) >= 4 else
                              (c.get_alpha() if c.get_alpha() is not None else 1.0))
        coll_list.append({
            '_idx':      i,
            '_key':      display,
            '_is_poly':  not _is_path,   # True for violin/box PolyCollections
            'facecolor': _c(fc[0]  if len(fc)  > 0 else [0.5, 0.5, 0.5, 1.0]),
            'edgecolor': _c(ec[0]  if len(ec)  > 0 else [0.0, 0.0, 0.0, 0.0]),
            'size':      float(_np.median(szs) if len(szs) > 0 else 36.0),
            'alpha':     alpha_val,
            'visible':   bool(c.get_visible()),
            'marker':    _detect_marker(c) if _is_path else 'o',
            'edgewidth': float(_np.median(lws) if len(lws) > 0 else 0.0),
        })
    if coll_list:
        params['collections'] = coll_list

    # Patches (Rectangle / PathPatch from barplot / histplot / boxplot etc.)
    # Labeled patches (e.g. barplot): deduplicated by label, applied by label.
    # Unlabeled patches (e.g. boxplot PathPatches): given auto-key, applied by _idx.
    patch_list = []
    patch_seen: dict = {}
    _anon_counter = 0
    # Collect _nolegend_ patches separately — they need color-grouping
    _nolegend_by_color: dict = {}
    for i, p in enumerate(ax.patches):
        try:
            lbl = p.get_label() or ''
        except Exception:
            lbl = ''
        # _nolegend_ → histogram / bar patches; group by facecolor below
        if lbl == '_nolegend_':
            fc_key = tuple(round(float(v), 4) for v in _c(p.get_facecolor()))
            _nolegend_by_color.setdefault(fc_key, []).append((i, p))
            continue
        if lbl.startswith('_'):     # skip internal matplotlib child markers
            continue
        by_idx = not bool(lbl)      # True for empty-label patches (seaborn boxplot)
        display = lbl if lbl else f'Box {_anon_counter}'
        if not by_idx:
            _anon_counter = 0  # reset counter if we switched back to labeled
        if display in patch_seen:
            if not by_idx:
                continue            # skip duplicate-label (same group, multiple bars)
        else:
            patch_seen[display] = True
            if by_idx:
                _anon_counter += 1
        fc = p.get_facecolor()
        ec = p.get_edgecolor()
        patch_list.append({
            '_idx':    i,           # absolute position in ax.patches
            '_key':    display,
            '_by_idx': by_idx,      # True → match by index; False → match by label
            'facecolor': _c(fc),
            'edgecolor': _c(ec),
            'alpha':   float(p.get_alpha() if p.get_alpha() is not None else 1.0),
            'visible': bool(p.get_visible()),
            'edgewidth': float(p.get_linewidth()),
        })
    # Group _nolegend_ patches (histogram / bar) by facecolor.
    # Each color-group becomes one editable entry: "Bars 1", "Bars 2", …
    for gi, (fc_key, group) in enumerate(_nolegend_by_color.items()):
        _i0, p0 = group[0]
        display = f'Bars {gi + 1}'
        patch_list.append({
            '_indices': [idx for idx, _ in group],  # all bar indices
            '_key':     display,
            '_by_idx':  False,
            '_by_indices': True,
            'facecolor': _c(p0.get_facecolor()),
            'edgecolor': _c(p0.get_edgecolor()),
            'alpha':   float(p0.get_alpha() if p0.get_alpha() is not None else 1.0),
            'visible': bool(p0.get_visible()),
            'edgewidth': float(p0.get_linewidth()),
        })
    if patch_list:
        params['patches'] = patch_list

    # Font family (read from title as representative)
    try:
        ff = ax.title.get_fontfamily()
        params['font_family'] = ff[0] if ff else 'sans-serif'
    except Exception:
        params['font_family'] = 'sans-serif'

    # DPI
    params['dpi'] = float(fig.get_dpi())

    # Legend
    _LOC_INT_TO_STR = {0: 'best', 1: 'upper right', 2: 'upper left',
                       3: 'lower left', 4: 'lower right', 5: 'right',
                       6: 'center left', 7: 'center right',
                       8: 'lower center', 9: 'upper center', 10: 'center'}
    legend = ax.get_legend()
    if legend is not None:
        texts = legend.get_texts()
        title_artist = legend.get_title()
        loc_int = getattr(legend, '_loc', 1)
        frame = legend.get_frame()
        ncols = getattr(legend, '_ncols', None)
        if ncols is None and hasattr(legend, 'get_ncols'):
            try:
                ncols = legend.get_ncols()
            except Exception:
                ncols = 1
        if ncols is None:
            ncols = 1
        params['legend'] = {
            'visible':        bool(legend.get_visible()),
            'frameon':        bool(legend.get_frame_on()),
            'fontsize':       float(texts[0].get_fontsize() if texts else 10.0),
            'title':          title_artist.get_text() if title_artist else '',
            'title_fontsize': float(title_artist.get_fontsize() if title_artist else 10.0),
            'loc':            _LOC_INT_TO_STR.get(loc_int, 'upper right'),
            'labelcolor':     _c(texts[0].get_color() if texts else [0.0, 0.0, 0.0, 1.0]),
            'framealpha':     float(frame.get_alpha() if frame and frame.get_alpha() is not None else 1.0),
            'facecolor':      _c(frame.get_facecolor() if frame else [1.0, 1.0, 1.0, 1.0]),
            'edgecolor':      _c(frame.get_edgecolor() if frame else [0.0, 0.0, 0.0, 1.0]),
            'ncols':          int(ncols),
            'labels':         [t.get_text() for t in texts] if texts else [],
            'markerscale':    float(getattr(legend, 'markerscale', 1.0)),
            'borderpad':      float(getattr(legend, 'borderpad', 0.4)),
            'labelspacing':   float(getattr(legend, 'labelspacing', 0.5)),
            'handlelength':   float(getattr(legend, 'handlelength', 2.0)),
            'handletextpad':  float(getattr(legend, 'handletextpad', 0.8)),
            'columnspacing':  float(getattr(legend, 'columnspacing', 2.0)),
            'borderaxespad':  float(getattr(legend, 'borderaxespad', 0.5)),
        }

    # Custom text annotations (tagged with gid='synapse_text')
    import matplotlib.colors as _mc
    text_list = []
    for txt in ax.texts:
        if txt.get_gid() == 'synapse_text':
            try:
                rgba = list(_mc.to_rgba(txt.get_color()))
            except Exception:
                rgba = [0., 0., 0., 1.]
            text_list.append({
                'text':     txt.get_text(),
                'x':        float(txt.get_position()[0]),
                'y':        float(txt.get_position()[1]),
                'fontsize': float(txt.get_fontsize()),
                'color':    rgba,
                'ha':       str(txt.get_ha()),
                'va':       str(txt.get_va()),
                'rotation': float(txt.get_rotation()),
            })
    params['texts'] = text_list

    return params


def _apply_params(fig, params):
    """Apply a params dict to a matplotlib Figure (in-place)."""
    if not fig.axes or not params:
        return
    ax = fig.axes[0]

    # Text
    for key, obj, axis_obj in [('title',  ax.title,        None),
                                ('xaxis',  ax.xaxis.label,  ax.xaxis),
                                ('yaxis',  ax.yaxis.label,  ax.yaxis)]:
        p = params.get(key)
        if not p:
            continue
        if p.get('text')       is not None: obj.set_text(p['text'])
        if p.get('fontsize')   is not None: obj.set_fontsize(p['fontsize'])
        if p.get('fontweight') is not None: obj.set_fontweight(p['fontweight'])
        if p.get('color')      is not None: obj.set_color(p['color'])
        if axis_obj is not None and p.get('labelpad') is not None:
            axis_obj.labelpad = float(p['labelpad'])
        if key == 'title' and p.get('pad') is not None:
            try:
                ax.set_title(ax.get_title(), pad=float(p['pad']))
            except Exception:
                pass

    # Ticks
    import matplotlib.ticker as _mticker
    for axis_key, axis_name in [('xtick', 'x'), ('ytick', 'y')]:
        p = params.get(axis_key)
        if p:
            ax.tick_params(axis=axis_name, which='major',
                           labelsize=p.get('labelsize', 10.0),
                           labelrotation=p.get('labelrotation', 0.0),
                           direction=p.get('direction', 'out'),
                           length=p.get('length', 3.5))
            if 'major_visible' in p:
                visible = bool(p['major_visible'])
                if axis_name == 'x':
                    ax.tick_params(axis='x', which='major', bottom=visible, top=False)
                else:
                    ax.tick_params(axis='y', which='major', left=visible, right=False)
            if p.get('minor_visible', False):
                if axis_name == 'x':
                    ax.xaxis.set_minor_locator(_mticker.AutoMinorLocator())
                else:
                    ax.yaxis.set_minor_locator(_mticker.AutoMinorLocator())
                ax.tick_params(axis=axis_name, which='minor',
                               length=p.get('minor_length', 2.0))
            else:
                if axis_name == 'x':
                    ax.xaxis.set_minor_locator(_mticker.NullLocator())
                else:
                    ax.yaxis.set_minor_locator(_mticker.NullLocator())

    # Grid
    for grid_key, axis_obj in [('xgrid', ax.xaxis), ('ygrid', ax.yaxis)]:
        p = params.get(grid_key)
        if p:
            if p.get('visible', False):
                axis_obj.grid(visible=True,
                              linestyle=p.get('linestyle', 'dashed'),
                              linewidth=p.get('linewidth', 0.8),
                              color=p.get('color', [0.5, 0.5, 0.5, 1.0]))
            else:
                axis_obj.grid(visible=False)

    # Limits
    p = params.get('limit')
    if p:
        if p.get('x_lim'): ax.set_xlim(p['x_lim'])
        if p.get('y_lim'): ax.set_ylim(p['y_lim'])

    # Figure size
    if params.get('figsize'):
        fig.set_size_inches(*params['figsize'])
    if params.get('dpi'):
        fig.set_dpi(float(params['dpi']))

    # Background
    p = params.get('background')
    if p:
        if p.get('axes_bg') is not None: ax.set_facecolor(p['axes_bg'])
        if p.get('fig_bg')  is not None: fig.set_facecolor(p['fig_bg'])

    # Padding
    p = params.get('padding')
    if p:
        try:
            fig.subplots_adjust(**p)
        except Exception:
            pass

    # Spines
    for pos, sp_dict in (params.get('spines') or {}).items():
        if pos in ax.spines:
            spine = ax.spines[pos]
            if sp_dict.get('visible')   is not None: spine.set_visible(sp_dict['visible'])
            if sp_dict.get('linewidth') is not None: spine.set_linewidth(sp_dict['linewidth'])
            if sp_dict.get('color')     is not None: spine.set_edgecolor(sp_dict['color'])

    # Lines
    if params.get('lines'):
        all_lines = list(ax.get_lines())
        line_map = {l.get_label(): l for l in all_lines
                    if str(l.get_label() or '') and not str(l.get_label()).startswith('_')}
        for label, lp in params['lines'].items():
            line = None
            idx = lp.get('_idx')
            if isinstance(idx, int) and 0 <= idx < len(all_lines):
                line = all_lines[idx]
            if line is None:
                line = line_map.get(label)
            if line is None:
                continue
            for attr, val in lp.items():
                if attr in ('_idx', '_xdata', '_ydata', 'x_offset', 'y_offset'):
                    continue
                try:
                    getattr(line, f'set_{attr}')(val)
                except Exception:
                    pass
            # Apply positional offsets relative to the stored baseline
            dx = float(lp.get('x_offset', 0.0))
            dy = float(lp.get('y_offset', 0.0))
            text_gap = float(lp.get('text_gap', 0.0))
            base_x = lp.get('_xdata', list(line.get_xdata()))
            base_y = lp.get('_ydata', list(line.get_ydata()))
            if dx != 0.0 or dy != 0.0:
                line.set_xdata([x + dx for x in base_x])
                line.set_ydata([y + dy for y in base_y])
            # Reposition the matching stat-text annotation (idempotent)
            stat_gid = f'stat_text:{label}'
            for txt in ax.texts:
                if txt.get_gid() == stat_gid:
                    mid_x = sum(base_x) / len(base_x) + dx if base_x else 0
                    line_y = max(base_y) + dy if base_y else 0
                    txt.set_position((mid_x, line_y + text_gap))

    # Collections (PathCollection from scatter / swarm / strip; PolyCollection from violin / boxplot)
    if params.get('collections'):
        from matplotlib.markers import MarkerStyle as _MS
        colls = ax.collections
        for entry in params['collections']:
            idx = entry.get('_idx', -1)
            if not (0 <= idx < len(colls)):
                continue
            c = colls[idx]
            _is_poly = entry.get('_is_poly', False)
            if entry.get('facecolor') is not None:
                c.set_facecolor(entry['facecolor'])
            if entry.get('edgecolor') is not None:
                c.set_edgecolor(entry['edgecolor'])
            # set_sizes / set_alpha are meaningful only for PathCollections (scatter dots).
            # Calling them on PolyCollections (violin/box bodies) corrupts the geometry.
            if not _is_poly:
                if entry.get('size') is not None:
                    n = max(1, len(c.get_offsets()))
                    c.set_sizes([entry['size']] * n)
                if entry.get('alpha') is not None:
                    c.set_alpha(entry['alpha'])
                if entry.get('marker') is not None and entry['marker'] != 'None':
                    try:
                        ms_obj = _MS(entry['marker'])
                        path = ms_obj.get_path().transformed(ms_obj.get_transform())
                        c.set_paths([path])
                    except Exception:
                        pass
                if entry.get('edgewidth') is not None:
                    c.set_linewidth(entry['edgewidth'])
            else:
                # For PolyCollections: only set_visible and edgewidth are safe
                if entry.get('edgewidth') is not None:
                    c.set_linewidth(entry['edgewidth'])
            if entry.get('visible') is not None:
                c.set_visible(entry['visible'])

    # Patches (barplot / histplot / boxplot)
    if params.get('patches'):
        for entry in params['patches']:
            key    = entry['_key']
            by_idx = entry.get('_by_idx', False)
            idx    = entry.get('_idx', -1)

            if entry.get('_by_indices'):
                # Grouped _nolegend_ patches (histogram bars): apply by stored indices
                indices = entry.get('_indices', [])
                patches_to_update = [ax.patches[j] for j in indices
                                     if 0 <= j < len(ax.patches)]
            elif by_idx:
                # Un-labeled patches (e.g. seaborn boxplot PathPatch): apply by index
                patches_to_update = [ax.patches[idx]] if 0 <= idx < len(ax.patches) else []
            else:
                # Labeled patches (e.g. barplot): apply to all patches with matching label
                patches_to_update = []
                for p in ax.patches:
                    try:
                        lbl = p.get_label() or ''
                    except Exception:
                        lbl = ''
                    if lbl == key:
                        patches_to_update.append(p)

            for p in patches_to_update:
                if entry.get('facecolor') is not None:
                    p.set_facecolor(entry['facecolor'])
                if entry.get('edgecolor') is not None:
                    p.set_edgecolor(entry['edgecolor'])
                if entry.get('alpha') is not None:
                    p.set_alpha(entry['alpha'])
                if entry.get('visible') is not None:
                    p.set_visible(entry['visible'])
                if entry.get('edgewidth') is not None:
                    p.set_linewidth(entry['edgewidth'])

    # Font family
    if params.get('font_family'):
        ff = params['font_family']
        for obj in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            try: obj.set_fontfamily(ff)
            except Exception: pass
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            try: lbl.set_fontfamily(ff)
            except Exception: pass
        for txt in ax.texts:          # covers ax.text() and ax.annotate() labels
            try: txt.set_fontfamily(ff)
            except Exception: pass

    # DPI
    if params.get('dpi') is not None:
        try: fig.set_dpi(float(params['dpi']))
        except Exception: pass

    # Legend
    p = params.get('legend')
    if p:
        legend = ax.get_legend()
        if legend:
            # Check for custom label sorting
            if p.get('labels') is not None:
                # We need to rebuild the legend entirely to change its drawing order
                handles, current_labels = ax.get_legend_handles_labels()
                lbl_to_hdl = dict(zip(current_labels, handles))
                new_labels = []
                new_handles = []
                for tgt_lbl in p['labels']:
                    if tgt_lbl in lbl_to_hdl:
                        new_labels.append(tgt_lbl)
                        new_handles.append(lbl_to_hdl[tgt_lbl])
                # Append any remaining ones that might not have been in the order list
                for lbl, hdl in zip(current_labels, handles):
                    if lbl not in new_labels:
                        new_labels.append(lbl)
                        new_handles.append(hdl)
                if new_handles:
                    old_title = legend.get_title().get_text() if legend.get_title() else None
                    legend = ax.legend(new_handles, new_labels, title=old_title)

            if p.get('visible')  is not None: legend.set_visible(p['visible'])
            if p.get('frameon')  is not None: legend.set_frame_on(p['frameon'])
            if p.get('fontsize') is not None:
                for t in legend.get_texts():
                    t.set_fontsize(p['fontsize'])
            if p.get('labelcolor') is not None:
                for t in legend.get_texts():
                    t.set_color(p['labelcolor'])
            title_artist = legend.get_title()
            if title_artist:
                if p.get('title')          is not None: title_artist.set_text(p['title'])
                if p.get('title_fontsize') is not None: title_artist.set_fontsize(p['title_fontsize'])
            frame = legend.get_frame()
            if frame:
                if p.get('framealpha') is not None:
                    frame.set_alpha(float(p['framealpha']))
                if p.get('facecolor') is not None:
                    frame.set_facecolor(p['facecolor'])
                if p.get('edgecolor') is not None:
                    frame.set_edgecolor(p['edgecolor'])
            if p.get('ncols') is not None:
                try:
                    legend._ncols = int(p['ncols'])
                except Exception:
                    pass
            for attr in ('markerscale', 'borderpad', 'labelspacing',
                         'handlelength', 'handletextpad', 'columnspacing', 'borderaxespad'):
                if p.get(attr) is not None and hasattr(legend, attr):
                    try:
                        setattr(legend, attr, float(p[attr]))
                    except Exception:
                        pass
            if p.get('loc') is not None:
                _LOC_STR_TO_INT = {'best': 0, 'upper right': 1, 'upper left': 2,
                                   'lower left': 3, 'lower right': 4, 'right': 5,
                                   'center left': 6, 'center right': 7,
                                   'lower center': 8, 'upper center': 9, 'center': 10}
                legend._loc = _LOC_STR_TO_INT.get(p['loc'], 1)

    # Custom text annotations
    for txt in list(ax.texts):
        if txt.get_gid() == 'synapse_text':
            txt.remove()
    for entry in params.get('texts', []):
        ax.text(
            float(entry.get('x', 0.5)),
            float(entry.get('y', 0.5)),
            str(entry.get('text', '')),
            fontsize=float(entry.get('fontsize', 12.0)),
            color=entry.get('color', [0., 0., 0., 1.]),
            ha=str(entry.get('ha', 'center')),
            va=str(entry.get('va', 'center')),
            rotation=float(entry.get('rotation', 0.0)),
            transform=ax.transData,
            gid='synapse_text',
        )


# ── Figure-editor dialog ──────────────────────────────────────────────────────

class FigureEditDialog(QtWidgets.QDialog):
    """
    Tabbed dialog for editing matplotlib figure aesthetics.

    Accepts a params dict (from `_extract_params` or previously stored)
    and returns the modified dict via `get_params()` after the user
    clicks OK.
    """
    _LINESTYLES = ['solid', 'dashed', 'dotted', 'dashdot', 'None']
    _WEIGHTS    = ['normal', 'bold', 'light', 'heavy', 'ultralight']
    _TICK_DIRS  = ['in', 'out', 'inout']
    _MARKERS    = ['None', '.', 'o', 's', '^', 'v', '<', '>', 'D', 'x', '+', '*', 'p', 'h', '|', '_']

    def __init__(self, params: dict, parent=None, on_apply=None):
        super().__init__(parent)
        self.setWindowTitle("Figure Editor")
        self.setMinimumWidth(520)
        self._params = copy.deepcopy(params)
        self._color_cache: dict = {}   # key → [r, g, b, a]
        self._line_widgets: dict = {}  # line_label → {widget_name: widget}
        self._current_line: str = ''
        self._coll_widgets: dict = {}  # _key → {widget_name: widget}
        self._current_coll_key: str = ''
        self._patch_widgets: dict = {}  # _key → {widget_name: widget}
        self._current_patch_key: str = ''
        self._ann_idx: int = -1
        self._on_apply = on_apply      # callable(params) or None
        self._build_ui()

    # ── color-button factory ──────────────────────────────────────────────────

    def _color_btn(self, key: str, initial: list) -> QtWidgets.QPushButton:
        """A preview button that stores the selected RGBA and opens QColorDialog."""
        if key not in self._color_cache:
            self._color_cache[key] = list(initial)
        btn = QtWidgets.QPushButton()
        btn.setFixedSize(52, 22)
        self._refresh_btn(btn, self._color_cache[key])

        def _pick():
            r, g, b, a = self._color_cache[key]
            qc = QtGui.QColor.fromRgbF(r, g, b, a)
            new_qc = QtWidgets.QColorDialog.getColor(
                qc, self, "Select Color",
                QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)
            if new_qc.isValid():
                rgba = list(new_qc.getRgbF())
                self._color_cache[key] = rgba
                self._refresh_btn(btn, rgba)

        btn.clicked.connect(_pick)
        return btn

    @staticmethod
    def _refresh_btn(btn, rgba):
        r, g, b = [max(0.0, min(1.0, x)) for x in rgba[:3]]
        hex_col = QtGui.QColor.fromRgbF(r, g, b).name()
        btn.setStyleSheet(f"background-color:{hex_col}; border:1px solid #666;")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._make_text_tab(),  "Text")
        tabs.addTab(self._make_axis_tab(),  "Axis")
        tabs.addTab(self._make_style_tab(), "Style")
        if self._params.get('legend') is not None:
            tabs.addTab(self._make_legend_tab(), "Legend")
        if self._params.get('lines'):
            tabs.addTab(self._make_lines_tab(), "Lines")
        if self._params.get('collections'):
            tabs.addTab(self._make_collections_tab(), "Groups")
        if self._params.get('patches'):
            tabs.addTab(self._make_patches_tab(), "Bars")
        tabs.addTab(self._make_annotations_tab(), "Annotations")
        root.addWidget(tabs)

        bbox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bbox.accepted.connect(self._collect_and_accept)
        bbox.rejected.connect(self.reject)

        # Apply button — saves params and pings downstream without closing
        if self._on_apply is not None:
            apply_btn = QtWidgets.QPushButton("Apply")
            apply_btn.setToolTip("Save settings and update downstream nodes without closing")
            apply_btn.clicked.connect(self._do_apply)
            bbox.addButton(apply_btn, QtWidgets.QDialogButtonBox.ButtonRole.ApplyRole)

        root.addWidget(bbox)

    # ── text tab ──────────────────────────────────────────────────────────────

    def _text_section(self, layout, key, title):
        p = self._params.get(key, {})
        grp = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(grp)

        te = QtWidgets.QLineEdit(p.get('text', ''))
        form.addRow("Text", te);  setattr(self, f'_{key}_text', te)

        sz = QtWidgets.QDoubleSpinBox()
        sz.setRange(1, 72); sz.setSingleStep(0.5)
        sz.setValue(p.get('fontsize', 12.0))
        form.addRow("Font Size", sz);  setattr(self, f'_{key}_size', sz)

        wt = QtWidgets.QComboBox()
        wt.addItems(self._WEIGHTS)
        wt.setCurrentText(str(p.get('fontweight', 'normal')))
        form.addRow("Weight", wt);  setattr(self, f'_{key}_weight', wt)

        cb = self._color_btn(f'{key}_color', p.get('color', [0., 0., 0., 1.]))
        form.addRow("Color", cb)

        if key in ('xaxis', 'yaxis'):
            lp = QtWidgets.QDoubleSpinBox()
            lp.setRange(-50.0, 200.0); lp.setSingleStep(1.0); lp.setDecimals(1)
            lp.setValue(float(p.get('labelpad', 4.0)))
            form.addRow("Label Pad", lp);  setattr(self, f'_{key}_labelpad', lp)

        if key == 'title':
            tp = QtWidgets.QDoubleSpinBox()
            tp.setRange(-50.0, 200.0); tp.setSingleStep(1.0); tp.setDecimals(1)
            tp.setValue(float(p.get('pad', 6.0)))
            form.addRow("Title Pad", tp);  setattr(self, '_title_pad', tp)

        layout.addWidget(grp)

    def _make_text_tab(self):
        w = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(w)
        self._text_section(vbox, 'title', 'Title')
        self._text_section(vbox, 'xaxis', 'X Label')
        self._text_section(vbox, 'yaxis', 'Y Label')
        vbox.addStretch()
        return w

    # ── axis tab ──────────────────────────────────────────────────────────────

    def _make_axis_tab(self):
        inner = QtWidgets.QWidget()
        vbox  = QtWidgets.QVBoxLayout(inner)

        # Limits
        lim_grp  = QtWidgets.QGroupBox("Axis Limits")
        lim_form = QtWidgets.QFormLayout(lim_grp)
        xlim = self._params.get('limit', {}).get('x_lim', [0.0, 1.0])
        ylim = self._params.get('limit', {}).get('y_lim', [0.0, 1.0])
        for attr, label, val in [
            ('_xlim_lo', 'X Min', xlim[0]), ('_xlim_hi', 'X Max', xlim[1]),
            ('_ylim_lo', 'Y Min', ylim[0]), ('_ylim_hi', 'Y Max', ylim[1]),
        ]:
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-1e9, 1e9); sp.setDecimals(6); sp.setValue(float(val))
            lim_form.addRow(label, sp);  setattr(self, attr, sp)
        vbox.addWidget(lim_grp)

        # Ticks
        for axis_key, label in [('xtick', 'X Tick'), ('ytick', 'Y Tick')]:
            p = self._params.get(axis_key, {})
            grp  = QtWidgets.QGroupBox(label)
            form = QtWidgets.QFormLayout(grp)

            maj_vis = QtWidgets.QCheckBox("Major Ticks Visible")
            maj_vis.setChecked(bool(p.get('major_visible', True)))
            form.addRow("", maj_vis);  setattr(self, f'_{axis_key}_major_vis', maj_vis)

            sz = QtWidgets.QDoubleSpinBox()
            sz.setRange(1, 30); sz.setSingleStep(0.5); sz.setValue(p.get('labelsize', 10.0))
            form.addRow("Label Size", sz);  setattr(self, f'_{axis_key}_size', sz)

            rot = QtWidgets.QDoubleSpinBox()
            rot.setRange(0, 360); rot.setSingleStep(5.0); rot.setValue(p.get('labelrotation', 0.0))
            form.addRow("Rotation", rot);  setattr(self, f'_{axis_key}_rot', rot)

            dc = QtWidgets.QComboBox(); dc.addItems(self._TICK_DIRS)
            dc.setCurrentText(p.get('direction', 'out'))
            form.addRow("Direction", dc);  setattr(self, f'_{axis_key}_dir', dc)

            min_vis = QtWidgets.QCheckBox("Minor Ticks Visible")
            min_vis.setChecked(bool(p.get('minor_visible', False)))
            form.addRow("", min_vis);  setattr(self, f'_{axis_key}_minor_vis', min_vis)

            min_len = QtWidgets.QDoubleSpinBox()
            min_len.setRange(0, 20); min_len.setSingleStep(0.5)
            min_len.setValue(float(p.get('minor_length', 2.0)))
            form.addRow("Minor Length", min_len);  setattr(self, f'_{axis_key}_minor_len', min_len)

            vbox.addWidget(grp)

        # Grid
        for grid_key, label in [('xgrid', 'X Grid'), ('ygrid', 'Y Grid')]:
            p = self._params.get(grid_key, {})
            grp  = QtWidgets.QGroupBox(label)
            form = QtWidgets.QFormLayout(grp)

            vis = QtWidgets.QCheckBox("Visible")
            vis.setChecked(bool(p.get('visible', False)))
            form.addRow("", vis);  setattr(self, f'_{grid_key}_vis', vis)

            ls = QtWidgets.QComboBox(); ls.addItems(self._LINESTYLES)
            ls.setCurrentText(p.get('linestyle', 'dashed'))
            form.addRow("Line Style", ls);  setattr(self, f'_{grid_key}_ls', ls)

            lw = QtWidgets.QDoubleSpinBox()
            lw.setRange(0, 10); lw.setSingleStep(0.1); lw.setValue(p.get('linewidth', 0.8))
            form.addRow("Line Width", lw);  setattr(self, f'_{grid_key}_lw', lw)

            cb = self._color_btn(f'{grid_key}_color', p.get('color', [0.5, 0.5, 0.5, 1.0]))
            form.addRow("Color", cb)

            vbox.addWidget(grp)

        vbox.addStretch()
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(inner)
        outer = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(outer).addWidget(scroll)
        outer.layout().setContentsMargins(0, 0, 0, 0)
        return outer

    # ── style tab ────────────────────────────────────────────────────────────

    def _make_style_tab(self):
        inner = QtWidgets.QWidget()
        vbox  = QtWidgets.QVBoxLayout(inner)

        # Font family — populated from matplotlib's font cache
        from matplotlib import font_manager as _fm
        _generic   = ['sans-serif', 'serif', 'monospace']
        _system    = sorted({f.name for f in _fm.fontManager.ttflist})
        _all_fonts = _generic + [f for f in _system if f not in _generic]
        current_ff = self._params.get('font_family', 'sans-serif')
        if current_ff not in _all_fonts:
            _all_fonts.insert(len(_generic), current_ff)

        ff_g = QtWidgets.QGroupBox("Font")
        ff_f = QtWidgets.QFormLayout(ff_g)
        self._font_family_cb = QtWidgets.QComboBox()
        self._font_family_cb.setMaxVisibleItems(20)
        self._font_family_cb.setEditable(True)
        self._font_family_cb.addItems(_all_fonts)
        self._font_family_cb.setCurrentText(current_ff)
        ff_f.addRow("Family", self._font_family_cb)
        vbox.addWidget(ff_g)

        # Figure size
        fs    = self._params.get('figsize', [10.0, 8.0])
        fg    = QtWidgets.QGroupBox("Figure Size (inches)")
        fform = QtWidgets.QFormLayout(fg)
        self._fig_w = QtWidgets.QDoubleSpinBox(); self._fig_w.setRange(1, 50); self._fig_w.setSingleStep(0.5); self._fig_w.setValue(fs[0])
        self._fig_h = QtWidgets.QDoubleSpinBox(); self._fig_h.setRange(1, 50); self._fig_h.setSingleStep(0.5); self._fig_h.setValue(fs[1])
        fform.addRow("Width",  self._fig_w)
        fform.addRow("Height", self._fig_h)
        self._dpi_sb = QtWidgets.QDoubleSpinBox()
        self._dpi_sb.setRange(72, 600); self._dpi_sb.setSingleStep(12); self._dpi_sb.setDecimals(0)
        self._dpi_sb.setValue(float(self._params.get('dpi', 100.0)))
        fform.addRow("DPI", self._dpi_sb)
        vbox.addWidget(fg)

        # Background
        bg_p  = self._params.get('background', {})
        bg_g  = QtWidgets.QGroupBox("Background Colors")
        bg_f  = QtWidgets.QFormLayout(bg_g)
        bg_f.addRow("Axes BG",   self._color_btn('axes_bg', bg_p.get('axes_bg', [1., 1., 1., 1.])))
        bg_f.addRow("Figure BG", self._color_btn('fig_bg',  bg_p.get('fig_bg',  [1., 1., 1., 1.])))
        vbox.addWidget(bg_g)

        # Subplot padding
        pad_p = self._params.get('padding', {'left': 0.125, 'right': 0.9, 'top': 0.9, 'bottom': 0.11})
        pad_g = QtWidgets.QGroupBox("Subplot Padding (0–1)")
        pad_f = QtWidgets.QFormLayout(pad_g)
        for side in ('left', 'right', 'top', 'bottom'):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(0.0, 1.0); sp.setSingleStep(0.01); sp.setDecimals(3)
            sp.setValue(pad_p.get(side, 0.1))
            pad_f.addRow(side.capitalize(), sp);  setattr(self, f'_pad_{side}', sp)
        vbox.addWidget(pad_g)

        # Spines
        sp_p  = self._params.get('spines', {})
        sp_g  = QtWidgets.QGroupBox("Spines")
        sp_gl = QtWidgets.QGridLayout(sp_g)
        for col, pos in enumerate(('left', 'right', 'top', 'bottom')):
            d   = sp_p.get(pos, {'visible': True, 'linewidth': 1.0, 'color': [0., 0., 0., 1.]})
            vis = QtWidgets.QCheckBox(pos.capitalize()); vis.setChecked(bool(d.get('visible', True)))
            lw  = QtWidgets.QDoubleSpinBox(); lw.setRange(0, 10); lw.setSingleStep(0.5); lw.setValue(d.get('linewidth', 1.0))
            sp_gl.addWidget(QtWidgets.QLabel(pos.capitalize()), 0, col, QtCore.Qt.AlignmentFlag.AlignCenter)
            sp_gl.addWidget(vis, 1, col)
            sp_gl.addWidget(lw,  2, col)
            setattr(self, f'_spine_{pos}_vis', vis)
            setattr(self, f'_spine_{pos}_lw',  lw)
        vbox.addWidget(sp_g)

        vbox.addStretch()
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(inner)
        outer = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(outer).addWidget(scroll)
        outer.layout().setContentsMargins(0, 0, 0, 0)
        return outer

    # ── legend tab ───────────────────────────────────────────────────────────

    def _make_legend_tab(self):
        w    = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(w)
        p    = self._params.get('legend', {})
        grp  = QtWidgets.QGroupBox("Legend")
        form = QtWidgets.QFormLayout(grp)

        self._legend_vis = QtWidgets.QCheckBox("Visible")
        self._legend_vis.setChecked(bool(p.get('visible', True)))
        form.addRow("", self._legend_vis)

        self._legend_frameon = QtWidgets.QCheckBox("Show Frame")
        self._legend_frameon.setChecked(bool(p.get('frameon', True)))
        form.addRow("", self._legend_frameon)

        _LOC_CHOICES = ['best', 'upper right', 'upper left', 'lower left', 'lower right',
                        'right', 'center left', 'center right', 'lower center',
                        'upper center', 'center']
        self._legend_loc = QtWidgets.QComboBox()
        self._legend_loc.addItems(_LOC_CHOICES)
        self._legend_loc.setCurrentText(p.get('loc', 'upper right'))
        form.addRow("Location", self._legend_loc)

        self._legend_fontsize = QtWidgets.QDoubleSpinBox()
        self._legend_fontsize.setRange(1, 72); self._legend_fontsize.setSingleStep(0.5)
        self._legend_fontsize.setValue(float(p.get('fontsize', 10.0)))
        form.addRow("Font Size", self._legend_fontsize)

        self._legend_title = QtWidgets.QLineEdit(p.get('title', ''))
        form.addRow("Title", self._legend_title)

        self._legend_title_fs = QtWidgets.QDoubleSpinBox()
        self._legend_title_fs.setRange(1, 72); self._legend_title_fs.setSingleStep(0.5)
        self._legend_title_fs.setValue(float(p.get('title_fontsize', 10.0)))
        form.addRow("Title Font Size", self._legend_title_fs)

        self._legend_label_color = self._color_btn(
            'legend_label_color', p.get('labelcolor', [0.0, 0.0, 0.0, 1.0]))
        form.addRow("Label Color", self._legend_label_color)

        self._legend_ncols = QtWidgets.QSpinBox()
        self._legend_ncols.setRange(1, 12)
        self._legend_ncols.setValue(int(p.get('ncols', 1)))
        form.addRow("Columns", self._legend_ncols)

        self._legend_markerscale = QtWidgets.QDoubleSpinBox()
        self._legend_markerscale.setRange(0.1, 10.0)
        self._legend_markerscale.setSingleStep(0.1)
        self._legend_markerscale.setValue(float(p.get('markerscale', 1.0)))
        form.addRow("Marker Scale", self._legend_markerscale)

        self._legend_labelspacing = QtWidgets.QDoubleSpinBox()
        self._legend_labelspacing.setRange(0.0, 5.0)
        self._legend_labelspacing.setSingleStep(0.1)
        self._legend_labelspacing.setValue(float(p.get('labelspacing', 0.5)))
        form.addRow("Label Spacing", self._legend_labelspacing)

        self._legend_handlelength = QtWidgets.QDoubleSpinBox()
        self._legend_handlelength.setRange(0.0, 10.0)
        self._legend_handlelength.setSingleStep(0.1)
        self._legend_handlelength.setValue(float(p.get('handlelength', 2.0)))
        form.addRow("Handle Length", self._legend_handlelength)

        self._legend_handletextpad = QtWidgets.QDoubleSpinBox()
        self._legend_handletextpad.setRange(0.0, 5.0)
        self._legend_handletextpad.setSingleStep(0.1)
        self._legend_handletextpad.setValue(float(p.get('handletextpad', 0.8)))
        form.addRow("Handle Text Pad", self._legend_handletextpad)

        self._legend_columnspacing = QtWidgets.QDoubleSpinBox()
        self._legend_columnspacing.setRange(0.0, 8.0)
        self._legend_columnspacing.setSingleStep(0.1)
        self._legend_columnspacing.setValue(float(p.get('columnspacing', 2.0)))
        form.addRow("Column Spacing", self._legend_columnspacing)

        self._legend_borderpad = QtWidgets.QDoubleSpinBox()
        self._legend_borderpad.setRange(0.0, 5.0)
        self._legend_borderpad.setSingleStep(0.1)
        self._legend_borderpad.setValue(float(p.get('borderpad', 0.4)))
        form.addRow("Border Pad", self._legend_borderpad)

        self._legend_borderaxespad = QtWidgets.QDoubleSpinBox()
        self._legend_borderaxespad.setRange(0.0, 5.0)
        self._legend_borderaxespad.setSingleStep(0.1)
        self._legend_borderaxespad.setValue(float(p.get('borderaxespad', 0.5)))
        form.addRow("Border Axes Pad", self._legend_borderaxespad)

        self._legend_framealpha = QtWidgets.QDoubleSpinBox()
        self._legend_framealpha.setRange(0.0, 1.0)
        self._legend_framealpha.setSingleStep(0.05)
        self._legend_framealpha.setDecimals(2)
        self._legend_framealpha.setValue(float(p.get('framealpha', 1.0)))
        form.addRow("Frame Alpha", self._legend_framealpha)

        self._legend_face_color = self._color_btn(
            'legend_face_color', p.get('facecolor', [1.0, 1.0, 1.0, 1.0]))
        form.addRow("Frame Face", self._legend_face_color)

        self._legend_edge_color = self._color_btn(
            'legend_edge_color', p.get('edgecolor', [0.0, 0.0, 0.0, 1.0]))
        form.addRow("Frame Edge", self._legend_edge_color)
        
        # Add draggable legend ordering list if labels exist
        if 'labels' in p and p['labels']:
            self._legend_order_list = QtWidgets.QListWidget()
            self._legend_order_list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
            self._legend_order_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
            self._legend_order_list.setMinimumHeight(100)
            self._legend_order_list.addItems(p['labels'])
            form.addRow("Label Order\n(Drag to Reorder)", self._legend_order_list)
        else:
            self._legend_order_list = None

        vbox.addWidget(grp)
        vbox.addStretch()
        return w

    # ── lines tab ────────────────────────────────────────────────────────────

    def _make_lines_tab(self):
        w    = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(w)

        line_keys = list(self._params.get('lines', {}).keys())
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Line:"))
        self._line_selector = QtWidgets.QComboBox()
        self._line_selector.addItems(line_keys)
        row.addWidget(self._line_selector, 1)
        vbox.addLayout(row)

        self._line_prop_grp  = QtWidgets.QGroupBox("Properties")
        self._line_prop_form = QtWidgets.QFormLayout(self._line_prop_grp)
        vbox.addWidget(self._line_prop_grp)
        vbox.addStretch()

        if line_keys:
            self._build_line_widgets(line_keys[0])
        self._line_selector.currentTextChanged.connect(self._switch_line)
        return w

    def _build_line_widgets(self, label: str):
        if not label:
            return
        form = self._line_prop_form
        while form.rowCount():
            form.removeRow(0)

        p = self._params.get('lines', {}).get(label, {})

        vis = QtWidgets.QCheckBox("Visible"); vis.setChecked(bool(p.get('visible', True)))
        form.addRow("", vis)

        ls = QtWidgets.QComboBox(); ls.addItems(self._LINESTYLES)
        ls.setCurrentText(p.get('linestyle', 'solid'))
        form.addRow("Line Style", ls)

        lw = QtWidgets.QDoubleSpinBox(); lw.setRange(0, 20); lw.setSingleStep(0.5)
        lw.setValue(float(p.get('linewidth', 1.5)))
        form.addRow("Line Width", lw)

        cb = self._color_btn(f'_line_{label}_color', p.get('color', [0., 0., 1., 1.]))
        form.addRow("Line Color", cb)

        al = QtWidgets.QDoubleSpinBox(); al.setRange(0.0, 1.0); al.setSingleStep(0.05); al.setDecimals(2)
        al.setValue(float(p.get('alpha', 1.0)))
        form.addRow("Alpha", al)

        mk = QtWidgets.QComboBox(); mk.addItems(self._MARKERS)
        mk.setCurrentText(str(p.get('marker', 'None')))
        form.addRow("Marker", mk)

        ms = QtWidgets.QDoubleSpinBox(); ms.setRange(0, 30); ms.setSingleStep(0.5)
        ms.setValue(float(p.get('markersize', 6.0)))
        form.addRow("Marker Size", ms)

        mfc = self._color_btn(f'_line_{label}_mfc', p.get('markerfacecolor', [0., 0., 1., 1.]))
        form.addRow("Marker Fill", mfc)

        mec = self._color_btn(f'_line_{label}_mec', p.get('markeredgecolor', [0., 0., 0., 1.]))
        form.addRow("Marker Edge Color", mec)

        mew = QtWidgets.QDoubleSpinBox(); mew.setRange(0, 10); mew.setSingleStep(0.5)
        mew.setValue(float(p.get('markeredgewidth', 1.0)))
        form.addRow("Marker Edge Width", mew)

        # Position controls — shown only for tagged lines (stat brackets / subgroup brackets)
        _is_tagged = '–' in label or label.startswith('subgrp:')
        dx_spin = dy_spin = None
        if _is_tagged:
            form.addRow(QtWidgets.QLabel(""))  # spacer
            hdr = QtWidgets.QLabel("<b>Position Offset</b>")
            form.addRow(hdr)

            dx_spin = QtWidgets.QDoubleSpinBox()
            dx_spin.setRange(-50, 50); dx_spin.setSingleStep(0.1); dx_spin.setDecimals(3)
            dx_spin.setValue(float(p.get('x_offset', 0.0)))
            form.addRow("X Offset", dx_spin)

            dy_spin = QtWidgets.QDoubleSpinBox()
            dy_spin.setRange(-50, 50); dy_spin.setSingleStep(0.1); dy_spin.setDecimals(3)
            dy_spin.setValue(float(p.get('y_offset', 0.0)))
            form.addRow("Y Offset", dy_spin)

            tg_spin = QtWidgets.QDoubleSpinBox()
            tg_spin.setRange(-10, 10); tg_spin.setSingleStep(0.05); tg_spin.setDecimals(3)
            tg_spin.setValue(float(p.get('text_gap', 0.0)))
            tg_spin.setToolTip("Extra gap between the bracket line and the stars / ns text")
            form.addRow("Text–Line Gap", tg_spin)

        self._line_widgets[label] = {
            'vis': vis, 'ls': ls, 'lw': lw,
            'color_key': f'_line_{label}_color',
            'al': al,
            'mk': mk, 'ms': ms,
            'mfc_key': f'_line_{label}_mfc',
            'mec_key': f'_line_{label}_mec',
            'mew': mew,
            'dx': dx_spin,
            'dy': dy_spin,
            'tg': tg_spin if _is_tagged else None,
        }
        self._current_line = label

    def _save_current_line(self):
        label = self._current_line
        if not label or label not in self._line_widgets:
            return
        w  = self._line_widgets[label]
        lp = self._params.setdefault('lines', {}).setdefault(label, {})
        lp['visible']          = w['vis'].isChecked()
        lp['linestyle']        = w['ls'].currentText()
        lp['linewidth']        = w['lw'].value()
        lp['color']            = self._color_cache.get(w['color_key'], lp.get('color', [0., 0., 1., 1.]))
        lp['alpha']            = w['al'].value()
        lp['marker']           = w['mk'].currentText()
        lp['markersize']       = w['ms'].value()
        lp['markerfacecolor']  = self._color_cache.get(w['mfc_key'], lp.get('markerfacecolor', [0., 0., 1., 1.]))
        lp['markeredgecolor']  = self._color_cache.get(w['mec_key'], lp.get('markeredgecolor', [0., 0., 0., 1.]))
        lp['markeredgewidth']  = w['mew'].value()
        if w.get('dx') is not None:
            lp['x_offset'] = w['dx'].value()
        if w.get('dy') is not None:
            lp['y_offset'] = w['dy'].value()
        if w.get('tg') is not None:
            lp['text_gap'] = w['tg'].value()

    def _switch_line(self, new_label: str):
        self._save_current_line()
        if new_label:
            self._build_line_widgets(new_label)

    # ── groups (collections) tab ─────────────────────────────────────────────

    def _make_collections_tab(self):
        w    = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(w)

        colls = self._params.get('collections', [])
        keys  = [e['_key'] for e in colls]

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Group:"))
        self._coll_selector = QtWidgets.QComboBox()
        self._coll_selector.addItems(keys)
        row.addWidget(self._coll_selector, 1)
        vbox.addLayout(row)

        self._coll_prop_grp  = QtWidgets.QGroupBox("Properties")
        self._coll_prop_form = QtWidgets.QFormLayout(self._coll_prop_grp)
        vbox.addWidget(self._coll_prop_grp)
        vbox.addStretch()

        if colls:
            self._build_coll_widgets(colls[0]['_key'])
        self._coll_selector.currentTextChanged.connect(self._switch_coll)
        return w

    def _build_coll_widgets(self, key: str):
        if not key:
            return
        form = self._coll_prop_form
        while form.rowCount():
            form.removeRow(0)

        colls = self._params.get('collections', [])
        entry = next((e for e in colls if e['_key'] == key), {})

        vis = QtWidgets.QCheckBox("Visible")
        vis.setChecked(bool(entry.get('visible', True)))
        form.addRow("", vis)

        fc = self._color_btn(f'_coll_{key}_fc', entry.get('facecolor', [0.5, 0.5, 0.5, 1.]))
        form.addRow("Fill Color", fc)

        ec = self._color_btn(f'_coll_{key}_ec', entry.get('edgecolor', [0., 0., 0., 0.]))
        form.addRow("Edge Color", ec)

        sz = QtWidgets.QDoubleSpinBox(); sz.setRange(0, 2000); sz.setSingleStep(5.0)
        sz.setValue(float(entry.get('size', 36.0)))

        mk = QtWidgets.QComboBox()
        mk.addItems(self._MARKERS)
        mk.setCurrentText(str(entry.get('marker', 'o')))

        al = QtWidgets.QDoubleSpinBox(); al.setRange(0.0, 1.0); al.setSingleStep(0.05); al.setDecimals(2)
        al.setValue(float(entry.get('alpha', 1.0)))
        form.addRow("Alpha", al)

        ew = QtWidgets.QDoubleSpinBox(); ew.setRange(0.0, 10.0); ew.setSingleStep(0.25); ew.setDecimals(2)
        ew.setValue(float(entry.get('edgewidth', 0.0)))
        form.addRow("Edge Width", ew)

        _is_poly = entry.get('_is_poly', False)
        if not _is_poly:
            form.addRow("Size (pts²)", sz)
            form.addRow("Marker", mk)

        self._coll_widgets[key] = {
            'vis': vis,
            'fc_key': f'_coll_{key}_fc',
            'ec_key': f'_coll_{key}_ec',
            'size': sz,
            'alpha': al,
            'marker': mk,
            'edgewidth': ew,
            '_is_poly': _is_poly,
        }
        self._current_coll_key = key

    def _save_current_coll(self):
        key = self._current_coll_key
        if not key or key not in self._coll_widgets:
            return
        w     = self._coll_widgets[key]
        colls = self._params.get('collections', [])
        entry = next((e for e in colls if e['_key'] == key), None)
        if entry is None:
            return
        entry['visible']   = w['vis'].isChecked()
        entry['facecolor'] = self._color_cache.get(w['fc_key'], entry.get('facecolor', [0.5, 0.5, 0.5, 1.]))
        entry['edgecolor'] = self._color_cache.get(w['ec_key'], entry.get('edgecolor', [0., 0., 0., 0.]))
        entry['alpha']     = w['alpha'].value()
        entry['edgewidth'] = w['edgewidth'].value()
        if not w.get('_is_poly', False):
            entry['size']   = w['size'].value()
            entry['marker'] = w['marker'].currentText()

    def _switch_coll(self, new_key: str):
        self._save_current_coll()
        if new_key:
            self._build_coll_widgets(new_key)

    # ── patches tab (barplot / histplot / boxplot) ────────────────────────────

    def _make_patches_tab(self):
        w    = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(w)

        patches = self._params.get('patches', [])
        keys    = [e['_key'] for e in patches]

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Group:"))
        self._patch_selector = QtWidgets.QComboBox()
        self._patch_selector.addItems(keys)
        row.addWidget(self._patch_selector, 1)
        vbox.addLayout(row)

        self._patch_prop_grp  = QtWidgets.QGroupBox("Properties")
        self._patch_prop_form = QtWidgets.QFormLayout(self._patch_prop_grp)
        vbox.addWidget(self._patch_prop_grp)
        vbox.addStretch()

        if patches:
            self._build_patch_widgets(patches[0]['_key'])
        self._patch_selector.currentTextChanged.connect(self._switch_patch)
        return w

    def _build_patch_widgets(self, key: str):
        if not key:
            return
        form = self._patch_prop_form
        while form.rowCount():
            form.removeRow(0)

        patches = self._params.get('patches', [])
        entry   = next((e for e in patches if e['_key'] == key), {})

        vis = QtWidgets.QCheckBox("Visible")
        vis.setChecked(bool(entry.get('visible', True)))
        form.addRow("", vis)

        fc = self._color_btn(f'_patch_{key}_fc', entry.get('facecolor', [0.5, 0.5, 0.5, 1.]))
        form.addRow("Fill Color", fc)

        ec = self._color_btn(f'_patch_{key}_ec', entry.get('edgecolor', [0., 0., 0., 1.]))
        form.addRow("Edge Color", ec)

        al = QtWidgets.QDoubleSpinBox(); al.setRange(0.0, 1.0); al.setSingleStep(0.05); al.setDecimals(2)
        al.setValue(float(entry.get('alpha', 1.0)))
        form.addRow("Alpha", al)

        ew = QtWidgets.QDoubleSpinBox(); ew.setRange(0.0, 10.0); ew.setSingleStep(0.25); ew.setDecimals(2)
        ew.setValue(float(entry.get('edgewidth', 0.0)))
        form.addRow("Edge Width", ew)

        self._patch_widgets[key] = {
            'vis':    vis,
            'fc_key': f'_patch_{key}_fc',
            'ec_key': f'_patch_{key}_ec',
            'alpha':  al,
            'edgewidth': ew,
        }
        self._current_patch_key = key

    def _save_current_patch(self):
        key = self._current_patch_key
        if not key or key not in self._patch_widgets:
            return
        w       = self._patch_widgets[key]
        patches = self._params.get('patches', [])
        entry   = next((e for e in patches if e['_key'] == key), None)
        if entry is None:
            return
        entry['visible']   = w['vis'].isChecked()
        entry['facecolor'] = self._color_cache.get(w['fc_key'], entry.get('facecolor', [0.5, 0.5, 0.5, 1.]))
        entry['edgecolor'] = self._color_cache.get(w['ec_key'], entry.get('edgecolor', [0., 0., 0., 1.]))
        entry['alpha']     = w['alpha'].value()
        entry['edgewidth'] = w['edgewidth'].value()

    def _switch_patch(self, new_key: str):
        self._save_current_patch()
        if new_key:
            self._build_patch_widgets(new_key)

    # ── annotations tab ──────────────────────────────────────────────────────

    @staticmethod
    def _ann_item_label(i: int, text: str) -> str:
        preview = (text or '').strip()[:28] or '(empty)'
        return f'{i + 1}: {preview}'

    def _make_annotations_tab(self):
        w    = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(w)

        top = QtWidgets.QHBoxLayout()
        self._ann_list = QtWidgets.QListWidget()
        self._ann_list.setMaximumHeight(120)
        top.addWidget(self._ann_list, 1)

        btn_col = QtWidgets.QVBoxLayout()
        add_btn = QtWidgets.QPushButton("+")
        add_btn.setFixedWidth(28)
        add_btn.setToolTip("Add annotation")
        add_btn.clicked.connect(self._on_ann_add)
        btn_col.addWidget(add_btn)
        rem_btn = QtWidgets.QPushButton("−")
        rem_btn.setFixedWidth(28)
        rem_btn.setToolTip("Remove selected annotation")
        rem_btn.clicked.connect(self._on_ann_remove)
        btn_col.addWidget(rem_btn)
        btn_col.addStretch()
        top.addLayout(btn_col)
        vbox.addLayout(top)

        self._ann_grp  = QtWidgets.QGroupBox("Properties")
        self._ann_form = QtWidgets.QFormLayout(self._ann_grp)
        self._ann_grp.setEnabled(False)
        vbox.addWidget(self._ann_grp)
        vbox.addStretch()

        for i, entry in enumerate(self._params.get('texts', [])):
            self._ann_list.addItem(self._ann_item_label(i, entry.get('text', '')))

        self._ann_list.currentRowChanged.connect(self._on_ann_row_changed)
        if self._params.get('texts'):
            self._ann_list.setCurrentRow(0)
        return w

    def _build_ann_widgets(self, idx: int):
        form = self._ann_form
        while form.rowCount():
            form.removeRow(0)
        texts = self._params.get('texts', [])
        if not (0 <= idx < len(texts)):
            return
        entry = texts[idx]

        te = QtWidgets.QLineEdit(str(entry.get('text', '')))
        form.addRow("Text", te);  self._ann_te = te

        xs = QtWidgets.QDoubleSpinBox()
        xs.setRange(-1e9, 1e9); xs.setDecimals(4); xs.setSingleStep(0.1)
        xs.setValue(float(entry.get('x', 0.5)))
        form.addRow("X", xs);  self._ann_xs = xs

        ys = QtWidgets.QDoubleSpinBox()
        ys.setRange(-1e9, 1e9); ys.setDecimals(4); ys.setSingleStep(0.1)
        ys.setValue(float(entry.get('y', 0.5)))
        form.addRow("Y", ys);  self._ann_ys = ys

        fs = QtWidgets.QDoubleSpinBox()
        fs.setRange(1, 72); fs.setSingleStep(0.5)
        fs.setValue(float(entry.get('fontsize', 12.0)))
        form.addRow("Font Size", fs);  self._ann_fs = fs

        color_key = f'_ann_{idx}_color'
        if color_key not in self._color_cache:
            self._color_cache[color_key] = list(entry.get('color', [0., 0., 0., 1.]))
        cb = self._color_btn(color_key, self._color_cache[color_key])
        form.addRow("Color", cb)
        self._ann_color_key = color_key

        ha = QtWidgets.QComboBox()
        ha.addItems(['left', 'center', 'right'])
        ha.setCurrentText(str(entry.get('ha', 'center')))
        form.addRow("H Align", ha);  self._ann_ha = ha

        va = QtWidgets.QComboBox()
        va.addItems(['top', 'center', 'bottom', 'baseline'])
        va.setCurrentText(str(entry.get('va', 'center')))
        form.addRow("V Align", va);  self._ann_va = va

        rot = QtWidgets.QDoubleSpinBox()
        rot.setRange(0, 360); rot.setSingleStep(5.0)
        rot.setValue(float(entry.get('rotation', 0.0)))
        form.addRow("Rotation", rot);  self._ann_rot = rot

        self._ann_idx = idx

    def _save_ann_fields(self):
        idx = self._ann_idx
        texts = self._params.get('texts', [])
        if not (0 <= idx < len(texts)):
            return
        entry = texts[idx]
        if not hasattr(self, '_ann_te'):
            return
        entry['text']     = self._ann_te.text()
        entry['x']        = self._ann_xs.value()
        entry['y']        = self._ann_ys.value()
        entry['fontsize'] = self._ann_fs.value()
        entry['color']    = self._color_cache.get(self._ann_color_key, entry.get('color', [0., 0., 0., 1.]))
        entry['ha']       = self._ann_ha.currentText()
        entry['va']       = self._ann_va.currentText()
        entry['rotation'] = self._ann_rot.value()
        item = self._ann_list.item(idx)
        if item:
            item.setText(self._ann_item_label(idx, entry['text']))

    def _on_ann_row_changed(self, row: int):
        self._save_ann_fields()
        self._ann_grp.setEnabled(row >= 0)
        if row >= 0:
            self._build_ann_widgets(row)

    def _on_ann_add(self):
        texts = self._params.setdefault('texts', [])
        idx = len(texts)
        texts.append({
            'text': '', 'x': 0.5, 'y': 0.5, 'fontsize': 12.0,
            'color': [0., 0., 0., 1.], 'ha': 'center', 'va': 'center', 'rotation': 0.0,
        })
        self._ann_list.addItem(self._ann_item_label(idx, ''))
        self._ann_list.setCurrentRow(idx)

    def _on_ann_remove(self):
        idx = self._ann_list.currentRow()
        if idx < 0:
            return
        self._save_ann_fields()
        texts = self._params.get('texts', [])
        if 0 <= idx < len(texts):
            texts.pop(idx)
        self._ann_list.takeItem(idx)
        # Refresh remaining labels (indices shifted)
        for i in range(self._ann_list.count()):
            entry = texts[i] if i < len(texts) else {}
            item = self._ann_list.item(i)
            if item:
                item.setText(self._ann_item_label(i, entry.get('text', '')))
        # Also re-key color cache entries so indices stay in sync
        for old_i in range(idx, len(texts) + 1):
            old_key = f'_ann_{old_i + 1}_color'
            new_key = f'_ann_{old_i}_color'
            if old_key in self._color_cache:
                self._color_cache[new_key] = self._color_cache.pop(old_key)
        self._ann_idx = -1
        new_row = min(idx, self._ann_list.count() - 1)
        if new_row >= 0:
            self._ann_list.setCurrentRow(new_row)
        else:
            self._ann_grp.setEnabled(False)

    # ── collect & accept ─────────────────────────────────────────────────────

    def _collect_params(self):
        """Harvest all widget values into self._params (without closing the dialog)."""
        # Text
        for key in ('title', 'xaxis', 'yaxis'):
            p = self._params.setdefault(key, {})
            p['text']       = getattr(self, f'_{key}_text').text()
            p['fontsize']   = getattr(self, f'_{key}_size').value()
            p['fontweight'] = getattr(self, f'_{key}_weight').currentText()
            p['color']      = self._color_cache.get(f'{key}_color', p.get('color', [0., 0., 0., 1.]))
            if key in ('xaxis', 'yaxis') and hasattr(self, f'_{key}_labelpad'):
                p['labelpad'] = getattr(self, f'_{key}_labelpad').value()
            if key == 'title' and hasattr(self, '_title_pad'):
                p['pad'] = self._title_pad.value()

    def _collect_and_accept(self):
        self._collect_and_accept_inner()
        self.accept()

    def _do_apply(self):
        """Apply button: save current state and call on_apply without closing."""
        # Collect everything exactly as _collect_and_accept does, minus accept()
        self._collect_and_accept_inner()
        if self._on_apply is not None:
            self._on_apply(copy.deepcopy(self._params))

    def _collect_and_accept_inner(self):
        """Shared collection logic used by both OK and Apply."""
        for key in ('title', 'xaxis', 'yaxis'):
            p = self._params.setdefault(key, {})
            p['text']       = getattr(self, f'_{key}_text').text()
            p['fontsize']   = getattr(self, f'_{key}_size').value()
            p['fontweight'] = getattr(self, f'_{key}_weight').currentText()
            p['color']      = self._color_cache.get(f'{key}_color', p.get('color', [0., 0., 0., 1.]))
            if key in ('xaxis', 'yaxis') and hasattr(self, f'_{key}_labelpad'):
                p['labelpad'] = getattr(self, f'_{key}_labelpad').value()
            if key == 'title' and hasattr(self, '_title_pad'):
                p['pad'] = self._title_pad.value()
        self._params['limit'] = {
            'x_lim': [self._xlim_lo.value(), self._xlim_hi.value()],
            'y_lim': [self._ylim_lo.value(), self._ylim_hi.value()],
        }
        for axis_key in ('xtick', 'ytick'):
            p = self._params.setdefault(axis_key, {})
            p['labelsize']     = getattr(self, f'_{axis_key}_size').value()
            p['labelrotation'] = getattr(self, f'_{axis_key}_rot').value()
            p['direction']     = getattr(self, f'_{axis_key}_dir').currentText()
            p['major_visible'] = getattr(self, f'_{axis_key}_major_vis').isChecked()
            p['minor_visible'] = getattr(self, f'_{axis_key}_minor_vis').isChecked()
            p['minor_length']  = getattr(self, f'_{axis_key}_minor_len').value()
        for grid_key in ('xgrid', 'ygrid'):
            p = self._params.setdefault(grid_key, {})
            p['visible']   = getattr(self, f'_{grid_key}_vis').isChecked()
            p['linestyle'] = getattr(self, f'_{grid_key}_ls').currentText()
            p['linewidth'] = getattr(self, f'_{grid_key}_lw').value()
            p['color']     = self._color_cache.get(f'{grid_key}_color', p.get('color', [0.5, 0.5, 0.5, 1.]))
        self._params['figsize'] = [self._fig_w.value(), self._fig_h.value()]
        self._params['background'] = {
            'axes_bg': self._color_cache.get('axes_bg', [1., 1., 1., 1.]),
            'fig_bg':  self._color_cache.get('fig_bg',  [1., 1., 1., 1.]),
        }
        self._params['padding'] = {
            side: getattr(self, f'_pad_{side}').value()
            for side in ('left', 'right', 'top', 'bottom')
        }
        for pos in ('left', 'right', 'top', 'bottom'):
            sp = self._params.setdefault('spines', {}).setdefault(pos, {})
            sp['visible']   = getattr(self, f'_spine_{pos}_vis').isChecked()
            sp['linewidth'] = getattr(self, f'_spine_{pos}_lw').value()
        self._params['font_family'] = self._font_family_cb.currentText()
        self._params['dpi']         = self._dpi_sb.value()
        if self._params.get('legend') is not None and hasattr(self, '_legend_vis'):
            p = self._params['legend']
            p['visible']        = self._legend_vis.isChecked()
            p['frameon']        = self._legend_frameon.isChecked()
            p['loc']            = self._legend_loc.currentText()
            p['fontsize']       = self._legend_fontsize.value()
            p['title']          = self._legend_title.text()
            p['title_fontsize'] = self._legend_title_fs.value()
            p['labelcolor']     = self._color_cache.get('legend_label_color', p.get('labelcolor', [0.0, 0.0, 0.0, 1.0]))
            p['ncols']          = self._legend_ncols.value()
            p['markerscale']    = self._legend_markerscale.value()
            p['labelspacing']   = self._legend_labelspacing.value()
            p['handlelength']   = self._legend_handlelength.value()
            p['handletextpad']  = self._legend_handletextpad.value()
            p['columnspacing']  = self._legend_columnspacing.value()
            p['borderpad']      = self._legend_borderpad.value()
            p['borderaxespad']  = self._legend_borderaxespad.value()
            p['framealpha']     = self._legend_framealpha.value()
            p['facecolor']      = self._color_cache.get('legend_face_color', p.get('facecolor', [1.0, 1.0, 1.0, 1.0]))
            p['edgecolor']      = self._color_cache.get('legend_edge_color', p.get('edgecolor', [0.0, 0.0, 0.0, 1.0]))
            if hasattr(self, '_legend_order_list') and self._legend_order_list:
                p['labels'] = [self._legend_order_list.item(i).text() for i in range(self._legend_order_list.count())]
        if self._params.get('lines'):
            self._save_current_line()
        if self._params.get('collections'):
            self._save_current_coll()
        if self._params.get('patches'):
            self._save_current_patch()
        self._save_ann_fields()

    def get_params(self) -> dict:
        return self._params


class DoubleVarPlotNode(BaseExecutionNode):
    """
    Generates a 2D seaborn plot from a data table.

    Plot types:
    - *scatter* — X vs Y scatter plot
    - *box* — box-and-whisker plot
    - *violin* — violin density plot
    - *pairplot* — all-pairs scatter matrix

    Columns:
    - **x_col** — column for the X axis
    - **y_col** — column for the Y axis
    - **hue** — optional column for colour grouping

    Keywords: scatter, boxplot, violin, pairplot, seaborn, 繪圖, 圖表, 散點圖, 小提琴圖, 盒鬚圖
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME = 'Double-Variable Plot'
    PORT_SPEC = {'inputs': ['table'], 'outputs': ['figure']}

    def __init__(self):
        super(DoubleVarPlotNode, self).__init__()
        self.add_input('data', multi_input=True, color=PORT_COLORS['table'])
        self.add_output('plot', multi_output=True, color=PORT_COLORS['figure'])

        items = ['scatter', 'box', 'violin', 'pairplot']
        self.add_combo_menu('plot_type', 'Plot Type', items=items)
        self._add_column_selector('x_col', 'X Column', text='', mode='single')
        self._add_column_selector('y_col', 'Y Column', text='', mode='single')
        self._add_column_selector('hue', 'Hue Column (Optional)', text='', mode='single')

    def evaluate(self):
        self.reset_progress()
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import seaborn as sns
        
        in_values = []
        in_port = self.inputs().get('data')
        if in_port and in_port.connected_ports():
            for connected in in_port.connected_ports():
                upstream_node = connected.node()
                up_val = upstream_node.output_values.get(connected.name(), None)
                if isinstance(up_val, TableData):
                    up_val = up_val.df
                elif hasattr(up_val, 'payload'):
                    up_val = up_val.payload
                in_values.append(up_val)

        if not in_values or in_values[0] is None:
            self.mark_error()
            return False, "No input data"

        df = in_values[0]
        if not isinstance(df, pd.DataFrame):
            self.mark_error()
            return False, "Input must be a pandas DataFrame"

        self._refresh_column_selectors(df, 'x_col', 'y_col', 'hue')

        try:
            self.set_progress(10)
            plot_type = self.get_property('plot_type')
            x_col = self.get_property('x_col') or None
            y_col = self.get_property('y_col') or None
            hue = self.get_property('hue') or None
            
            if plot_type == 'scatter' and (not x_col or not y_col):
                self.reset_progress()
                return False, "Scatter plot requires both X and Y columns."
            
            self.set_progress(30)
            sns.set_theme(style="darkgrid")
            if plot_type == 'pairplot':
                g = sns.pairplot(data=df, hue=hue)
                fig = g.fig
            else:
                fig = Figure(figsize=(10, 8))
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot(111)
                
                if plot_type == 'scatter':
                    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
                elif plot_type == 'box':
                    sns.boxplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
                elif plot_type == 'violin':
                    sns.violinplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
                
                self.set_progress(80)
                fig.tight_layout()
                
            self.output_values['plot'] = FigureData(payload=fig)
            self.mark_clean()
            self.set_progress(100)
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)


class PlotToolboxMixin:
    """
    Mixin that provides shared toolbox infrastructure for plot nodes.

    Inherit before BaseExecutionNode:
    `class MyNode(PlotToolboxMixin, BaseExecutionNode)`.
    Call `self._build_toolbox(height)` in `__init__` after creating all
    properties.
    """

    def _build_toolbox(self, height=320):
        """Create the NodeToolBoxWidget and register it. Call after properties are created."""
        self._toolbox_widgets = {}
        self.toolbox = NodeToolBoxWidget(self.view, name='plot_toolbox', label='Plot Settings')
        self.toolbox._toolbox.setFixedHeight(height)
        self.toolbox._toolbox.setMinimumWidth(280)
        self.add_custom_widget(self.toolbox)

    def _tb_text(self, name, label, page, default=''):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(2)
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet("color: #aaa; font-size: 8pt; font-weight: bold;")
        current_val = self.get_property(name)
        if current_val is None:
            current_val = default
        edit = QtWidgets.QLineEdit(str(current_val))
        edit.setStyleSheet(
            "QLineEdit { background: #222; border: 1px solid #444; color: #eee;"
            " padding: 2px; border-radius: 2px; }"
            "QLineEdit:focus { border: 1px solid #2e7d32; }"
        )
        edit.editingFinished.connect(lambda n=name, e=edit: self.set_property(n, e.text()))
        layout.addWidget(lbl)
        layout.addWidget(edit)
        self.toolbox.add_widget_to_page(page, container)
        self._toolbox_widgets[name] = edit

    def _tb_column_selector(self, name, label, page, default=''):
        """Text input + dropdown button for column selection inside toolbox."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(2)
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet("color: #aaa; font-size: 8pt; font-weight: bold;")
        current_val = self.get_property(name)
        if current_val is None:
            current_val = default

        row = QtWidgets.QWidget()
        rl = QtWidgets.QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)

        edit = QtWidgets.QLineEdit(str(current_val))
        edit.setStyleSheet(
            "QLineEdit { background: #222; border: 1px solid #444; color: #eee;"
            " padding: 2px; border-radius: 2px; }"
            "QLineEdit:focus { border: 1px solid #2e7d32; }"
        )
        edit.editingFinished.connect(
            lambda n=name, e=edit: self.set_property(n, e.text()))

        btn = QtWidgets.QToolButton()
        btn.setText('▼')
        btn.setFixedWidth(22)
        btn.setStyleSheet(
            "QToolButton { background: #333; border: 1px solid #444; color: #ccc;"
            " border-radius: 2px; }"
            "QToolButton:hover { background: #444; }")
        btn._columns = []

        def _show_menu(_checked=False, b=btn, e=edit, n=name):
            menu = QtWidgets.QMenu()
            for col in b._columns:
                action = menu.addAction(str(col))
                action.triggered.connect(
                    lambda _c=False, c=col, ed=e, nm=n: (
                        ed.setText(str(c)), self.set_property(nm, str(c))))
            if not b._columns:
                a = menu.addAction('(run graph first)')
                a.setEnabled(False)
            menu.exec(QtGui.QCursor.pos())

        btn.clicked.connect(_show_menu)
        rl.addWidget(edit, 1)
        rl.addWidget(btn)
        layout.addWidget(lbl)
        layout.addWidget(row)
        self.toolbox.add_widget_to_page(page, container)
        self._toolbox_widgets[name] = edit
        if not hasattr(self, '_tb_col_buttons'):
            self._tb_col_buttons = {}
        self._tb_col_buttons[name] = btn

    def _tb_refresh_columns(self, df, *prop_names):
        """Update toolbox column selector dropdowns with DataFrame columns."""
        if df is None or not hasattr(self, '_tb_col_buttons'):
            return
        columns = list(df.columns)
        for name in prop_names:
            btn = self._tb_col_buttons.get(name)
            if btn:
                btn._columns = columns

    def _tb_checkbox(self, name, label, page, default=True):
        current_val = self.get_property(name)
        if current_val is None:
            current_val = default
        cb = QtWidgets.QCheckBox(label)
        cb.setChecked(bool(current_val))
        cb.setStyleSheet("color: #eee; font-size: 9pt;")
        cb.stateChanged.connect(lambda s, n=name: self.set_property(n, bool(s)))
        self.toolbox.add_widget_to_page(page, cb)
        self._toolbox_widgets[name] = cb

    def _tb_spinbox(self, name, label, page, default,
                    min_val=0, max_val=999999, step=1.0, decimals=3):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(2)
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet("color: #aaa; font-size: 8pt; font-weight: bold;")
        current_val = self.get_property(name)
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(min_val, max_val)
        sb.setDecimals(decimals)
        sb.setValue(float(current_val) if current_val is not None else float(default))
        sb.setSingleStep(step)
        sb.setStyleSheet(
            "QDoubleSpinBox { background: #222; border: 1px solid #444; color: #eee;"
            " padding: 2px; border-radius: 2px; }"
            "QDoubleSpinBox:focus { border: 1px solid #2e7d32; }"
        )
        sb.valueChanged.connect(lambda val, n=name: self.set_property(n, val))
        layout.addWidget(lbl)
        layout.addWidget(sb)
        self.toolbox.add_widget_to_page(page, container)
        self._toolbox_widgets[name] = sb

    def _tb_color(self, name, label, page):
        current_val = self.get_property(name)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 5, 0, 5)
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet("color: #aaa; font-size: 8pt; font-weight: bold;")
        btn = ColorPickerButtonWidget()
        if current_val:
            btn.set_value(current_val)
        btn.value_changed.connect(lambda val, n=name: self.set_property(n, val))
        layout.addWidget(lbl)
        layout.addWidget(btn)
        self.toolbox.add_widget_to_page(page, container)
        self._toolbox_widgets[name] = btn

    def _tb_combo(self, name, label, page, items):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(2)
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet("color: #aaa; font-size: 8pt; font-weight: bold;")
        combo = QtWidgets.QComboBox()
        combo.addItems(items)
        combo.setStyleSheet(
            "QComboBox { background: #222; border: 1px solid #444; color: #eee; padding: 2px; }"
            "QComboBox::drop-down { border: 0px; }"
        )
        current_val = self.get_property(name)
        if current_val and str(current_val) in items:
            combo.setCurrentText(str(current_val))
        combo.currentTextChanged.connect(lambda val, n=name: self.set_property(n, val))
        layout.addWidget(lbl)
        layout.addWidget(combo)
        self.toolbox.add_widget_to_page(page, container)
        self._toolbox_widgets[name] = combo

    @staticmethod
    def _parse_csv_order(value):
        return [x.strip() for x in str(value or '').split(',') if x.strip()]

    def _read_connected_table_df(self, input_name='data'):
        """Best-effort DataFrame fetch from a connected input port."""
        try:
            port = self.inputs().get(input_name)
            if not port or not port.connected_ports():
                return None
            cp = port.connected_ports()[0]
            data = cp.node().output_values.get(cp.name(), None)
            if isinstance(data, TableData):
                return data.df
            if isinstance(data, pd.DataFrame):
                return data
        except Exception:
            pass
        return None

    def _infer_order_candidates(self, order_prop_name):
        """
        Infer likely x-axis categories from connected data for order widgets.
        Returns a list of strings.
        """
        df = self._read_connected_table_df('data')
        if df is None or getattr(df, 'empty', True):
            return []
        try:
            if order_prop_name == 'x_axis_order':
                # Swarm logic: explicit group column, fallback to semantic names,
                # otherwise use numeric column names (wide-to-long behavior).
                group_col = str(self.get_property('group_col') or '').strip()
                if group_col and group_col in df.columns:
                    vals = df[group_col].dropna().astype(str).str.strip()
                    return [v for v in vals.unique().tolist() if v]
                for col in df.columns:
                    if str(col).lower() in ['group', 'class', 'treatment']:
                        vals = df[col].dropna().astype(str).str.strip()
                        return [v for v in vals.unique().tolist() if v]
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                return [str(c).strip() for c in num_cols if str(c).strip()]

            if order_prop_name == 'order':
                x_col = str(self.get_property('x_col') or '').strip()
                if x_col and x_col in df.columns:
                    vals = df[x_col].dropna().astype(str).str.strip()
                    return [v for v in vals.unique().tolist() if v]
                # Fallback: first non-numeric column, then first column.
                non_num = [c for c in df.columns if c not in df.select_dtypes(include=[np.number]).columns]
                col = non_num[0] if non_num else (df.columns[0] if len(df.columns) else None)
                if col is None:
                    return []
                vals = df[col].dropna().astype(str).str.strip()
                return [v for v in vals.unique().tolist() if v]
        except Exception:
            return []
        return []

    def _tb_order_list(self, name, label, page):
        """Drag-reorder list editor for comma-separated order properties."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(4)

        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet("color: #aaa; font-size: 8pt; font-weight: bold;")
        layout.addWidget(lbl)

        lst = QtWidgets.QListWidget()
        lst.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        lst.setDragEnabled(True)
        lst.setAcceptDrops(True)
        lst.viewport().setAcceptDrops(True)
        lst.setDropIndicatorShown(True)
        lst.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        lst.setDragDropOverwriteMode(False)
        lst.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        lst.setMinimumHeight(90)
        existing = self._parse_csv_order(self.get_property(name))
        inferred = self._infer_order_candidates(name)
        seed = existing + [v for v in inferred if v not in existing]
        lst.addItems(seed)
        for i in range(lst.count()):
            it = lst.item(i)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsDragEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled)
        layout.addWidget(lst)

        row1 = QtWidgets.QHBoxLayout()
        inp = QtWidgets.QLineEdit()
        inp.setPlaceholderText("Add item and press Enter")
        load_btn = QtWidgets.QPushButton("Load from Data")
        up_btn = QtWidgets.QPushButton("Up")
        down_btn = QtWidgets.QPushButton("Down")
        add_btn = QtWidgets.QPushButton("Add")
        rm_btn = QtWidgets.QPushButton("Remove")
        clear_btn = QtWidgets.QPushButton("Clear")
        row1.addWidget(inp, 1)
        row1.addWidget(add_btn)
        row1.addWidget(rm_btn)
        row1.addWidget(clear_btn)
        layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(load_btn)
        row2.addWidget(up_btn)
        row2.addWidget(down_btn)
        row2.addStretch(1)
        layout.addLayout(row2)

        def _push_to_model():
            vals = [lst.item(i).text().strip() for i in range(lst.count()) if lst.item(i).text().strip()]
            self.set_property(name, ','.join(vals))

        def _add_item():
            txt = inp.text().strip()
            if not txt:
                return
            exists = {lst.item(i).text().strip() for i in range(lst.count())}
            if txt in exists:
                inp.clear()
                return
            item = QtWidgets.QListWidgetItem(txt)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsDragEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled)
            lst.addItem(item)
            inp.clear()
            _push_to_model()

        def _remove_selected():
            for it in lst.selectedItems():
                lst.takeItem(lst.row(it))
            _push_to_model()

        def _clear_all():
            lst.clear()
            _push_to_model()

        def _move_selected(delta):
            selected_rows = sorted({lst.row(it) for it in lst.selectedItems()})
            if not selected_rows:
                return
            if delta < 0 and selected_rows[0] == 0:
                return
            if delta > 0 and selected_rows[-1] == lst.count() - 1:
                return

            rows = selected_rows if delta < 0 else list(reversed(selected_rows))
            for r in rows:
                it = lst.takeItem(r)
                lst.insertItem(r + delta, it)
                it.setSelected(True)
            _push_to_model()

        def _load_from_data():
            vals = self._infer_order_candidates(name)
            if not vals:
                return
            cur = [lst.item(i).text().strip() for i in range(lst.count()) if lst.item(i).text().strip()]
            merged = cur + [v for v in vals if v not in cur]
            lst.clear()
            for v in merged:
                item = QtWidgets.QListWidgetItem(v)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsDragEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled)
                lst.addItem(item)
            _push_to_model()

        inp.returnPressed.connect(_add_item)
        load_btn.clicked.connect(_load_from_data)
        up_btn.clicked.connect(lambda: _move_selected(-1))
        down_btn.clicked.connect(lambda: _move_selected(1))
        add_btn.clicked.connect(_add_item)
        rm_btn.clicked.connect(_remove_selected)
        clear_btn.clicked.connect(_clear_all)
        lst.model().rowsMoved.connect(lambda *args: _push_to_model())

        # If we discovered data categories not present in the stored property,
        # commit the merged list once so saved state matches visible order items.
        if seed != existing:
            self.set_property(name, ','.join(seed), push_undo=False)

        self.toolbox.add_widget_to_page(page, container)
        self._toolbox_widgets[name] = lst

    def _tb_add_stats_page(self, page='Stats Annotations'):
        """Add standard stat-annotation controls to a toolbox page."""
        self._tb_combo(
            'stat_label_mode',
            'Annotation Text',
            page,
            ['Stars (*, **, ***)', 'P-value (scientific)']
        )
        self._tb_checkbox('stat_show_ns',   'Show "ns" Annotations',     page, False)
        self._tb_spinbox('stat_y_offset',   'Y Offset (Fraction of Max)', page, 0.05,  0, 5,   0.01, 3)
        self._tb_spinbox('stat_line_width', 'Line Width',                 page, 1.5,   0, 10,  0.1,  1)
        self._tb_spinbox('stat_text_size',  'Text Font Size',             page, 12.0,  4, 40,  1.0,  0)
        self._tb_color('stat_line_color',   'Line Color',                 page)
        self._tb_color('stat_text_color',   'Text Color',                 page)

    def _tb_add_figure_page(self, page='Figure & Layout'):
        """Add standard figure size / tick-rotation controls to a toolbox page."""
        self._tb_spinbox('fig_width',     'Fig Width',              page, 8.0, 1, 40, 0.5, 1)
        self._tb_spinbox('fig_height',    'Fig Height',             page, 6.0, 1, 40, 0.5, 1)
        self._tb_spinbox('tick_rotation', 'X-Tick Rotation (Deg)', page, 0.0, 0, 180, 1.0, 0)

    # ── sync helpers ──────────────────────────────────────────────────────────

    def _sync_toolbox_from_model(self):
        """Sync all toolbox widgets from current model values (call after session load)."""
        if not hasattr(self, '_toolbox_widgets'):
            return
        for name, widget in self._toolbox_widgets.items():
            val = self.get_property(name)
            if val is None:
                continue
            try:
                widget.blockSignals(True)
                if isinstance(widget, QtWidgets.QLineEdit):
                    if widget.text() != str(val):
                        widget.setText(str(val))
                elif isinstance(widget, QtWidgets.QCheckBox):
                    if widget.isChecked() != bool(val):
                        widget.setChecked(bool(val))
                elif isinstance(widget, QtWidgets.QComboBox):
                    if widget.currentText() != str(val):
                        widget.setCurrentText(str(val))
                elif isinstance(widget, ColorPickerButtonWidget):
                    widget.set_value(val)
                elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                    try:
                        if abs(widget.value() - float(val)) > 1e-9:
                            widget.setValue(float(val))
                    except (TypeError, ValueError):
                        pass
                elif isinstance(widget, QtWidgets.QListWidget):
                    want = self._parse_csv_order(val)
                    cur  = [widget.item(i).text() for i in range(widget.count())]
                    if cur != want:
                        widget.clear()
                        widget.addItems(want)
                        for i in range(widget.count()):
                            it = widget.item(i)
                            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsDragEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled)
                widget.blockSignals(False)
            except Exception:
                pass

    def set_property(self, name, value, push_undo=True):
        """Override to sync toolbox widgets when a property changes."""
        cur_val = self.get_property(name)
        if isinstance(cur_val, (list, tuple)) and isinstance(value, (list, tuple)):
            if list(cur_val) == list(value):
                return
        elif cur_val == value:
            return
        super().set_property(name, value, push_undo)
        if not hasattr(self, '_toolbox_widgets') or name not in self._toolbox_widgets:
            return
        widget = self._toolbox_widgets[name]
        try:
            widget.blockSignals(True)
            if isinstance(widget, QtWidgets.QLineEdit):
                if widget.text() != str(value):
                    widget.setText(str(value))
            elif isinstance(widget, QtWidgets.QCheckBox):
                if widget.isChecked() != bool(value):
                    widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QComboBox):
                if widget.currentText() != str(value):
                    widget.setCurrentText(str(value))
            elif isinstance(widget, ColorPickerButtonWidget):
                widget.set_value(value)
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                try:
                    if abs(widget.value() - float(value)) > 1e-9:
                        widget.setValue(float(value))
                except (TypeError, ValueError):
                    pass
            elif isinstance(widget, QtWidgets.QListWidget):
                want = self._parse_csv_order(value)
                cur  = [widget.item(i).text() for i in range(widget.count())]
                if cur != want:
                    widget.clear()
                    widget.addItems(want)
                    for i in range(widget.count()):
                        it = widget.item(i)
                        it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsDragEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled)
            widget.blockSignals(False)
        except Exception:
            pass

    def update(self):
        """Sync toolbox widgets after node is added or a session is loaded."""
        super().update()
        self._sync_toolbox_from_model()

    def update_model(self):
        """Push toolbox widget values into the model before save."""
        super().update_model()
        if not hasattr(self, '_toolbox_widgets'):
            return
        for name, widget in self._toolbox_widgets.items():
            try:
                if isinstance(widget, QtWidgets.QLineEdit):
                    self.set_property(name, widget.text(), push_undo=False)
                elif isinstance(widget, QtWidgets.QCheckBox):
                    self.set_property(name, widget.isChecked(), push_undo=False)
                elif isinstance(widget, QtWidgets.QComboBox):
                    self.set_property(name, widget.currentText(), push_undo=False)
                elif isinstance(widget, ColorPickerButtonWidget):
                    self.set_property(name, widget.get_value(), push_undo=False)
                elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                    self.set_property(name, widget.value(), push_undo=False)
                elif isinstance(widget, QtWidgets.QListWidget):
                    vals = [widget.item(i).text().strip() for i in range(widget.count()) if widget.item(i).text().strip()]
                    self.set_property(name, ','.join(vals), push_undo=False)
            except Exception as e:
                print(f"[PlotToolboxMixin] update_model error for '{name}': {e}")


class SwarmPlotNode(PlotToolboxMixin, BaseExecutionNode):
    """
    Creates a swarm plot with optional statistical annotation overlay.

    Accepts a data table and an optional stats table (from
    PairwiseComparisonNode) for significance-bracket overlays.

    Columns:
    - **target_column** — numeric column for the Y axis
    - **group_col** — categorical column that defines groups on the X axis
    - **x_axis_order** — comma-separated group order
    - **control_group** — reference group for fold-change ratios

    Options:
    - *use_stripplot* — switch from beeswarm to jittered strip layout
    - *show_error_bars* — overlay mean with SE/SD/CI/PI error bars
    - *enable_subgroups* — split group labels by delimiter for sub-bracket display

    Keywords: swarmplot, jitter points, significance brackets, group comparison, stats overlay, 繪圖, 群組比較, 顯著性標記, 分組, 散點
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME = 'Swarm Plot + Stats'
    PORT_SPEC = {'inputs': ['table', 'table'], 'outputs': ['figure']}
    
    def __init__(self):
        # Initialize members BEFORE super().__init__ because super calls set_property
        self._toolbox_widgets = {}
        super(SwarmPlotNode, self).__init__()
        self.add_input('data', color=PORT_COLORS['table'])
        self.add_input('stats', color=PORT_COLORS['stat'])
        self.add_output('plot', color=PORT_COLORS['figure'])
        
        # Create properties first (so they exist in the model)
        self.create_property('x_axis_order', '', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('control_group', '', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('group_col', '', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('target_column', 'Value', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('y_label', 'Fold Change', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('x_label', 'Treatment Group', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('plot_title', '', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('dot_color', [105, 105, 105, 255], widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('dot_size', 5.0, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('dot_alpha', 0.8, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('y_min', '', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('y_max', '', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('enable_subgroups', True, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QCHECK_BOX.value, tab='Properties')
        self.create_property('fig_width', 10.0, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('fig_height', 8.0, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('use_stripplot', False, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QCHECK_BOX.value, tab='Properties')
        self.create_property('marker_cycle', 'o', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('edge_color_cycle', 'none', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('edge_width_cycle', '0.0', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('dot_palette', 'None', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QCOMBO_BOX.value, tab='Properties', items=['None', 'Set2', 'husl', 'viridis', 'colorblind'])
        self.create_property('show_error_bars', True, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QCHECK_BOX.value, tab='Properties')
        self.create_property('error_measure', 'se', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QCOMBO_BOX.value, tab='Properties', items=['se', 'sd', 'ci', 'pi'])
        self.create_property('error_value', 1.0, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QDOUBLESPIN_BOX.value, tab='Properties')
        self.create_property('error_capsize', 0.15, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QDOUBLESPIN_BOX.value, tab='Properties')
        self.create_property('error_markersize', 15.0, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QDOUBLESPIN_BOX.value, tab='Properties')
        self.create_property('error_linewidth', 1.5, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QDOUBLESPIN_BOX.value, tab='Properties')
        self.create_property('error_color', [255, 0, 0, 255], widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('title_fontsize', 12.0, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('label_fontsize', 10.0, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('tick_rotation', 0., widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('ratio_font_size', 8.0, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('ratio_text_xoffset', 0.25, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('ratio_text_yoffset', 0.0, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('ratio_text_color', [255, 0, 0, 255], widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('split_string', '/,_,-,|', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value, tab='Properties')
        self.create_property('subgroup_bracket_yoffset', -0.04, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('subgroup_text_yoffset', -0.05, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('subgroup_bracket_linewidth', 1.5, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('subgroup_text_fontsize', 11, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('stat_show_ns', False, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('stat_label_mode', 'Stars (*, **, ***)', widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('stat_line_color', [0, 0, 0, 255], widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('stat_line_width', 1.5, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('stat_text_color', [0, 0, 0, 255], widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('stat_text_size', 12.0, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('stat_y_offset', 0.05, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')
        self.create_property('group_spacing', 0.5, widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value, tab='Properties')

        # Create the toolbox widget
        self._build_toolbox(450)

        # Group widgets into pages
        self._tb_column_selector('group_col', 'Group Column', 'Data & Labels', '')
        self._tb_column_selector('target_column', 'Target Column', 'Data & Labels', 'Value')
        self._tb_order_list('x_axis_order', 'X-Axis Order', 'Data & Labels')
        self._tb_text('y_label', 'Y-Axis Label', 'Data & Labels', 'Fold Change')
        self._tb_text('x_label', 'X-Axis Label', 'Data & Labels', 'Treatment Group')
        self._tb_text('plot_title', 'Plot Title', 'Data & Labels')
        self._tb_spinbox('title_fontsize', 'Title Font Size', 'Data & Labels', 12.0, 0, 100, 0.5, 2)
        self._tb_spinbox('label_fontsize', 'Label Font Size', 'Data & Labels', 10.0, 0, 100, 0.5, 2)
        self._tb_spinbox('tick_rotation', 'X-Tick Rotation (Deg)', 'Data & Labels', 0.0, 0, 180, 1, 2)
        self._tb_spinbox('fig_width', 'Fig Width', 'Figure & Layout', 10.0, 0, 100, 0.5, 2)
        self._tb_spinbox('fig_height', 'Fig Height', 'Figure & Layout', 8.0, 0, 100, 0.5, 2)
        self._tb_spinbox('group_spacing', 'Group Spacing', 'Figure & Layout', 0.5, 0.1, 10.0, 0.05, 2)
        self._tb_combo('dot_palette', 'Color Palette (by Group)', 'Visuals', ['None', 'Set2', 'husl', 'viridis', 'colorblind', 'pastel', 'dark'])
        self._tb_color('dot_color', 'Dot Color (if No Palette)', 'Visuals')
        self._tb_spinbox('dot_size', 'Dot Size', 'Visuals', 5.0, 0, 100, 0.5, 2)
        self._tb_spinbox('dot_alpha', 'Dot Alpha (0-1)', 'Visuals', 0.8, 0, 1, 0.01, 2)
        self._tb_checkbox('use_stripplot', 'Use Stripplot Fallback', 'Visuals', False)
        self._tb_text('marker_cycle', 'Marker Cycle (e.g. o,s,^)', 'Visuals', 'o')
        self._tb_text('edge_color_cycle', 'Edge Color Cycle (e.g. black,none)', 'Visuals', 'none')
        self._tb_text('edge_width_cycle', 'Edge Width Cycle (e.g. 1.0,0.0)', 'Visuals', '0.0')
        self._tb_text('y_min', 'Y Min (Manual)', 'Figure & Layout', '')
        self._tb_text('y_max', 'Y Max (Manual)', 'Figure & Layout', '')
        self._tb_checkbox('show_error_bars', 'Show Mean / Error Bars', 'Error Bars', True)
        self._tb_color('error_color', 'Error Bar Color', 'Error Bars')
        self._tb_combo('error_measure', 'Error Measure', 'Error Bars', ['se', 'sd', 'ci', 'pi'])
        self._tb_spinbox('error_capsize', 'Error Cap Size', 'Error Bars', 0.15, 0, 100, 0.01, 2)
        self._tb_spinbox('error_markersize', 'Error Marker Size', 'Error Bars', 15.0, 0, 100, 0.1, 2)
        self._tb_spinbox('error_linewidth', 'Error Line Width', 'Error Bars', 1.5, 0, 100, 0.1, 2)
        self._tb_text('control_group', 'Ratio Control Group', 'Error Bars')
        self._tb_color('ratio_text_color', 'Ratio Text Color', 'Error Bars')
        self._tb_spinbox('ratio_font_size', 'Ratio Font Size', 'Error Bars', 8.0, 0, 100, 1, 2)
        self._tb_spinbox('ratio_text_xoffset', 'Ratio Text X Offset', 'Error Bars', 0.25, -100, 100, 0.25, 2)
        self._tb_spinbox('ratio_text_yoffset', 'Ratio Text Y Offset', 'Error Bars', 0.0, -100, 100, 0.25, 2)
        self._tb_checkbox('stat_show_ns', 'Show "ns" Annotations', 'Stats Annotations', False)
        self._tb_combo('stat_label_mode', 'Annotation Text', 'Stats Annotations', ['Stars (*, **, ***)', 'P-value (scientific)'])
        self._tb_color('stat_line_color', 'Line Color', 'Stats Annotations')
        self._tb_spinbox('stat_line_width', 'Line Width', 'Stats Annotations', 1.5, 0, 100, 0.1, 2)
        self._tb_color('stat_text_color', 'Text Color', 'Stats Annotations')
        self._tb_spinbox('stat_text_size', 'Text Font Size', 'Stats Annotations', 12.0, 1, 100, 0.5, 2)
        self._tb_spinbox('stat_y_offset', 'Y Offset (Fraction of Max)', 'Stats Annotations', 0.05, 0, 10, 0.005, 3)
        self._tb_text('split_string', 'Subgroup Split String (Sep1,Sep2)', 'Advanced', '/,_,-,|')
        self._tb_checkbox('enable_subgroups', 'Enable Subgroup Parsing', 'Advanced', True)
        self._tb_spinbox('subgroup_bracket_yoffset', 'Subgroup Bracket Y Offset', 'Advanced', -0.04, -100, 100, 0.005, 3)
        self._tb_spinbox('subgroup_bracket_linewidth', 'Subgroup Bracket Line Width', 'Advanced', 1.5, 0, 100, 0.5, 2)
        self._tb_spinbox('subgroup_text_yoffset', 'Subgroup Text Y Offset', 'Advanced', -0.05, -100, 100, 0.005, 3)
        self._tb_spinbox('subgroup_text_fontsize', 'Subgroup Text Font Size', 'Advanced', 11, 0, 100, 1, 2)

    def evaluate(self):
        self.reset_progress()
        
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import seaborn as sns
        import re
        
        data_port = self.inputs().get('data')
        if not data_port or not data_port.connected_ports():
            self.mark_error()
            return False, "No data connection"
            
        data_up_node = data_port.connected_ports()[0].node()
        data_val = data_up_node.output_values.get(data_port.connected_ports()[0].name(), None)
        
        if isinstance(data_val, TableData):
            df = data_val.df.copy()
        elif isinstance(data_val, pd.DataFrame):
            df = data_val.copy()
        else:
            self.mark_error()
            return False, "Expected TableData or DataFrame for port 'data'"

        self._tb_refresh_columns(df, 'group_col', 'target_column')

        tukey_df = None
        stats_port = self.inputs().get('stats')
        if stats_port and stats_port.connected_ports():
            stats_up_node = stats_port.connected_ports()[0].node()
            stats_val = stats_up_node.output_values.get(stats_port.connected_ports()[0].name(), None)
            if isinstance(stats_val, StatData):
                tukey_df = stats_val.df.copy()
                err = _check_group_stat_df(tukey_df)
                if err:
                    self.mark_error(); return False, err
            else:
                self.mark_error()
                return False, "Expected StatData for port 'stats'"
                
        user_group_col = str(self.get_property('group_col') or '').strip()
        group_col = None
        if user_group_col and user_group_col in df.columns:
            group_col = user_group_col
        else:
            for col in df.columns:
                if str(col).lower() in ['group', 'class', 'treatment']:
                    group_col = col
                    break

        if not group_col:
            num_cols = df.select_dtypes(include=[np.number]).columns
            df_long = df.melt(value_vars=num_cols, var_name='Group', value_name='Value')
        else:
            target_col = str(self.get_property('target_column')).strip()
            num_cols = df.select_dtypes(include=[np.number]).columns
            val_col = [c for c in num_cols if c != group_col]
            if target_col and target_col in df.columns:
                target_to_use = target_col
            elif val_col:
                target_to_use = val_col[0]
            else:
                self.mark_error()
                return False, "No numerical values"
            df_long = df[[group_col, target_to_use]].rename(columns={group_col: 'Group', target_to_use: 'Value'})
        
        df_long = df_long.dropna(subset=['Value'])
        df_long['Group'] = df_long['Group'].astype(str).str.strip()
        
        order_str = str(self.get_property('x_axis_order')).strip()
        unique_groups = df_long['Group'].unique()
        
        if order_str:
            forced_order = [x.strip() for x in order_str.split(',')]
            for g in unique_groups:
                if g not in forced_order:
                    forced_order.append(g)
            plot_order = [x for x in forced_order if x in unique_groups]
        else:
            plot_order = list(unique_groups)
            
        separators = self.get_property('split_string').split(',')

        enable_subgroups = self.get_property('enable_subgroups')
        has_subgroups = False
        x_labels = []
        subgroup_spans = []
        current_subgroup = None
        start_idx = 0
        
        for i, g in enumerate(plot_order):
            matched_sep = None
            if enable_subgroups:
                for sep in separators:
                    if sep in g:
                        matched_sep = sep
                        break
            
            if matched_sep:
                has_subgroups = True
                sg, conc = g.split(matched_sep, 1)
                x_labels.append(conc)
                if sg != current_subgroup:
                    if current_subgroup is not None:
                        subgroup_spans.append((current_subgroup, start_idx, i - 1))
                    current_subgroup = sg
                    start_idx = i
            else:
                x_labels.append(g)
                if current_subgroup is not None:
                    subgroup_spans.append((current_subgroup, start_idx, i - 1))
                    current_subgroup = None
        
        if current_subgroup is not None:
            subgroup_spans.append((current_subgroup, start_idx, len(plot_order) - 1))
        
        def parse_color(color_val, default: str = 'black'):
            if isinstance(color_val, (list, tuple)) and len(color_val) >= 3:
                r, g, b = color_val[:3]
                a = color_val[3] if len(color_val) > 3 else 255
                return (r/255.0, g/255.0, b/255.0, a/255.0)
            else:
                return default
        
        try:
            fig_w = float(self.get_property('fig_width'))
            fig_h = float(self.get_property('fig_height'))
            
            fig = Figure(figsize=(fig_w, fig_h))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            
            mpl_color = parse_color(self.get_property('dot_color'), 'dimgray')
            
            try:
                import itertools
                import matplotlib.markers as mmarkers
                palette_opt = str(self.get_property('dot_palette') or 'None').strip()
                swarm_size = float(self.get_property('dot_size'))
                swarm_alpha = float(self.get_property('dot_alpha'))
                swarm_kws = {'size': swarm_size, 'alpha': swarm_alpha, 'zorder': 1}
                
                use_strip = bool(self.get_property('use_stripplot'))
                if use_strip:
                    swarm_kws['jitter'] = True
                
                if palette_opt and palette_opt.lower() != 'none':
                    _colls_before = set(id(c) for c in ax.collections)
                    if use_strip:
                        sns.stripplot(data=df_long, x='Group', y='Value', order=plot_order, hue='Group', palette=palette_opt, legend=False, ax=ax, **swarm_kws)
                    else:
                        sns.swarmplot(data=df_long, x='Group', y='Value', order=plot_order, hue='Group', palette=palette_opt, legend=False, ax=ax, **swarm_kws)
                    # Label new collections with real group names in plot_order
                    _new_colls = [c for c in ax.collections if id(c) not in _colls_before]
                    for _ci, _c in enumerate(_new_colls):
                        _c.set_label(plot_order[_ci] if _ci < len(plot_order) else f'Group {_ci}')
                else:
                    _colls_before = set(id(c) for c in ax.collections)
                    if use_strip:
                        sns.stripplot(data=df_long, x='Group', y='Value', order=plot_order, color=mpl_color, ax=ax, **swarm_kws)
                    else:
                        sns.swarmplot(data=df_long, x='Group', y='Value', order=plot_order, color=mpl_color, ax=ax, **swarm_kws)
                    # Single-color: no legend needed — groups are on the x-axis
                    _new_colls = [c for c in ax.collections if id(c) not in _colls_before]
                    for _c in _new_colls:
                        _c.set_label('_nolegend_')
                        
                # After seaborn plot, collections contains one PathCollection per hue or group (usually)
                m_cycle_str = str(self.get_property('marker_cycle')).strip()
                if not m_cycle_str: m_cycle_str = 'o'
                ec_cycle_str = str(self.get_property('edge_color_cycle')).strip()
                if not ec_cycle_str: ec_cycle_str = 'none'
                ew_cycle_str = str(self.get_property('edge_width_cycle')).strip()
                if not ew_cycle_str: ew_cycle_str = '0.0'
                
                m_cycle = [x.strip() for x in m_cycle_str.split(',')]
                ec_cycle = [x.strip() for x in ec_cycle_str.split(',')]
                ew_cycle = [float(x.strip()) for x in ew_cycle_str.split(',')]
                
                m_iter = itertools.cycle(m_cycle)
                ec_iter = itertools.cycle(ec_cycle)
                ew_iter = itertools.cycle(ew_cycle)
                
                for c in ax.collections:
                    marker = next(m_iter)
                    edge_color = next(ec_iter)
                    edge_width = next(ew_iter)
                    
                    if edge_color.lower() != 'none' and edge_width > 0:
                        c.set_edgecolor(edge_color)
                        c.set_linewidth(edge_width)
                        
                    if marker != 'o':
                        try:
                            m_obj = mmarkers.MarkerStyle(marker)
                            c.set_paths([m_obj.get_path()])
                        except Exception as marker_e:
                            print(f"Warning: Could not apply marker '{marker}': {marker_e}")
            except Exception as plot_e:
                print(f"Warning: Error applying plot dot style logic: {plot_e}")
            
            try:
                y_min_val = self.get_property('y_min')
                y_max_val = self.get_property('y_max')
                
                y_bottom, y_top = ax.get_ylim()
                if y_min_val is not None and str(y_min_val).strip() != '':
                    y_bottom = float(y_min_val)
                if y_max_val is not None and str(y_max_val).strip() != '':
                    y_top = float(y_max_val)
                
                ax.set_ylim(bottom=y_bottom, top=y_top)
            except Exception as e:
                print(f"Warning: Could not set Y limits: {e}")
            
            if self.get_property('show_error_bars'):
                err_mpl_color = parse_color(self.get_property('error_color'), 'red')
                
                err_measure = str(self.get_property('error_measure') or 'se').strip()
                try: err_val = float(self.get_property('error_value'))
                except (ValueError, TypeError): err_val = 1.0
                err_bar_arg = (err_measure, err_val)
                
                try: cap = float(self.get_property('error_capsize') or 0.15)
                except ValueError: cap = 0.15
                try: msize = float(self.get_property('error_markersize') or 15.0)
                except ValueError: msize = 15.0
                try: lwidth = float(self.get_property('error_linewidth') or 1.5)
                except ValueError: lwidth = 1.5
                
                _lines_before = set(id(l) for l in ax.get_lines())
                sns.pointplot(data=df_long, x='Group', y='Value', order=plot_order, 
                              estimator='mean', errorbar=err_bar_arg, 
                              markers='_', capsize=cap, color=err_mpl_color, 
                              err_kws={'linewidth': lwidth, 'zorder': 10}, linestyles='none', markersize=msize, ax=ax, zorder=10)
                # Hide error bar lines from legend
                _new_lines = [l for l in ax.get_lines() if id(l) not in _lines_before]
                for _line in _new_lines:
                    _line.set_label('_nolegend_')

            ctrl_group = str(self.get_property('control_group') or '').strip()
            if ctrl_group and ctrl_group in plot_order:
                group_means = df_long.groupby('Group')['Value'].mean()
                ratio_fontsize = float(self.get_property('ratio_font_size'))
                ratio_text_xoffset = float(self.get_property('ratio_text_xoffset'))
                ratio_text_yoffset = float(self.get_property('ratio_text_yoffset'))
                ratio_text_color = parse_color(self.get_property('ratio_text_color'), 'red')
                if ctrl_group in group_means and group_means[ctrl_group] != 0 and not pd.isna(group_means[ctrl_group]):
                    ctrl_mean = group_means[ctrl_group]
                    
                    for i, group in enumerate(plot_order):
                        if group in group_means:
                            if group == ctrl_group:
                                continue
                            group_mean = group_means[group]
                            fold_change = group_mean / ctrl_mean
                            
                            ax.text(i + ratio_text_xoffset, group_mean + ratio_text_yoffset, f"{fold_change:.2f}", color=ratio_text_color, va='center', ha='left', fontsize=ratio_fontsize, fontweight='bold')
            
            if tukey_df is not None and not tukey_df.empty:
                sp = _stat_props(self)
                _gmax = float(df_long['Value'].max()) if not df_long.empty else 0.0
                y_max_per_group = {
                    g: (
                        float(df_long.loc[df_long['Group'] == g, 'Value'].max())
                        if (df_long['Group'] == g).any() else _gmax
                    )
                    for g in plot_order
                }
                _draw_stat_brackets(
                    ax,
                    tukey_df,
                    group_to_x_idx={g: i for i, g in enumerate(plot_order)},
                    y_max_per_group=y_max_per_group,
                    y_offset_frac=sp['y_offset'],
                    show_ns=sp['show_ns'],
                    line_color=sp['line_color'],
                    line_width=sp['line_width'],
                    text_color=sp['text_color'],
                    text_size=sp['text_size'],
                    label_mode=sp['label_mode'],
                )
                    
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            
            title_fs = float(self.get_property('title_fontsize'))
            label_fs = float(self.get_property('label_fontsize'))
            rot_val = float(self.get_property('tick_rotation'))
                
            x_label_text = str(self.get_property('x_label') or '').replace('\\n', '\n')
            y_label_text = str(self.get_property('y_label') or '').replace('\\n', '\n')
            title_text = str(self.get_property('plot_title') or '').replace('\\n', '\n')
            
            ax.set_ylabel(y_label_text, fontsize=label_fs)
            ax.set_title(title_text, fontweight='bold', fontsize=title_fs)

            # Tighten group spacing — groups sit at x = 0, 1, …, n-1;
            # this controls the padding on either side.
            try:
                spacing = float(self.get_property('group_spacing'))
            except (ValueError, TypeError):
                spacing = 0.5
            n_groups = len(plot_order)
            ax.set_xlim(-spacing, max(n_groups - 1, 0) + spacing)

            if has_subgroups:
                ax.set_xticks(range(len(plot_order)))
                ax.set_xticklabels(x_labels, fontweight='bold', rotation=rot_val)
                
                ax.set_xlabel(x_label_text, labelpad=30, fontsize=label_fs)
                
                line_y = float(self.get_property('subgroup_bracket_yoffset'))
                text_y = float(self.get_property('subgroup_text_yoffset'))
                line_w = float(self.get_property('subgroup_bracket_linewidth'))
                text_fs = float(self.get_property('subgroup_text_fontsize'))
                
                for sg_name, s_idx, e_idx in subgroup_spans:
                    ax.annotate('', xy=(s_idx, line_y), xytext=(e_idx, line_y), xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'), arrowprops=dict(arrowstyle='-', color='black', linewidth=line_w))
                    ax.text((s_idx + e_idx) / 2, text_y, sg_name, transform=ax.get_xaxis_transform(), ha='center', va='top', fontweight='bold', fontsize=text_fs)
            else:
                ax.set_xlabel(x_label_text, fontsize=label_fs)
                if rot_val != 0.0:
                    ax.tick_params(axis='x', rotation=rot_val)
            
            fig.tight_layout()
            
            self.output_values['plot'] = FigureData(payload=fig)
            
            self.set_progress(100)
            self.mark_clean()
            return True, None

        except Exception as e:
            self.mark_error()
            return False, str(e)


# ── Figure-editor node ────────────────────────────────────────────────────────

# Aesthetic presets: each maps a subset of _apply_params keys.
_FIGURE_PRESETS = {
    'Dark': {
        'background': {'fig_bg': [0.12, 0.12, 0.12, 1.0], 'axes_bg': [0.17, 0.17, 0.17, 1.0]},
        'title':  {'color': [0.92, 0.92, 0.92, 1.0]},
        'xaxis':  {'color': [0.85, 0.85, 0.85, 1.0]},
        'yaxis':  {'color': [0.85, 0.85, 0.85, 1.0]},
        'xtick':  {'labelsize': 10.0, 'direction': 'out'},
        'ytick':  {'labelsize': 10.0, 'direction': 'out'},
        'spines': {
            'left':   {'visible': True,  'color': [0.45, 0.45, 0.45, 1.0], 'linewidth': 1.0},
            'bottom': {'visible': True,  'color': [0.45, 0.45, 0.45, 1.0], 'linewidth': 1.0},
            'right':  {'visible': False, 'color': [0.45, 0.45, 0.45, 1.0], 'linewidth': 1.0},
            'top':    {'visible': False, 'color': [0.45, 0.45, 0.45, 1.0], 'linewidth': 1.0},
        },
        'xgrid': {'visible': True,  'color': [0.28, 0.28, 0.28, 1.0], 'linestyle': 'dashed', 'linewidth': 0.6},
        'ygrid': {'visible': False, 'color': [0.28, 0.28, 0.28, 1.0], 'linestyle': 'dashed', 'linewidth': 0.6},
    },
    'Light': {
        'background': {'fig_bg': [1.0, 1.0, 1.0, 1.0], 'axes_bg': [1.0, 1.0, 1.0, 1.0]},
        'title':  {'color': [0.1, 0.1, 0.1, 1.0]},
        'xaxis':  {'color': [0.2, 0.2, 0.2, 1.0]},
        'yaxis':  {'color': [0.2, 0.2, 0.2, 1.0]},
        'xtick':  {'labelsize': 10.0, 'direction': 'out'},
        'ytick':  {'labelsize': 10.0, 'direction': 'out'},
        'spines': {
            'left':   {'visible': True,  'color': [0.2, 0.2, 0.2, 1.0], 'linewidth': 1.0},
            'bottom': {'visible': True,  'color': [0.2, 0.2, 0.2, 1.0], 'linewidth': 1.0},
            'right':  {'visible': False, 'color': [0.2, 0.2, 0.2, 1.0], 'linewidth': 1.0},
            'top':    {'visible': False, 'color': [0.2, 0.2, 0.2, 1.0], 'linewidth': 1.0},
        },
        'xgrid': {'visible': False, 'color': [0.85, 0.85, 0.85, 1.0], 'linestyle': 'dashed', 'linewidth': 0.6},
        'ygrid': {'visible': False, 'color': [0.85, 0.85, 0.85, 1.0], 'linestyle': 'dashed', 'linewidth': 0.6},
    },
    'Publication': {
        'background': {'fig_bg': [1.0, 1.0, 1.0, 1.0], 'axes_bg': [1.0, 1.0, 1.0, 1.0]},
        'title':  {'color': [0.0, 0.0, 0.0, 1.0], 'fontweight': 'bold'},
        'xaxis':  {'color': [0.0, 0.0, 0.0, 1.0]},
        'yaxis':  {'color': [0.0, 0.0, 0.0, 1.0]},
        'xtick':  {'labelsize': 11.0, 'direction': 'out'},
        'ytick':  {'labelsize': 11.0, 'direction': 'out'},
        'spines': {
            'left':   {'visible': True,  'color': [0.0, 0.0, 0.0, 1.0], 'linewidth': 1.2},
            'bottom': {'visible': True,  'color': [0.0, 0.0, 0.0, 1.0], 'linewidth': 1.2},
            'right':  {'visible': False, 'color': [0.0, 0.0, 0.0, 1.0], 'linewidth': 1.2},
            'top':    {'visible': False, 'color': [0.0, 0.0, 0.0, 1.0], 'linewidth': 1.2},
        },
        'xgrid': {'visible': False, 'color': [0.9, 0.9, 0.9, 1.0], 'linestyle': 'dashed', 'linewidth': 0.5},
        'ygrid': {'visible': False, 'color': [0.9, 0.9, 0.9, 1.0], 'linestyle': 'dashed', 'linewidth': 0.5},
    },
    'Pastel': {
        'background': {'fig_bg': [0.97, 0.97, 1.0, 1.0], 'axes_bg': [0.95, 0.95, 1.0, 1.0]},
        'title':  {'color': [0.25, 0.25, 0.35, 1.0]},
        'xaxis':  {'color': [0.3,  0.3,  0.4,  1.0]},
        'yaxis':  {'color': [0.3,  0.3,  0.4,  1.0]},
        'xtick':  {'labelsize': 10.0, 'direction': 'out'},
        'ytick':  {'labelsize': 10.0, 'direction': 'out'},
        'spines': {
            'left':   {'visible': True,  'color': [0.6, 0.6, 0.75, 1.0], 'linewidth': 1.0},
            'bottom': {'visible': True,  'color': [0.6, 0.6, 0.75, 1.0], 'linewidth': 1.0},
            'right':  {'visible': False, 'color': [0.6, 0.6, 0.75, 1.0], 'linewidth': 1.0},
            'top':    {'visible': False, 'color': [0.6, 0.6, 0.75, 1.0], 'linewidth': 1.0},
        },
        'xgrid': {'visible': True,  'color': [0.82, 0.82, 0.92, 1.0], 'linestyle': 'solid', 'linewidth': 0.8},
        'ygrid': {'visible': False, 'color': [0.82, 0.82, 0.92, 1.0], 'linestyle': 'solid', 'linewidth': 0.8},
    },
}

class FigureEditNode(BaseExecutionNode):
    """
    Interactively edits the aesthetics of any FigureData input via a popup dialog.

    Takes any FigureData, lets the user adjust titles, axes, colours,
    spines, lines, and annotations, then outputs the modified figure.
    Stored settings are persisted with the node and re-applied on every
    run.

    Keywords: figure editor, style plot, adjust axes, annotation edit, post-format chart, 圖表編輯, 樣式調整, 標注, 格式化, 繪圖
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME = 'Figure Editor'
    PORT_SPEC = {'inputs': ['figure'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('figure', color=PORT_COLORS['figure'])
        self.add_output('plot', multi_output=True, color=PORT_COLORS['figure'])

        # Persisted settings stored as a JSON string (hidden from the Properties panel)
        self.create_property(
            '_fig_params_json', '',
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value)

        # Embed buttons in the node body via NodeToolBoxWidget
        self._ctrl = NodeToolBoxWidget(self.view, name='fig_edit_ctrl', label='Figure Editor')
        self._ctrl._toolbox.setFixedHeight(155)

        edit_btn  = QtWidgets.QPushButton("Edit Figure Settings\u2026")
        reset_btn = QtWidgets.QPushButton("Reset Settings")
        edit_btn.clicked.connect(self._open_edit_dialog)
        reset_btn.clicked.connect(self._reset_settings)

        # Preset selector
        preset_row = QtWidgets.QWidget()
        preset_layout = QtWidgets.QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        self._preset_combo = QtWidgets.QComboBox()
        self._preset_combo.addItems(list(_FIGURE_PRESETS.keys()))
        apply_preset_btn = QtWidgets.QPushButton("Apply Preset")
        apply_preset_btn.clicked.connect(self._apply_preset_action)
        preset_layout.addWidget(self._preset_combo, 1)
        preset_layout.addWidget(apply_preset_btn)

        self._ctrl.add_widget_to_page('Controls', edit_btn)
        self._ctrl.add_widget_to_page('Controls', reset_btn)
        self._ctrl.add_widget_to_page('Controls', preset_row)
        self.add_custom_widget(self._ctrl)

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_upstream_fig(self):
        """Return the matplotlib Figure from the upstream node, or None."""
        in_port = self.inputs().get('figure')
        if not in_port or not in_port.connected_ports():
            return None
        up  = in_port.connected_ports()[0]
        val = up.node().output_values.get(up.name(), None)
        if val is None:
            return None
        fig = val.payload if hasattr(val, 'payload') else val
        return fig if hasattr(fig, 'axes') else None

    def _open_edit_dialog(self):
        fig = self._get_upstream_fig()
        if fig is None:
            QtWidgets.QMessageBox.information(
                None, "No Figure",
                "Connect and run an upstream plot node first,\n"
                "then click 'Edit Figure Settings'.")
            return

        params = _extract_params(fig)

        # Re-inject ALL previously-saved settings so every user change
        # (colors, styles, offsets, etc.) is visible when the dialog reopens.
        stored_json = self.get_property('_fig_params_json')
        if stored_json:
            try:
                stored = json.loads(stored_json)
                def _deep_merge(base, override):
                    for k, v in override.items():
                        if isinstance(v, dict) and isinstance(base.get(k), dict):
                            _deep_merge(base[k], v)
                        else:
                            base[k] = v
                _deep_merge(params, stored)
            except Exception:
                pass

        def _on_apply(applied_params):
            """Save params and re-evaluate this node + direct downstream nodes."""
            try:
                self.set_property('_fig_params_json',
                                  json.dumps(applied_params),
                                  push_undo=False)
                self.evaluate()
                self.mark_clean()
                # Propagate to directly-connected downstream nodes (e.g. DisplayNode)
                for out_port in self.output_ports():
                    for cp in out_port.connected_ports():
                        down = cp.node()
                        if hasattr(down, 'evaluate'):
                            try:
                                down.evaluate()
                                down.mark_clean()
                            except Exception:
                                pass
            except Exception as e:
                print(f"[FigureEditNode] Apply error: {e}")

        dlg = FigureEditDialog(params, on_apply=_on_apply)
        if dlg.exec_():
            self.set_property('_fig_params_json',
                              json.dumps(dlg.get_params()),
                              push_undo=False)

    def _reset_settings(self):
        self.set_property('_fig_params_json', '', push_undo=False)

    def _apply_preset_action(self):
        """Deep-merge the selected preset into stored params and re-evaluate."""
        preset_name = self._preset_combo.currentText()
        preset = _FIGURE_PRESETS.get(preset_name)
        if not preset:
            return
        # Load existing stored params (or start fresh)
        params_json = self.get_property('_fig_params_json')
        try:
            params = json.loads(params_json) if params_json else {}
        except Exception:
            params = {}
        # Deep-merge: only overwrite keys defined in the preset
        def _deep_merge(base, overlay):
            for k, v in overlay.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    _deep_merge(base[k], v)
                else:
                    base[k] = v
        _deep_merge(params, preset)
        try:
            self.set_property('_fig_params_json', json.dumps(params), push_undo=False)
            self.evaluate()
            self.mark_clean()
            for out_port in self.output_ports():
                for cp in out_port.connected_ports():
                    down = cp.node()
                    if hasattr(down, 'evaluate'):
                        try:
                            down.evaluate()
                            down.mark_clean()
                        except Exception:
                            pass
        except Exception as e:
            print(f"[FigureEditNode] Preset apply error: {e}")

    # ── evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self):
        self.reset_progress()

        in_port = self.inputs().get('figure')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input figure connected"

        up  = in_port.connected_ports()[0]
        val = up.node().output_values.get(up.name(), None)
        if val is None:
            self.mark_error()
            return False, "Upstream node has no output yet"

        fig_in = val.payload if hasattr(val, 'payload') else val
        if not hasattr(fig_in, 'axes'):
            self.mark_error()
            return False, "Input is not a matplotlib Figure"

        # Work on a copy so upstream figures are not mutated
        try:
            fig = copy.deepcopy(fig_in)
        except Exception:
            fig = fig_in

        self.set_progress(50)

        params_json = self.get_property('_fig_params_json')
        if params_json:
            try:
                _apply_params(fig, json.loads(params_json))
            except Exception:
                pass   # silently fall back to unmodified figure

        self.output_values['plot'] = FigureData(payload=fig)
        self.mark_clean()
        self.set_progress(100)
        return True, None


# ===========================================================================
# Shared helpers for new plot nodes
# ===========================================================================

def _parse_color(color_val, default='black'):
    """Convert [R,G,B,A] list or string to a matplotlib color tuple."""
    from matplotlib.colors import to_rgba
    if isinstance(color_val, (list, tuple)) and len(color_val) >= 3:
        return tuple(c / 255.0 if c > 1 else c for c in color_val[:4])
    try:
        return to_rgba(str(color_val))
    except Exception:
        return to_rgba(default)


def _draw_stat_brackets(ax, stat_df, group_to_x_idx: dict, y_max_per_group: dict,
                         y_offset_frac: float, show_ns: bool,
                         line_color, line_width: float,
                         text_color, text_size: float,
                         label_mode: str = 'Stars (*, **, ***)'):
    """
    Draw significance brackets above a grouped axis.

    Parameters
    ----------
    group_to_x_idx   : {group_name: integer x position}
    y_max_per_group  : {group_name: current maximum y for that group's column}
    y_offset_frac    : fraction of global max used as step height between brackets
    """
    def _stars(p):
        if pd.isna(p): return 'ns'
        p = float(p)
        if p <= 0.0001: return '****'
        if p <= 0.001:  return '***'
        if p <= 0.01:   return '**'
        if p <= 0.05:   return '*'
        return 'ns'

    def _use_p_scientific(mode):
        m = str(mode or '').strip().lower()
        return m in {
            'p-value (scientific)', 'p value (scientific)',
            'p_sci', 'p-sci', 'scientific', 'pvalue_sci', 'p'
        }

    def _label_for_p(p, mode):
        if pd.isna(p):
            return 'ns'
        p = float(p)
        if _use_p_scientific(mode):
            return f"p={p:.2e}"
        return _stars(p)

    def _coerce_p(v):
        if pd.isna(v):
            return np.nan
        try:
            return float(v)
        except Exception:
            s = str(v).strip()
            if not s:
                return np.nan
            if s.startswith('<'):
                s = s[1:].strip()
            m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
            if not m:
                return np.nan
            try:
                return float(m.group(0))
            except Exception:
                return np.nan

    def _dedup_pairs(df, p_col_name):
        # Keep one row per unordered pair; if duplicated, keep the smallest p.
        best = {}
        order = []
        for _, r in df.iterrows():
            g1 = str(r.get('group1', '')).strip()
            g2 = str(r.get('group2', '')).strip()
            if not g1 or not g2:
                continue
            p = _coerce_p(r.get(p_col_name, np.nan))
            k = tuple(sorted((g1, g2)))
            if k not in best:
                best[k] = (g1, g2, p)
                order.append(k)
                continue
            _, _, prev_p = best[k]
            if (pd.isna(prev_p) and not pd.isna(p)) or (not pd.isna(prev_p) and not pd.isna(p) and p < prev_p):
                best[k] = (g1, g2, p)
        return [best[k] for k in order]

    p_col = next((c for c in stat_df.columns
                  if 'p-adj' in c.lower() or 'p_adj' in c.lower()), None)
    if not p_col:
        p_col = next((c for c in stat_df.columns
                      if 'p-value' in c.lower() or 'p_value' in c.lower()), None)
    if not p_col:
        return

    pairs = _dedup_pairs(stat_df, p_col)
    if not pairs:
        return

    # Sort: left group position (left to right), then by span (nearest first)
    pairs.sort(key=lambda t: (
        min(group_to_x_idx.get(t[0], 0), group_to_x_idx.get(t[1], 0)),
        abs(group_to_x_idx.get(t[0], 0) - group_to_x_idx.get(t[1], 0))))

    heights    = dict(y_max_per_group)
    raw_vals   = [v for v in heights.values() if pd.notna(v)]
    if raw_vals:
        y_min0, y_max0 = min(raw_vals), max(raw_vals)
    else:
        y_min0, y_max0 = ax.get_ylim()
    span = max(y_max0 - y_min0, abs(y_max0), 1.0)
    h_step = max(span * max(float(y_offset_frac), 0.0), np.finfo(float).eps)

    for g1, g2, p_val in pairs:
        if g1 not in group_to_x_idx or g2 not in group_to_x_idx:
            continue
        lbl = _label_for_p(p_val, label_mode)
        if lbl == 'ns' and not show_ns:
            continue

        i1, i2   = group_to_x_idx[g1], group_to_x_idx[g2]
        rel_idxs = set(range(min(i1, i2), max(i1, i2) + 1))
        base_h   = max(v for g, v in heights.items()
                       if group_to_x_idx.get(g) in rel_idxs)
        curr_h   = base_h + h_step

        _stat_lbl = f"{g1}–{g2}"
        line, = ax.plot([i1, i2], [curr_h, curr_h], lw=line_width, c=line_color)
        line.set_label(_stat_lbl)
        line.set_gid(f'stat_line:{_stat_lbl}')
        # Mark as stat bracket so legend can exclude it
        line._is_stat_bracket = True
        ax.text((i1 + i2) / 2, curr_h, lbl, ha='center', va='bottom',
                fontsize=text_size, color=text_color,
                gid=f'stat_text:{_stat_lbl}')

        for g in heights:
            if group_to_x_idx.get(g) in rel_idxs:
                heights[g] = curr_h + h_step

    if heights:
        ax.set_ylim(ax.get_ylim()[0], max(heights.values()) + h_step * 1.5)

    # Rebuild legend excluding stat bracket lines
    legend = ax.get_legend()
    if legend is not None:
        handles, labels = ax.get_legend_handles_labels()
        clean = [(h, l) for h, l in zip(handles, labels)
                 if not getattr(h, '_is_stat_bracket', False)]
        if clean:
            ax.legend(*zip(*clean))
        else:
            legend.remove()


def _check_group_stat_df(df) -> "str | None":
    """
    Validate that *df* is a group-comparison result table compatible with the
    significance-bracket overlay (i.e. produced by GroupedComparisonNode or
    PairwiseComparisonNode).

    Returns None when valid, or a human-readable error string when not.
    """
    if df is None:
        return None
    cols = list(df.columns)
    missing = {'group1', 'group2'} - set(cols)
    if missing:
        return (
            "The 'stats' port expects a group-comparison table with 'group1' and "
            "'group2' columns (connect the output of GroupedComparisonNode or "
            f"PairwiseComparisonNode). The connected output has columns: {cols[:10]}"
        )
    has_p = any(
        'p-value' in c.lower() or 'p_value' in c.lower() or
        'p-adj' in c.lower() or 'p_adj' in c.lower()
        for c in cols
    )
    if not has_p:
        return (
            "Stats table must contain a p-value column ('p-value' or 'p-adj'). "
            f"Got columns: {cols[:10]}"
        )
    return None


def _read_table_port(node, port_name: str):
    """Return a DataFrame from a table/stat port, or None."""
    port = node.inputs().get(port_name)
    if not port or not port.connected_ports():
        return None
    cp   = port.connected_ports()[0]
    data = cp.node().output_values.get(cp.name())
    if isinstance(data, TableData):
        return data.df
    if isinstance(data, StatData):
        return data.df
    return None


def _stat_props(node):
    """Read stats-overlay properties from a plot node."""
    return dict(
        show_ns    = bool(node.get_property('stat_show_ns')),
        label_mode = str(node.get_property('stat_label_mode') or 'Stars (*, **, ***)'),
        line_color = _parse_color(node.get_property('stat_line_color'), 'black'),
        line_width = float(node.get_property('stat_line_width') or 1.5),
        text_color = _parse_color(node.get_property('stat_text_color'), 'black'),
        text_size  = float(node.get_property('stat_text_size') or 12.0),
        y_offset   = float(node.get_property('stat_y_offset') or 0.05),
    )


def _add_stat_hidden_props(node):
    """Register stats-overlay properties as hidden (FigureEditNode-compatible)."""
    import NodeGraphQt
    H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
    node.create_property('stat_show_ns',    False,          widget_type=H)
    node.create_property('stat_label_mode', 'Stars (*, **, ***)', widget_type=H)
    node.create_property('stat_line_color', [0, 0, 0, 255], widget_type=H)
    node.create_property('stat_line_width', 1.5,            widget_type=H)
    node.create_property('stat_text_color', [0, 0, 0, 255], widget_type=H)
    node.create_property('stat_text_size',  12.0,           widget_type=H)
    node.create_property('stat_y_offset',   0.05,           widget_type=H)


def _make_fig(node):
    """Create a matplotlib Figure + canvas using fig_width / fig_height props."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    w   = float(node.get_property('fig_width')  or 8)
    h   = float(node.get_property('fig_height') or 6)
    dpi = float(node.get_property('fig_dpi')    or 100)
    fig = Figure(figsize=(w, h), dpi=dpi)
    FigureCanvasAgg(fig)
    ax  = fig.add_subplot(111)
    return fig, ax


def _finalize_ax(ax, node, x_label=None, y_label=None, title=None):
    """Apply label / title / spine cleanup."""
    ax.set_xlabel(x_label or str(node.get_property('x_label') or ''))
    ax.set_ylabel(y_label or str(node.get_property('y_label') or ''))
    ax.set_title(title   or str(node.get_property('plot_title') or ''),
                 fontweight='bold')
    rot = float(node.get_property('tick_rotation') or 0)
    if rot:
        ax.tick_params(axis='x', rotation=rot)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)


# ===========================================================================
# ViolinPlotNode
# ===========================================================================

class ViolinPlotNode(PlotToolboxMixin, BaseExecutionNode):
    """
    Creates a violin plot with optional significance-bracket overlay.

    Connects to the same StatData output as SwarmPlotNode. Use
    **order** to fix the x-axis group order (comma-separated).

    Columns:
    - **x_col** — categorical group column
    - **y_col** — numeric value column
    - **order** — comma-separated group order for the X axis

    Options:
    - *inner_box* — draw a mini box plot inside each violin
    - *palette* — colour palette for groups

    Keywords: violin plot, distribution plot, density by group, stats brackets, 小提琴圖, 分佈, 核密度, 分組, 繪圖
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'Violin Plot + Stats'
    PORT_SPEC      = {'inputs': ['table', 'table'], 'outputs': ['figure']}

    def __init__(self):
        self._toolbox_widgets = {}
        super().__init__()
        self.add_input('data',  color=PORT_COLORS['table'])
        self.add_input('stats', color=PORT_COLORS['stat'])
        self.add_output('plot', color=PORT_COLORS['figure'])

        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('x_col',       'Group', widget_type=H)
        self.create_property('y_col',       'Value', widget_type=H)
        self.create_property('order',       '',      widget_type=H)
        self.create_property('palette',     'Set2',  widget_type=H)
        self.create_property('x_label',     '',      widget_type=H)
        self.create_property('y_label',     '',      widget_type=H)
        self.create_property('plot_title',  '',      widget_type=H)
        self.create_property('inner_box',   True,    widget_type=H)
        self.create_property('bw_adjust',  1.0,     widget_type=H)
        self.create_property('violin_scale', 'area', widget_type=H)
        self.create_property('show_points', False,   widget_type=H)
        self.create_property('point_style', 'Strip', widget_type=H)
        self.create_property('point_size',  3,       widget_type=H)
        self.create_property('point_color', 'match', widget_type=H)
        self.create_property('fig_width',   8.0,     widget_type=H)
        self.create_property('fig_height',  6.0,     widget_type=H)
        self.create_property('tick_rotation', 0.0,   widget_type=H)
        _add_stat_hidden_props(self)

        self._build_toolbox(400)
        self._tb_column_selector('x_col',      'Group Column',   'Data', 'Group')
        self._tb_column_selector('y_col',      'Value Column',   'Data', 'Value')
        self._tb_order_list('order', 'X-Axis Order', 'Data')
        self._tb_combo('palette',   'Palette',         'Data',
                       ['Set2', 'husl', 'colorblind', 'pastel', 'muted', 'None'])
        self._tb_text('x_label',    'X Label',        'Data', '')
        self._tb_text('y_label',    'Y Label',        'Data', '')
        self._tb_text('plot_title', 'Title',           'Data', '')
        self._tb_checkbox('inner_box', 'Inner Box',   'Data', True)
        self._tb_spinbox('bw_adjust', 'Bandwidth Adjust', 'Data', 1.0, 0.1, 3.0, 0.1, 1)
        self._tb_combo('violin_scale', 'Violin Scale', 'Data', ['area', 'count', 'width'])
        self._tb_checkbox('show_points', 'Show Data Points', 'Data', False)
        self._tb_combo('point_style', 'Point Style',   'Data', ['Strip', 'Swarm'])
        self._tb_spinbox('point_size', 'Point Size',    'Data', 3, 1, 20)
        self._tb_combo('point_color', 'Point Color',   'Data', ['match', 'black', 'gray', 'white'])
        self._tb_add_figure_page()
        self._tb_add_stats_page()

    def evaluate(self):
        self.reset_progress()
        import seaborn as sns
        sns.set_theme(style='ticks')

        df       = _read_table_port(self, 'data')
        stat_df  = _read_table_port(self, 'stats')
        if df is None:
            self.mark_error(); return False, "No data connected"
        self._tb_refresh_columns(df, 'x_col', 'y_col')

        x_col   = str(self.get_property('x_col') or 'group')
        y_col   = str(self.get_property('y_col') or 'value')
        order_s = str(self.get_property('order') or '')
        palette = str(self.get_property('palette') or 'Set2')
        if palette == 'None': palette = None
        order   = [g.strip() for g in order_s.split(',') if g.strip()] or None
        inner   = 'box' if bool(self.get_property('inner_box')) else None

        if x_col not in df.columns or y_col not in df.columns:
            self.mark_error()
            return False, f"Columns '{x_col}' or '{y_col}' not found"

        groups = order or sorted(df[x_col].dropna().unique().tolist())
        g2idx  = {g: i for i, g in enumerate(groups)}

        fig, ax = _make_fig(self)
        _bw  = float(self.get_property('bw_adjust') or 1.0)
        _vsc = str(self.get_property('violin_scale') or 'area')
        sns.violinplot(data=df, x=x_col, y=y_col, hue=x_col, order=groups,
                       palette=palette, inner=inner, cut=0, bw_adjust=_bw,
                       scale=_vsc, legend=False, ax=ax)
        if bool(self.get_property('show_points')):
            pt_style = str(self.get_property('point_style') or 'Strip')
            pt_size = int(self.get_property('point_size') or 3)
            pt_color = str(self.get_property('point_color') or 'match')
            pt_kw = dict(data=df, x=x_col, y=y_col, order=groups,
                         alpha=0.6, size=pt_size, ax=ax)
            if pt_color == 'match':
                pt_kw['hue'] = x_col
                pt_kw['palette'] = palette
                pt_kw['legend'] = False
                pt_kw['edgecolor'] = 'gray'
                pt_kw['linewidth'] = 0.5
            else:
                pt_kw['color'] = pt_color
            if pt_style == 'Swarm':
                sns.swarmplot(**pt_kw)
            else:
                pt_kw['jitter'] = True
                sns.stripplot(**pt_kw)
        self.set_progress(60)

        if stat_df is not None:
            err = _check_group_stat_df(stat_df)
            if err:
                self.mark_error(); return False, err
            y_max = {g: float(df[df[x_col] == g][y_col].max()) for g in groups}
            sp    = _stat_props(self)
            _draw_stat_brackets(ax, stat_df, g2idx, y_max,
                                 sp['y_offset'], sp['show_ns'],
                                 sp['line_color'], sp['line_width'],
                                 sp['text_color'], sp['text_size'],
                                 sp['label_mode'])

        _finalize_ax(ax, self)
        fig.tight_layout()
        self.output_values['plot'] = FigureData(payload=fig)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ===========================================================================
# BoxPlotNode
# ===========================================================================

class BoxPlotNode(PlotToolboxMixin, BaseExecutionNode):
    """
    Creates a box-and-whisker plot with optional significance-bracket overlay.

    Columns:
    - **x_col** — categorical group column
    - **y_col** — numeric value column
    - **order** — comma-separated group order for the X axis

    Options:
    - *show_points* — overlay individual data points on the boxes
    - *palette* — colour palette for groups

    Keywords: box plot, whisker, quartile, distribution summary, outlier display, 盒鬚圖, 四分位, 分佈, 異常值, 繪圖
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'Box Plot + Stats'
    PORT_SPEC      = {'inputs': ['table', 'table'], 'outputs': ['figure']}

    def __init__(self):
        self._toolbox_widgets = {}
        super().__init__()
        self.add_input('data',  color=PORT_COLORS['table'])
        self.add_input('stats', color=PORT_COLORS['stat'])
        self.add_output('plot', color=PORT_COLORS['figure'])

        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('x_col',        'Group', widget_type=H)
        self.create_property('y_col',        'Value', widget_type=H)
        self.create_property('order',        '',      widget_type=H)
        self.create_property('palette',      'Set2',  widget_type=H)
        self.create_property('x_label',      '',      widget_type=H)
        self.create_property('y_label',      '',      widget_type=H)
        self.create_property('plot_title',   '',      widget_type=H)
        self.create_property('show_points',  False,   widget_type=H)
        self.create_property('point_style', 'Strip', widget_type=H)
        self.create_property('point_size',  3,       widget_type=H)
        self.create_property('point_color', 'match', widget_type=H)
        self.create_property('box_width',   0.8,     widget_type=H)
        self.create_property('show_mean',   False,   widget_type=H)
        self.create_property('fig_width',    8.0,     widget_type=H)
        self.create_property('fig_height',   6.0,     widget_type=H)
        self.create_property('tick_rotation', 0.0,    widget_type=H)
        _add_stat_hidden_props(self)

        self._build_toolbox(400)
        self._tb_column_selector('x_col',       'Group Column',    'Data', 'Group')
        self._tb_column_selector('y_col',       'Value Column',    'Data', 'Value')
        self._tb_order_list('order', 'X-Axis Order', 'Data')
        self._tb_combo('palette',    'Palette',          'Data',
                       ['Set2', 'husl', 'colorblind', 'pastel', 'muted', 'None'])
        self._tb_spinbox('box_width', 'Box Width', 'Data', 0.8, 0.1, 1.0, 0.05, 2)
        self._tb_checkbox('show_mean', 'Show Mean Marker', 'Data', False)
        self._tb_text('x_label',     'X Label',         'Data', '')
        self._tb_text('y_label',     'Y Label',         'Data', '')
        self._tb_text('plot_title',  'Title',            'Data', '')
        self._tb_checkbox('show_points', 'Show Data Points', 'Data', False)
        self._tb_combo('point_style', 'Point Style',     'Data', ['Strip', 'Swarm'])
        self._tb_spinbox('point_size', 'Point Size',      'Data', 3, 1, 20)
        self._tb_combo('point_color', 'Point Color',     'Data', ['match', 'black', 'gray', 'white'])
        self._tb_add_figure_page()
        self._tb_add_stats_page()

    def evaluate(self):
        self.reset_progress()
        import seaborn as sns
        sns.set_theme(style='ticks')

        df      = _read_table_port(self, 'data')
        stat_df = _read_table_port(self, 'stats')
        if df is None:
            self.mark_error(); return False, "No data connected"
        self._tb_refresh_columns(df, 'x_col', 'y_col')

        x_col   = str(self.get_property('x_col') or 'Group')
        y_col   = str(self.get_property('y_col') or 'Value')
        order_s = str(self.get_property('order') or '')
        palette = str(self.get_property('palette') or 'Set2')
        if palette == 'None': palette = None
        order   = [g.strip() for g in order_s.split(',') if g.strip()] or None

        if x_col not in df.columns or y_col not in df.columns:
            self.mark_error()
            return False, f"Columns '{x_col}' or '{y_col}' not found"

        groups  = order or sorted(df[x_col].dropna().unique().tolist())
        g2idx   = {g: i for i, g in enumerate(groups)}
        fig, ax = _make_fig(self)

        _bw = float(self.get_property('box_width') or 0.8)
        sns.boxplot(data=df, x=x_col, y=y_col, hue=x_col, order=groups,
                    palette=palette, width=_bw, legend=False, ax=ax)
        if bool(self.get_property('show_mean')):
            for gi, grp in enumerate(groups):
                vals = df[df[x_col] == grp][y_col].dropna()
                if len(vals):
                    ax.scatter([gi], [vals.mean()], marker='D', color='white',
                               edgecolors='black', s=40, zorder=5, linewidths=1.2)
        if bool(self.get_property('show_points')):
            pt_style = str(self.get_property('point_style') or 'Strip')
            pt_size = int(self.get_property('point_size') or 3)
            pt_color = str(self.get_property('point_color') or 'match')
            pt_kw = dict(data=df, x=x_col, y=y_col, order=groups,
                         alpha=0.6, size=pt_size, ax=ax)
            if pt_color == 'match':
                pt_kw['hue'] = x_col
                pt_kw['palette'] = palette
                pt_kw['legend'] = False
                pt_kw['edgecolor'] = 'gray'
                pt_kw['linewidth'] = 0.5
            else:
                pt_kw['color'] = pt_color
            if pt_style == 'Swarm':
                sns.swarmplot(**pt_kw)
            else:
                pt_kw['jitter'] = True
                sns.stripplot(**pt_kw)
        self.set_progress(60)

        if stat_df is not None:
            err = _check_group_stat_df(stat_df)
            if err:
                self.mark_error(); return False, err
            y_max = {g: float(df[df[x_col] == g][y_col].max()) for g in groups}
            sp    = _stat_props(self)
            _draw_stat_brackets(ax, stat_df, g2idx, y_max,
                                 sp['y_offset'], sp['show_ns'],
                                 sp['line_color'], sp['line_width'],
                                 sp['text_color'], sp['text_size'],
                                 sp['label_mode'])

        _finalize_ax(ax, self)
        fig.tight_layout()
        self.output_values['plot'] = FigureData(payload=fig)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ===========================================================================
# BarPlotNode
# ===========================================================================

class BarPlotNode(PlotToolboxMixin, BaseExecutionNode):
    """
    Creates a bar plot showing group means with error bars and optional significance-bracket overlay.

    Columns:
    - **x_col** — categorical group column
    - **y_col** — numeric value column
    - **order** — comma-separated group order for the X axis

    Options:
    - *error_type* — error bar measure: `se`, `sd`, `ci`, or `pi`
    - *show_bar_values* — annotate each bar with its numeric value
    - *palette* — colour palette for groups

    Keywords: bar chart, mean error bars, confidence interval, grouped bars, significance overlay, 長條圖, 誤差棒, 信賴區間, 分組, 繪圖
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'Bar Plot + Stats'
    PORT_SPEC      = {'inputs': ['table', 'table'], 'outputs': ['figure']}

    def __init__(self):
        self._toolbox_widgets = {}
        super().__init__()
        self.add_input('data',  color=PORT_COLORS['table'])
        self.add_input('stats', color=PORT_COLORS['stat'])
        self.add_output('plot', color=PORT_COLORS['figure'])

        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('x_col',        'Group', widget_type=H)
        self.create_property('y_col',        'Value', widget_type=H)
        self.create_property('order',        '',      widget_type=H)
        self.create_property('palette',      'Set2',  widget_type=H)
        self.create_property('error_type',   'se',    widget_type=H)
        self.create_property('x_label',      '',      widget_type=H)
        self.create_property('y_label',      '',      widget_type=H)
        self.create_property('plot_title',   '',      widget_type=H)
        self.create_property('fig_width',    8.0,     widget_type=H)
        self.create_property('fig_height',   6.0,     widget_type=H)
        self.create_property('tick_rotation', 0.0,    widget_type=H)
        self.create_property('show_bar_values',      False,   widget_type=H)
        self.create_property('bar_value_fmt',        '.2f',   widget_type=H)
        self.create_property('bar_value_fontsize',   9,       widget_type=H)
        self.create_property('bar_value_color',      [0, 0, 0, 255], widget_type=H)
        self.create_property('bar_value_fontweight', 'normal',widget_type=H)
        self.create_property('bar_value_offset',    8,       widget_type=H)
        self.create_property('bar_width',   0.8,     widget_type=H)
        self.create_property('capsize',     0.1,     widget_type=H)
        self.create_property('show_points', False,   widget_type=H)
        self.create_property('point_style', 'Strip', widget_type=H)
        self.create_property('point_size',  3,       widget_type=H)
        self.create_property('point_color', 'match', widget_type=H)
        _add_stat_hidden_props(self)

        self._build_toolbox(500)
        self._tb_column_selector('x_col',       'Group Column',    'Data', 'Group')
        self._tb_column_selector('y_col',       'Value Column',    'Data', 'Value')
        self._tb_order_list('order', 'X-Axis Order', 'Data')
        self._tb_combo('palette',    'Palette',          'Data',
                       ['Set2', 'husl', 'colorblind', 'pastel', 'muted', 'None'])
        self._tb_combo('error_type', 'Error Bars',       'Data', ['se', 'sd', 'ci', 'pi'])
        self._tb_spinbox('bar_width', 'Bar Width', 'Data', 0.8, 0.1, 1.0, 0.05, 2)
        self._tb_spinbox('capsize', 'Error Bar Cap', 'Data', 0.1, 0.0, 0.5, 0.02, 2)
        self._tb_checkbox('show_points', 'Show Data Points', 'Data', False)
        self._tb_combo('point_style', 'Point Style',     'Data', ['Strip', 'Swarm'])
        self._tb_spinbox('point_size', 'Point Size',      'Data', 3, 1, 20)
        self._tb_combo('point_color', 'Point Color',     'Data', ['match', 'black', 'gray', 'white'])
        self._tb_checkbox('show_bar_values', 'Show Bar Values', 'Data', False)
        self._tb_text('bar_value_fmt', 'Value Format (e.g. .2f, .0f, .1%)', 'Data', '.2f')
        self._tb_spinbox('bar_value_fontsize', 'Label Font Size', 'Data', 9, 1, 48)
        self._tb_color('bar_value_color', 'Label Color', 'Data')
        self._tb_combo('bar_value_fontweight', 'Label Font Weight', 'Data',
                       ['normal', 'bold', 'semibold', 'light'])
        self._tb_spinbox('bar_value_offset', 'Label Offset (pt)', 'Data', 4, 0, 50)
        self._tb_text('x_label',     'X Label',         'Data', '')
        self._tb_text('y_label',     'Y Label',         'Data', '')
        self._tb_text('plot_title',  'Title',            'Data', '')
        self._tb_add_figure_page()
        self._tb_add_stats_page()

    def evaluate(self):
        self.reset_progress()
        import seaborn as sns
        sns.set_theme(style='ticks')

        df      = _read_table_port(self, 'data')
        stat_df = _read_table_port(self, 'stats')
        if df is None:
            self.mark_error(); return False, "No data connected"

        x_col      = str(self.get_property('x_col') or 'Group')
        y_col      = str(self.get_property('y_col') or 'Value')
        order_s    = str(self.get_property('order') or '')
        palette    = str(self.get_property('palette') or 'Set2')
        if palette == 'None': palette = None
        order      = [g.strip() for g in order_s.split(',') if g.strip()] or None
        error_type = str(self.get_property('error_type') or 'se')

        if x_col not in df.columns or y_col not in df.columns:
            self.mark_error()
            return False, f"Columns '{x_col}' or '{y_col}' not found"

        groups  = order or sorted(df[x_col].dropna().unique().tolist())
        g2idx   = {g: i for i, g in enumerate(groups)}
        fig, ax = _make_fig(self)

        capsize   = float(self.get_property('capsize') or 0.1)
        bar_width = float(self.get_property('bar_width') or 0.8)
        try:
            # seaborn ≥ 0.12: errorbar kwarg
            sns.barplot(data=df, x=x_col, y=y_col, hue=x_col, order=groups,
                        palette=palette, errorbar=error_type, capsize=capsize,
                        width=bar_width, legend=False, ax=ax)
        except TypeError:
            # older seaborn: ci kwarg
            ci = 68 if error_type == 'se' else 95
            sns.barplot(data=df, x=x_col, y=y_col, hue=x_col, order=groups,
                        palette=palette, legend=False, ci=ci, capsize=capsize,
                        width=bar_width, ax=ax)

        # legend=False skips patch labelling; assign group names explicitly so
        # _extract_params can expose per-bar coloring in the figure editor.
        bar_patches = [p for p in ax.patches if p.get_width() > 0]
        for i, g in enumerate(groups):
            if i < len(bar_patches):
                bar_patches[i].set_label(str(g))

        if bool(self.get_property('show_points')):
            pt_style = str(self.get_property('point_style') or 'Strip')
            pt_size = int(self.get_property('point_size') or 3)
            pt_color = str(self.get_property('point_color') or 'match')
            pt_kw = dict(data=df, x=x_col, y=y_col, order=groups,
                         alpha=0.6, size=pt_size, ax=ax)
            if pt_color == 'match':
                pt_kw['hue'] = x_col
                pt_kw['palette'] = palette
                pt_kw['legend'] = False
                pt_kw['edgecolor'] = 'gray'
                pt_kw['linewidth'] = 0.5
            else:
                pt_kw['color'] = pt_color
            if pt_style == 'Swarm':
                sns.swarmplot(**pt_kw)
            else:
                pt_kw['jitter'] = True
                sns.stripplot(**pt_kw)

        self.set_progress(60)

        if bool(self.get_property('show_bar_values')):
            fmt    = str(self.get_property('bar_value_fmt')        or '.2f').strip()
            fsize  = int(self.get_property('bar_value_fontsize')   or 9)
            _fc_raw = self.get_property('bar_value_color') or [0, 0, 0, 255]
            if isinstance(_fc_raw, (list, tuple)):
                fcolor = tuple(c / 255 for c in _fc_raw[:4])
            else:
                fcolor = str(_fc_raw).strip() or 'black'
            fweight= str(self.get_property('bar_value_fontweight') or 'normal').strip()
            foffset= int(self.get_property('bar_value_offset') or 4)

            # Compute error bar tops from data (more reliable than parsing plot lines)
            err_tops = {}
            error_type = str(self.get_property('error_type') or 'se').strip()
            for i, g in enumerate(groups):
                vals = df[df[x_col].astype(str).str.strip() == str(g)][y_col].dropna().values
                if len(vals) == 0:
                    continue
                mean = np.mean(vals)
                if error_type == 'sd':
                    err = np.std(vals, ddof=1) if len(vals) > 1 else 0
                elif error_type == 'se':
                    err = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                elif error_type == 'ci':
                    from scipy.stats import sem, t as t_dist
                    se = sem(vals)
                    ci_val = t_dist.ppf(0.975, len(vals) - 1) * se if len(vals) > 1 else 0
                    err = ci_val
                else:
                    err = np.std(vals, ddof=1) if len(vals) > 1 else 0
                err_tops[i] = mean + err

            for patch in ax.patches:
                h = patch.get_height()
                if h == 0:
                    continue
                bar_x = patch.get_x() + patch.get_width() / 2
                # Find closest group index
                bar_idx = int(round(bar_x))
                label_y = err_tops.get(bar_idx, h)
                try:
                    label = f'{h:{fmt}}'
                except (ValueError, TypeError):
                    label = str(round(h, 2))
                ax.annotate(
                    label,
                    xy=(bar_x, label_y),
                    xytext=(0, foffset),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=fsize,
                    color=fcolor,
                    fontweight=fweight,
                )

        if stat_df is not None:
            err = _check_group_stat_df(stat_df)
            if err:
                self.mark_error(); return False, err
            # When data points are overlaid, use actual max (dots extend above bar)
            # Otherwise use mean + se (top of error bar)
            y_max = {}
            show_pts = bool(self.get_property('show_points'))
            for g in groups:
                sub = df[df[x_col] == g][y_col].dropna()
                if show_pts:
                    y_max[g] = float(sub.max())
                else:
                    y_max[g] = float(sub.mean() + sub.sem()) if len(sub) > 1 else float(sub.mean())
            sp = _stat_props(self)
            _draw_stat_brackets(ax, stat_df, g2idx, y_max,
                                 sp['y_offset'], sp['show_ns'],
                                 sp['line_color'], sp['line_width'],
                                 sp['text_color'], sp['text_size'],
                                 sp['label_mode'])

        _finalize_ax(ax, self)
        fig.tight_layout()
        self.output_values['plot'] = FigureData(payload=fig)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ===========================================================================
# ScatterPlotNode
# ===========================================================================

class ScatterPlotNode(BaseExecutionNode):
    """
    Creates a scatter plot (X vs Y) with optional regression line and hue grouping.

    Columns:
    - **x_col** — numeric column for the X axis
    - **y_col** — numeric column for the Y axis
    - **hue_col** — optional column for colour-coding by group

    Options:
    - *regression* — overlay a linear regression line
    - *palette* — colour palette for hue groups

    Keywords: scatter, regression, correlation plot, x y chart, hue groups, 散點圖, 迴歸, 相關性, 繪圖, 分組
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'Scatter Plot'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('data',  color=PORT_COLORS['table'])
        self.add_output('plot', color=PORT_COLORS['figure'])

        self._add_column_selector('x_col',       'X Column', text='x', mode='single')
        self._add_column_selector('y_col',       'Y Column', text='y', mode='single')
        self._add_column_selector('hue_col',     'Hue Column (optional)', text='', mode='single')
        self.add_combo_menu('palette',     'Palette',
                            items=['Set2', 'husl', 'colorblind', 'viridis', 'None'])
        self.add_checkbox('regression',    '', text='Regression line', state=False)
        self.add_text_input('x_label',    'X Label',  text='')
        self.add_text_input('y_label',    'Y Label',  text='')
        self.add_text_input('plot_title', 'Title',    text='')

        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('marker_size',   25,    widget_type=H)
        self.create_property('alpha',         0.7,   widget_type=H)
        self.create_property('marker',        'o',   widget_type=H)
        self.create_property('fig_width',     7.0,   widget_type=H)
        self.create_property('fig_height',    6.0,   widget_type=H)
        self.create_property('tick_rotation', 0.0,   widget_type=H)

    def evaluate(self):
        self.reset_progress()
        import seaborn as sns
        sns.set_theme(style='ticks')

        df = _read_table_port(self, 'data')
        if df is None:
            self.mark_error(); return False, "No data connected"

        self._refresh_column_selectors(df, 'x_col', 'y_col', 'hue_col')

        x_col   = str(self.get_property('x_col') or 'x')
        y_col   = str(self.get_property('y_col') or 'y')
        hue_col = str(self.get_property('hue_col') or '') or None
        palette = str(self.get_property('palette') or 'Set2')
        if palette == 'None': palette = None
        reg     = bool(self.get_property('regression'))

        if x_col not in df.columns or y_col not in df.columns:
            self.mark_error()
            return False, f"Columns '{x_col}' or '{y_col}' not found"

        m_size  = int(self.get_property('marker_size') or 25)
        m_alpha = float(self.get_property('alpha') or 0.7)
        m_style = str(self.get_property('marker') or 'o')

        fig, ax = _make_fig(self)
        if reg and hue_col is None:
            sns.regplot(data=df, x=x_col, y=y_col,
                        scatter_kws={'alpha': m_alpha, 's': m_size, 'marker': m_style},
                        ax=ax)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col,
                            hue=hue_col, palette=palette, alpha=m_alpha,
                            s=m_size, marker=m_style, ax=ax)
        self.set_progress(70)
        _finalize_ax(ax, self)
        fig.tight_layout()
        self.output_values['plot'] = FigureData(payload=fig)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ===========================================================================
# HistogramNode
# ===========================================================================

class HistogramNode(BaseExecutionNode):
    """
    Creates a histogram with optional grouping and KDE overlay.

    Columns:
    - **value_col** — numeric column to bin
    - **group_col** — optional categorical column for grouped histograms

    Options:
    - *bins* — number of bins (integer or `"auto"`)
    - *binwidth* — explicit bin width (overrides bins when set)
    - *kde* — overlay a kernel density estimate curve
    - *palette* — colour palette for groups

    Keywords: histogram, distribution, bins, frequency, grouped histogram, 直方圖, 分佈, 頻率, 分組, 繪圖
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'Histogram'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('data',  color=PORT_COLORS['table'])
        self.add_output('plot', color=PORT_COLORS['figure'])

        self._add_column_selector('value_col',   'Value Column',         text='value', mode='multi')
        self._add_column_selector('group_col',   'Group Column (opt.)',  text='', mode='single')
        self.add_text_input('bins',        'Bins (int or "auto")', text='auto')
        self.add_text_input('binwidth',    'Bin width', text='')
        self.add_combo_menu('palette',     'Palette',
                            items=['Set2', 'husl', 'colorblind', 'viridis', 'None'])
        self.add_checkbox('kde',           '', text='KDE overlay', state=True)
        self.add_text_input('x_label',    'X Label',  text='')
        self.add_text_input('y_label',    'Y Label',  text='')
        self.add_text_input('plot_title', 'Title',    text='')

        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('stat',          'count', widget_type=H)
        self.create_property('hist_alpha',    0.7,     widget_type=H)
        self.create_property('fig_width',     7.0,     widget_type=H)
        self.create_property('fig_height',    5.0,     widget_type=H)
        self.create_property('tick_rotation', 0.0,     widget_type=H)

    def evaluate(self):
        self.reset_progress()
        import seaborn as sns
        sns.set_theme(style='ticks')

        df = _read_table_port(self, 'data')
        if df is None:
            self.mark_error(); return False, "No data connected"
        self._refresh_column_selectors(df, 'value_col', 'group_col')

        val_col   = str(self.get_property('value_col') or 'value')
        grp_col   = str(self.get_property('group_col') or '') or None
        bins_s    = str(self.get_property('bins') or 'auto')
        palette   = str(self.get_property('palette') or 'Set2')
        if palette == 'None': palette = None
        kde       = bool(self.get_property('kde'))

        if val_col not in df.columns:
            self.mark_error()
            return False, f"Column '{val_col}' not found"

        try:
            bins = int(bins_s)
        except ValueError:
            bins = 'auto'
        
        try:
            binwidth = float(str(self.get_property('binwidth') or ''))
        except ValueError:
            binwidth = None
        
        hue = grp_col if grp_col and grp_col in df.columns else None
        fig, ax = _make_fig(self)
        _stat  = str(self.get_property('stat') or 'count')
        _alpha = float(self.get_property('hist_alpha') or 0.7)
        sns.histplot(data=df, x=val_col, hue=hue, palette=palette,
                     bins=bins, binwidth=binwidth, kde=kde,
                     stat=_stat, alpha=_alpha, ax=ax)
        self.set_progress(70)
        _finalize_ax(ax, self)
        fig.tight_layout()
        self.output_values['plot'] = FigureData(payload=fig)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ===========================================================================
# JointPlotNode
# ===========================================================================

class JointPlotNode(BaseExecutionNode):
    """
    Creates a joint plot — scatter with marginal distributions on each axis.

    Columns:
    - **x_col** — numeric column for the X axis
    - **y_col** — numeric column for the Y axis
    - **hue_col** — optional column for colour-coding by group

    Options:
    - *kind* — scatter, kde, hex, hist, or reg (scatter + regression)
    - *marginal* — histogram, kde, or both for the marginal distributions
    - *palette* — colour palette for hue groups

    Keywords: joint plot, scatter, marginal distribution, histogram, kde, regression, 聯合圖, 散點, 邊際分佈, 繪圖
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'Joint Plot'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('data',  color=PORT_COLORS['table'])
        self.add_output('plot', color=PORT_COLORS['figure'])

        self._add_column_selector('x_col',    'X Column', text='x', mode='single')
        self._add_column_selector('y_col',    'Y Column', text='y', mode='single')
        self._add_column_selector('hue_col',  'Hue Column (optional)', text='', mode='single')
        self.add_combo_menu('kind',     'Kind',
                            items=['scatter', 'reg', 'kde', 'hex', 'hist'])
        self.add_combo_menu('marginal', 'Marginal',
                            items=['hist', 'kde', 'both'])
        self.add_combo_menu('palette',  'Palette',
                            items=['Set2', 'husl', 'colorblind', 'viridis', 'None'])
        self.add_text_input('x_label',    'X Label',  text='')
        self.add_text_input('y_label',    'Y Label',  text='')
        self.add_text_input('plot_title', 'Title',    text='')

        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('fig_height', 7.0, widget_type=H)

    def evaluate(self):
        self.reset_progress()
        import seaborn as sns
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        sns.set_theme(style='ticks')

        df = _read_table_port(self, 'data')
        if df is None:
            self.mark_error(); return False, "No data connected"

        self._refresh_column_selectors(df, 'x_col', 'y_col', 'hue_col')

        x_col   = str(self.get_property('x_col') or 'x')
        y_col   = str(self.get_property('y_col') or 'y')
        hue_col = str(self.get_property('hue_col') or '').strip() or None
        kind    = str(self.get_property('kind') or 'scatter')
        marginal = str(self.get_property('marginal') or 'hist')
        palette = str(self.get_property('palette') or 'Set2')
        if palette == 'None':
            palette = None

        if x_col not in df.columns or y_col not in df.columns:
            self.mark_error()
            return False, f"Columns '{x_col}' or '{y_col}' not found"
        if hue_col and hue_col not in df.columns:
            hue_col = None

        self.set_progress(20)

        height = float(self.get_property('fig_height') or 7)

        # Map marginal option
        marginal_kws = {}
        if marginal == 'both':
            marginal_kind = 'hist'
            marginal_kws['kde'] = True
        else:
            marginal_kind = marginal

        try:
            g = sns.jointplot(
                data=df, x=x_col, y=y_col,
                hue=hue_col, kind=kind,
                palette=palette, height=height,
                marginal_kws=marginal_kws if marginal == 'both' else {},
            )
            # Override marginal kind if not 'both' (jointplot uses hist by default)
            if marginal != 'both' and kind in ('scatter', 'reg'):
                g.plot_marginals(
                    sns.histplot if marginal == 'hist' else sns.kdeplot,
                    **({'kde': False} if marginal == 'hist' else {}))
        except Exception as e:
            self.mark_error()
            return False, str(e)

        self.set_progress(70)

        # Labels and title
        x_label = str(self.get_property('x_label') or '').strip()
        y_label = str(self.get_property('y_label') or '').strip()
        title   = str(self.get_property('plot_title') or '').strip()
        if x_label:
            g.ax_joint.set_xlabel(x_label)
        if y_label:
            g.ax_joint.set_ylabel(y_label)
        if title:
            g.figure.suptitle(title, y=1.02)

        # Attach canvas for rendering
        FigureCanvasAgg(g.figure)

        self.output_values['plot'] = FigureData(payload=g.figure)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ===========================================================================
# KdePlotNode
# ===========================================================================

class KdePlotNode(BaseExecutionNode):
    """
    Creates a kernel density estimate plot for smooth distribution visualisation.

    Supports optional grouping for comparing multiple distributions
    on the same axes.

    Columns:
    - **value_col** — numeric column to estimate density for
    - **group_col** — optional categorical column for overlaid group curves

    Options:
    - *fill* — fill the area under the density curve
    - *palette* — colour palette for groups

    Keywords: kde, density curve, smooth distribution, probability density, group comparison, 核密度, 密度曲線, 分佈, 機率, 繪圖
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'KDE Plot'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('data',  color=PORT_COLORS['table'])
        self.add_output('plot', color=PORT_COLORS['figure'])

        self._add_column_selector('value_col',   'Value Column',        text='value', mode='single')
        self._add_column_selector('group_col',   'Group Column (opt.)', text='', mode='single')
        self.add_combo_menu('palette',     'Palette',
                            items=['Set2', 'husl', 'colorblind', 'viridis', 'None'])
        self.add_checkbox('fill',          '', text='Fill under curve', state=True)
        self.add_text_input('x_label',    'X Label',  text='')
        self.add_text_input('y_label',    'Y Label',  text='Density')
        self.add_text_input('plot_title', 'Title',    text='')

        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('fig_width',     7.0, widget_type=H)
        self.create_property('fig_height',    5.0, widget_type=H)
        self.create_property('tick_rotation', 0.0, widget_type=H)

    def evaluate(self):
        self.reset_progress()
        import seaborn as sns
        sns.set_theme(style='ticks')

        df = _read_table_port(self, 'data')
        if df is None:
            self.mark_error(); return False, "No data connected"

        self._refresh_column_selectors(df, 'value_col', 'group_col')

        val_col = str(self.get_property('value_col') or 'value')
        grp_col = str(self.get_property('group_col') or '') or None
        palette = str(self.get_property('palette') or 'Set2')
        if palette == 'None': palette = None
        fill    = bool(self.get_property('fill'))

        if val_col not in df.columns:
            self.mark_error()
            return False, f"Column '{val_col}' not found"

        hue = grp_col if grp_col and grp_col in df.columns else None
        fig, ax = _make_fig(self)
        sns.kdeplot(data=df, x=val_col, hue=hue, palette=palette,
                    fill=fill, ax=ax)
        self.set_progress(70)
        _finalize_ax(ax, self)
        fig.tight_layout()
        self.output_values['plot'] = FigureData(payload=fig)
        self.set_progress(100)
        self.mark_clean()
        return True, None


# ===========================================================================
# XYLinePlotNode  — classic Prism XY graph with error bars
# ===========================================================================

class XYLinePlotNode(BaseExecutionNode):
    """
    Creates an XY line plot with error bars in the classic Prism graph style.

    Groups the data by an optional group column, computes mean +/- error
    per unique X value, and connects the means with lines. Optionally
    overlays individual data points and accepts a stats table from
    PairwiseComparisonNode for significance-bracket overlays.

    Columns:
    - **x_col** — numeric or categorical column for the X axis
    - **y_col** — numeric column for the Y axis
    - **group_col** — optional column to split data into separate lines

    Options:
    - *error_type* — error bar measure: `SEM`, `SD`, `95% CI`, or `None`
    - *show_points* — overlay individual data points
    - *x_order* — comma-separated order for X axis categories
    - *palette* — colour palette for groups

    Keywords: XY plot, line graph, error bar, mean SEM SD CI, dose-response, time course, grouped line, connected means, 折線圖, 誤差棒, 均值, 標準誤, 劑量反應, 時間序列
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'XY Line Plot'
    PORT_SPEC      = {'inputs': ['table', 'stat'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('data',  color=PORT_COLORS['table'])
        self.add_input('stats', color=PORT_COLORS['stat'])
        self.add_output('plot', color=PORT_COLORS['figure'])

        self._add_column_selector('x_col',     'X Column',               text='', mode='single')
        self._add_column_selector('y_col',     'Y Column',               text='', mode='single')
        self._add_column_selector('group_col', 'Group Column (opt.)',    text='', mode='single')
        self.add_combo_menu('error_type', 'Error Type',
                            items=['SEM', 'SD', '95% CI', 'None'])
        self.add_text_input('x_order',   'X Order (comma-sep, opt.)', text='')
        self.add_checkbox('show_points', '', text='Show Individual Points', state=True)
        self.add_combo_menu('palette', 'Color Palette',
                            items=['Set2', 'tab10', 'colorblind', 'husl', 'dark', 'None'])
        self.add_text_input('x_label',    'X Label',    text='')
        self.add_text_input('y_label',    'Y Label',    text='')
        self.add_text_input('plot_title', 'Title',      text='')

        import NodeGraphQt as _nq
        H = _nq.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('line_width',    1.8,   widget_type=H)
        self.create_property('line_style',    '-',   widget_type=H)
        self.create_property('marker_size',   5,     widget_type=H)
        self.create_property('marker',        'o',   widget_type=H)
        self.create_property('capsize',       4.0,   widget_type=H)
        self.create_property('fig_width',     8.0,   widget_type=H)
        self.create_property('fig_height',    6.0,   widget_type=H)
        self.create_property('tick_rotation', 0.0,   widget_type=H)
        _add_stat_hidden_props(self)

    def evaluate(self):
        self.reset_progress()
        import seaborn as sns
        sns.set_theme(style='ticks')

        df = _read_table_port(self, 'data')
        if df is None:
            self.mark_error(); return False, "No data connected"

        self._refresh_column_selectors(df, 'x_col', 'y_col', 'group_col')

        stat_df    = _read_table_port(self, 'stats')
        x_col      = str(self.get_property('x_col')     or '').strip() or None
        y_col      = str(self.get_property('y_col')     or '').strip() or None
        group_col  = str(self.get_property('group_col') or '').strip() or None
        error_type = str(self.get_property('error_type') or 'SEM')
        x_order_s  = str(self.get_property('x_order')   or '').strip()
        show_pts   = bool(self.get_property('show_points'))
        palette    = str(self.get_property('palette') or 'Set2')
        if palette == 'None':
            palette = None

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not x_col or x_col not in df.columns:
            x_col = num_cols[0] if num_cols else None
        if not y_col or y_col not in df.columns:
            y_col = num_cols[1] if len(num_cols) > 1 else None
        if not x_col or not y_col:
            self.mark_error(); return False, "Need X and Y columns"
        if group_col and group_col not in df.columns:
            group_col = None

        try:
            self.set_progress(20)
            colors  = sns.color_palette(palette) if palette else sns.color_palette()
            fig, ax = _make_fig(self)

            groups  = df[group_col].dropna().unique().tolist() if group_col else [None]
            x_order = [s.strip() for s in x_order_s.split(',') if s.strip()] if x_order_s else None

            group_to_xi     = {}
            y_max_per_group = {}

            for gi, grp in enumerate(groups):
                sub = (df[df[group_col] == grp][[x_col, y_col]].dropna()
                       if group_col else df[[x_col, y_col]].dropna())
                color = colors[gi % len(colors)]
                label = str(grp) if group_col else None

                gs = sub.groupby(x_col)[y_col].agg(['mean', 'std', 'count']).reset_index()
                gs.columns = [x_col, 'mean', 'std', 'n']

                if x_order:
                    try:
                        gs[x_col] = pd.Categorical(gs[x_col], categories=x_order, ordered=True)
                        gs = gs.sort_values(x_col)
                    except Exception:
                        pass

                xs   = gs[x_col].values
                ys   = gs['mean'].values
                stds = gs['std'].fillna(0).values
                ns   = gs['n'].values

                if error_type == 'SEM':
                    errs = stds / np.sqrt(np.maximum(ns, 1))
                elif error_type == 'SD':
                    errs = stds
                elif error_type == '95% CI':
                    from scipy.stats import t as _t
                    errs = np.array([
                        _t.ppf(0.975, max(n - 1, 1)) * (s / np.sqrt(max(n, 1)))
                        for s, n in zip(stds, ns)
                    ])
                else:
                    errs = None

                try:
                    xs_num   = xs.astype(float)
                    sort_idx = np.argsort(xs_num)
                    xs_p = xs_num[sort_idx]
                    ys_p = ys[sort_idx]
                    ep   = errs[sort_idx] if errs is not None else None
                    if show_pts:
                        sub_s = sub.sort_values(x_col)
                        ax.scatter(sub_s[x_col].astype(float), sub_s[y_col],
                                   color=color, alpha=0.35, s=18, zorder=2)
                    numeric_x = True
                except (ValueError, TypeError):
                    xs_p      = np.arange(len(xs))
                    ys_p      = ys
                    ep        = errs
                    rot       = float(self.get_property('tick_rotation') or 0)
                    ax.set_xticks(xs_p)
                    ax.set_xticklabels(xs, rotation=rot,
                                       ha='right' if rot else 'center')
                    if show_pts:
                        for xi, xv in enumerate(xs):
                            pts = sub[sub[x_col] == xv][y_col].values
                            ax.scatter([xi] * len(pts), pts,
                                       color=color, alpha=0.35, s=18, zorder=2)
                    numeric_x = False

                _lw  = float(self.get_property('line_width') or 1.8)
                _ls  = str(self.get_property('line_style') or '-')
                _ms  = int(self.get_property('marker_size') or 5)
                _mk  = str(self.get_property('marker') or 'o')
                _mk  = _mk if _mk != 'None' else ''
                _cap = float(self.get_property('capsize') or 4.0)
                if ep is not None:
                    ax.errorbar(xs_p, ys_p, yerr=ep, color=color, label=label,
                                marker=_mk, markersize=_ms, linewidth=_lw,
                                linestyle=_ls, capsize=_cap, capthick=max(1, _lw * 0.8),
                                elinewidth=max(1, _lw * 0.8), zorder=3)
                else:
                    ax.plot(xs_p, ys_p, color=color, label=label,
                            marker=_mk, markersize=_ms, linewidth=_lw,
                            linestyle=_ls, zorder=3)

                if not numeric_x:
                    for xi, xv in enumerate(xs):
                        group_to_xi[str(xv)] = xi
                        hi = float(ys_p[xi] + (ep[xi] if ep is not None else 0))
                        y_max_per_group[str(xv)] = hi

            self.set_progress(75)

            if stat_df is not None and group_to_xi:
                err = _check_group_stat_df(stat_df)
                if err:
                    self.mark_error(); return False, err
                sp = _stat_props(self)
                _draw_stat_brackets(
                    ax, stat_df, group_to_xi, y_max_per_group,
                    sp['y_offset'], sp['show_ns'],
                    sp['line_color'], sp['line_width'],
                    sp['text_color'], sp['text_size'],
                    sp['label_mode'],
                )

            if group_col and len(groups) > 1:
                ax.legend(title=group_col, frameon=False)
            _finalize_ax(ax, self)
            fig.tight_layout()
            self.output_values['plot'] = FigureData(payload=fig)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error(); return False, str(e)


# ===========================================================================
# HeatmapNode
# ===========================================================================

class HeatmapNode(BaseExecutionNode):
    """
    Creates a heatmap with optional hierarchical clustering of rows and/or columns.

    Supports value annotations inside cells and a wide range of colour
    maps. Input can be a correlation matrix, gene-expression matrix, or
    any numeric table.

    Columns:
    - **row_label_col** — optional column to use as row labels
    - **value_cols** — comma-separated numeric columns (blank = all numeric)

    Options:
    - *cluster_rows* — apply hierarchical clustering to rows
    - *cluster_cols* — apply hierarchical clustering to columns
    - *annotate* — show numeric values inside cells
    - *cmap* — colour map (e.g. `viridis`, `coolwarm`, `RdYlGn`)

    Keywords: heatmap, clustermap, clustering, expression matrix, correlation, colour map, annotation, hierarchical, dendrogram, 熱圖, 聚類分析, 相關矩陣, 表達矩陣, 樹狀圖
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'Heatmap'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('data',  color=PORT_COLORS['table'])
        self.add_output('plot', color=PORT_COLORS['figure'])

        self._add_column_selector('row_label_col', 'Row Label Column (opt.)', text='', mode='single')
        self._add_column_selector('value_cols', 'Value Columns (blank=all numeric)', text='', mode='multi')
        self.add_combo_menu('cmap', 'Colormap',
                            items=['viridis', 'RdYlGn', 'coolwarm', 'seismic',
                                   'Blues', 'Reds', 'Greens', 'Purples', 'Oranges',
                                   'YlOrRd', 'YlGnBu', 'PuBuGn', 'PuOr', 'vlag', 'Spectral',
                                   'bwr', 'RdBu', 'PRGn', 'PiYG', 'magma', 'inferno', 'plasma', 'cividis'])
        self.add_checkbox('cluster_rows', '', text='Cluster Rows',    state=False)
        self.add_checkbox('cluster_cols', '', text='Cluster Columns', state=False)
        self.add_checkbox('annotate',     '', text='Show Values',     state=True)
        self.add_text_input('annot_fmt',  'Value Format',          text='.2f')
        self.add_text_input('vmin',       'Color Min (blank=auto)', text='')
        self.add_text_input('vmax',       'Color Max (blank=auto)', text='')
        self.add_text_input('plot_title', 'Title',                  text='')

        import NodeGraphQt as _nq
        H = _nq.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('fig_width',     10.0, widget_type=H)
        self.create_property('fig_height',     8.0, widget_type=H)
        self.create_property('tick_rotation', 45.0, widget_type=H)

    def evaluate(self):
        self.reset_progress()
        import seaborn as sns
        sns.set_theme(style='white')

        df = _read_table_port(self, 'data')
        if df is None:
            self.mark_error(); return False, "No data connected"

        self._refresh_column_selectors(df, 'row_label_col', 'value_cols')

        row_lbl  = str(self.get_property('row_label_col') or '').strip() or None
        vcols_s  = str(self.get_property('value_cols')    or '').strip()
        cmap     = str(self.get_property('cmap')          or 'viridis')
        clust_r  = bool(self.get_property('cluster_rows'))
        clust_c  = bool(self.get_property('cluster_cols'))
        annotate = bool(self.get_property('annotate'))
        vmin_s   = str(self.get_property('vmin') or '').strip()
        vmax_s   = str(self.get_property('vmax') or '').strip()
        title    = str(self.get_property('plot_title') or '')
        rot      = float(self.get_property('tick_rotation') or 45)
        fig_w    = float(self.get_property('fig_width')     or 10)
        fig_h    = float(self.get_property('fig_height')    or 8)

        try:
            self.set_progress(20)
            if vcols_s:
                vcols = [c.strip() for c in vcols_s.split(',') if c.strip() in df.columns]
            else:
                vcols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not vcols:
                self.mark_error(); return False, "No numeric columns found"

            mat  = df[vcols].copy()
            if row_lbl and row_lbl in df.columns:
                mat.index = df[row_lbl].values

            vmin = float(vmin_s) if vmin_s else None
            vmax = float(vmax_s) if vmax_s else None
            fmt  = str(self.get_property('annot_fmt') or '.2f').strip() if annotate else ''

            self.set_progress(40)
            if clust_r or clust_c:
                g = sns.clustermap(
                    mat, cmap=cmap, annot=annotate, fmt=fmt,
                    row_cluster=clust_r, col_cluster=clust_c,
                    vmin=vmin, vmax=vmax,
                    figsize=(fig_w, fig_h),
                    xticklabels=True, yticklabels=True,
                )
                g.ax_heatmap.tick_params(axis='x', rotation=rot)
                if title:
                    g.ax_heatmap.set_title(title, fontweight='bold', pad=10)
                fig = g.fig
            else:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                fig = Figure(figsize=(fig_w, fig_h))
                FigureCanvasAgg(fig)
                ax = fig.add_subplot(111)
                sns.heatmap(mat, cmap=cmap, annot=annotate, fmt=fmt,
                            vmin=vmin, vmax=vmax, ax=ax,
                            xticklabels=True, yticklabels=True)
                ax.tick_params(axis='x', rotation=rot)
                ax.set_title(title, fontweight='bold')
                for sp in ['top', 'right', 'left', 'bottom']:
                    ax.spines[sp].set_visible(False)
                fig.tight_layout()

            self.output_values['plot'] = FigureData(payload=fig)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error(); return False, str(e)


# ===========================================================================
# VolcanoPlotNode
# ===========================================================================

class VolcanoPlotNode(BaseExecutionNode):
    """
    Creates a volcano plot showing log2(fold change) vs -log10(p-value).

    Colours up-regulated, down-regulated, and non-significant points
    separately, draws fold-change and significance threshold lines, and
    optionally labels the top N most significant features. Also outputs
    the significant-hit rows as a table for downstream filtering.

    Columns:
    - **fc_col** — column containing log2 fold-change values
    - **p_col** — column containing p-values
    - **label_col** — optional column for feature labels

    Parameters:
    - **fc_thresh** — fold-change threshold (`|log2FC|`)
    - **p_thresh** — p-value significance cutoff
    - **n_labels** — number of top significant features to label (0 = none)
    - **point_size** — scatter point size

    Keywords: volcano plot, fold change, log2FC, p-value, adjusted p-value, differential expression, omics, significance, up-regulated, down-regulated, 火山圖, 差異表達, 統計顯著性, 倍數變化
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'Volcano Plot'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['figure', 'table']}

    def __init__(self):
        super().__init__()
        self.add_input('data',        color=PORT_COLORS['table'])
        self.add_output('plot',       color=PORT_COLORS['figure'])
        self.add_output('significant', color=PORT_COLORS['table'])

        self._add_column_selector('fc_col',    'Log2 FC Column',           text='', mode='single')
        self._add_column_selector('p_col',     'p-value Column',           text='', mode='single')
        self._add_column_selector('label_col', 'Label Column (opt.)',      text='', mode='single')
        
        self._add_float_spinbox('fc_thresh', 'FC Threshold (|log2FC|)', value=1.0, step=0.1)
        self._add_float_spinbox('p_thresh',  'p-value Threshold',       value=0.05, step=0.01, decimals=4)
        self._add_int_spinbox('n_labels',    'Top N Labels (0=none)',   value=10)
        self._add_int_spinbox('label_fontsize', 'Label Font Size',      value=8)
        self._add_int_spinbox('point_size',     'Point Size',           value=18)

        self.add_text_input('x_label',   'X Label',  text='log2(Fold Change)')
        self.add_text_input('y_label',   'Y Label',  text='-log10(p-value)')
        self.add_text_input('plot_title', 'Title',   text='Volcano Plot')

        import NodeGraphQt as _nq
        H = _nq.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('fig_width',     8.0, widget_type=H)
        self.create_property('fig_height',    7.0, widget_type=H)
        self.create_property('fig_dpi',       100.0, widget_type=H)
        self.create_property('tick_rotation', 0.0, widget_type=H)
        self.create_property('color_up',   [220,  50,  50, 255], widget_type=H)
        self.create_property('color_down', [ 50, 120, 220, 255], widget_type=H)
        self.create_property('color_ns',   [180, 180, 180, 255], widget_type=H)

    def evaluate(self):
        self.reset_progress()

        df = _read_table_port(self, 'data')
        if df is None:
            self.mark_error(); return False, "No data connected"

        self._refresh_column_selectors(df, 'fc_col', 'p_col', 'label_col')

        fc_col    = str(self.get_property('fc_col')    or '').strip() or None
        p_col     = str(self.get_property('p_col')     or '').strip() or None
        label_col = str(self.get_property('label_col') or '').strip() or None

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not fc_col or fc_col not in df.columns:
            fc_col = num_cols[0] if num_cols else None
        if not p_col or p_col not in df.columns:
            p_col = num_cols[1] if len(num_cols) > 1 else None
        if not fc_col or not p_col:
            self.mark_error(); return False, "Need fold-change and p-value columns"

        fc_thresh = float(self.get_property('fc_thresh') or 1.0)
        p_thresh  = float(self.get_property('p_thresh')  or 0.05)
        n_labels  = int(self.get_property('n_labels')    or 10)
        label_fs  = int(self.get_property('label_fontsize') or 8)
        point_sz  = int(self.get_property('point_size')     or 18)

        try:
            self.set_progress(20)
            df_c = df.copy().dropna(subset=[fc_col, p_col])
            df_c[p_col] = pd.to_numeric(df_c[p_col], errors='coerce')
            df_c = df_c[df_c[p_col] > 0]
            df_c['_log_p'] = -np.log10(df_c[p_col])
            df_c['_fc']    =  df_c[fc_col].astype(float)

            up_m   = (df_c['_fc'] >=  fc_thresh) & (df_c[p_col] <= p_thresh)
            down_m = (df_c['_fc'] <= -fc_thresh) & (df_c[p_col] <= p_thresh)
            ns_m   = ~(up_m | down_m)
            
            self.set_progress(40)
            fig, ax = _make_fig(self)
            c_up   = _parse_color(self.get_property('color_up'),   '#DC3232')
            c_down = _parse_color(self.get_property('color_down'), '#3278DC')
            c_ns   = _parse_color(self.get_property('color_ns'),   '#B4B4B4')

            ax.scatter(df_c.loc[ns_m,   '_fc'], df_c.loc[ns_m,   '_log_p'],
                       c=[c_ns],   s=point_sz * 0.6, alpha=0.5, linewidths=0, label='n.s.')
            ax.scatter(df_c.loc[down_m, '_fc'], df_c.loc[down_m, '_log_p'],
                       c=[c_down], s=point_sz, alpha=0.8, linewidths=0, label='Down')
            ax.scatter(df_c.loc[up_m,   '_fc'], df_c.loc[up_m,   '_log_p'],
                       c=[c_up],   s=point_sz, alpha=0.8, linewidths=0, label='Up')
            
            ax.axhline(-np.log10(p_thresh), color='#444', linestyle='--',
                       linewidth=0.9, alpha=0.65)
            ax.axvline( fc_thresh, color='#444', linestyle='--', linewidth=0.9, alpha=0.65)
            ax.axvline(-fc_thresh, color='#444', linestyle='--', linewidth=0.9, alpha=0.65)

            if n_labels > 0 and label_col and label_col in df_c.columns:
                sig = df_c[up_m | down_m].nlargest(n_labels, '_log_p')
                texts = []
                for _, row in sig.iterrows():
                    texts.append(ax.text(float(row['_fc']), float(row['_log_p']),
                                         str(row[label_col]), fontsize=label_fs,
                                         ha='left', va='bottom'))
                
                if texts:
                    try:
                        from adjustText import adjust_text
                        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=ax)
                    except ImportError:
                        pass

            self.set_progress(80)
            ax.legend(frameon=False, markerscale=1.5)
            _finalize_ax(ax, self)
            fig.tight_layout()

            sig_df = (df_c[up_m | down_m]
                      .drop(columns=['_log_p', '_fc'])
                      .reset_index(drop=True))
            self.output_values['plot']        = FigureData(payload=fig)
            self.output_values['significant'] = TableData(payload=sig_df)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error(); return False, str(e)


# ===========================================================================
# RegressionPlotNode
# ===========================================================================

class RegressionPlotNode(BaseExecutionNode):
    """
    Creates a scatter plot with a fitted regression line and optional 95% confidence band.

    Optionally accepts a pre-computed curve table from
    NonlinearRegressionNode to overlay a custom fit. For simple linear
    fits the equation and R-squared are annotated on the plot
    automatically.

    Columns:
    - **x_col** — numeric column for the X axis
    - **y_col** — numeric column for the Y axis
    - **group_col** — optional column for per-group fits

    Options:
    - *fit_type* — auto-fit when no curve input: `Linear`, `Polynomial deg 2`, `Polynomial deg 3`, or `None`
    - *show_ci* — show 95% confidence band around the fit
    - *show_equation* — annotate with equation and R-squared
    - *palette* — colour palette for groups

    Keywords: regression plot, scatter fit, confidence interval, linear fit, nonlinear fit, R-squared, fitted line, correlation scatter, 回歸圖, 散點擬合, 置信區間, 相關圖, R平方
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'Regression Plot'
    PORT_SPEC      = {'inputs': ['table', 'table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('data',  color=PORT_COLORS['table'])
        self.add_input('curve', color=PORT_COLORS['table'])
        self.add_output('plot', color=PORT_COLORS['figure'])

        self._add_column_selector('x_col',     'X Column',            text='', mode='single')
        self._add_column_selector('y_col',     'Y Column',            text='', mode='single')
        self._add_column_selector('group_col', 'Group Column (opt.)', text='', mode='single')
        self.add_combo_menu('fit_type', 'Auto-Fit (no curve input)',
                            items=['Linear', 'Polynomial deg 2',
                                   'Polynomial deg 3', 'None'])
        self.add_checkbox('show_ci',       '', text='Show 95% Confidence Band', state=True)
        self.add_checkbox('show_equation', '', text='Show Equation / R²',       state=True)
        self.add_combo_menu('palette', 'Color Palette',
                            items=['Set2', 'tab10', 'colorblind', 'husl', 'None'])
        self.add_text_input('x_label',    'X Label',    text='')
        self.add_text_input('y_label',    'Y Label',    text='')
        self.add_text_input('plot_title', 'Title',      text='')

        import NodeGraphQt as _nq
        H = _nq.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('fig_width',     8.0, widget_type=H)
        self.create_property('fig_height',    6.0, widget_type=H)
        self.create_property('tick_rotation', 0.0, widget_type=H)
        
        # New customizable properties for the equation
        self.add_text_input('eq_x',       'Equation X Pos',        text='0.05')
        self.add_text_input('eq_y',       'Equation Y Pos',        text='0.95')
        self.add_text_input('eq_size',    'Equation Font Size',    text='9')
        self.add_text_input('eq_spacing', 'Eq / R² Line Spacing',  text='1.5')

    def evaluate(self):
        self.reset_progress()
        import seaborn as sns
        sns.set_theme(style='ticks')

        df = _read_table_port(self, 'data')
        if df is None:
            self.mark_error(); return False, "No data connected"

        self._refresh_column_selectors(df, 'x_col', 'y_col', 'group_col')

        curve_df  = _read_table_port(self, 'curve')
        x_col     = str(self.get_property('x_col')    or '').strip() or None
        y_col     = str(self.get_property('y_col')    or '').strip() or None
        group_col = str(self.get_property('group_col')or '').strip() or None
        fit_type  = str(self.get_property('fit_type') or 'Linear')
        show_ci   = bool(self.get_property('show_ci'))
        show_eq   = bool(self.get_property('show_equation'))
        eq_x      = float(self.get_property('eq_x') or 0.05)
        eq_y      = float(self.get_property('eq_y') or 0.95)
        eq_size   = float(self.get_property('eq_size') or 9)
        eq_space  = float(self.get_property('eq_spacing') or 1.5)
        palette   = str(self.get_property('palette')  or 'Set2')
        if palette == 'None':
            palette = None

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not x_col or x_col not in df.columns:
            x_col = num_cols[0] if num_cols else None
        if not y_col or y_col not in df.columns:
            y_col = num_cols[1] if len(num_cols) > 1 else None
        if not x_col or not y_col:
            self.mark_error(); return False, "Need X and Y columns"
        if group_col and group_col not in df.columns:
            group_col = None

        try:
            self.set_progress(20)
            colors  = sns.color_palette(palette) if palette else sns.color_palette()
            fig, ax = _make_fig(self)
            groups  = df[group_col].dropna().unique().tolist() if group_col else [None]

            for gi, grp in enumerate(groups):
                sub   = (df[df[group_col] == grp][[x_col, y_col]].dropna()
                         if group_col else df[[x_col, y_col]].dropna())
                xs    = sub[x_col].astype(float).values
                ys    = sub[y_col].astype(float).values
                color = colors[gi % len(colors)]
                label = str(grp) if group_col else None

                ax.scatter(xs, ys, color=color, alpha=0.55, s=28,
                           label=label, zorder=2)

                if curve_df is not None and gi == 0:
                    # External curve provided — draw line and CI if available
                    cx = curve_df.iloc[:, 0].values
                    cy = curve_df.iloc[:, 1].values
                    ax.plot(cx, cy, color=color, linewidth=2.2, zorder=3)

                    # Use pre-computed CI columns if present (from regression nodes)
                    ci_lo_col = [c for c in curve_df.columns if c.endswith('_ci_lo')]
                    ci_hi_col = [c for c in curve_df.columns if c.endswith('_ci_hi')]
                    if show_ci and ci_lo_col and ci_hi_col:
                        ax.fill_between(cx,
                                        curve_df[ci_lo_col[0]].values,
                                        curve_df[ci_hi_col[0]].values,
                                        alpha=0.14, color=color)

                    # Compute R² from raw data vs curve interpolation
                    sort_idx = np.argsort(xs)
                    xs_s, ys_s = xs[sort_idx], ys[sort_idx]
                    y_interp = np.interp(xs_s, cx, cy)
                    ss_res = float(np.sum((ys_s - y_interp) ** 2))
                    ss_tot = float(np.sum((ys_s - ys_s.mean()) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

                    if show_eq:
                        eq = f'R\u00b2 = {r2:.4f}'
                        ax.text(eq_x, eq_y, eq, transform=ax.transAxes,
                                fontsize=eq_size,
                                linespacing=eq_space,
                                va='top',
                                bbox=dict(boxstyle='round,pad=0.3',
                                          fc='white', alpha=0.85))

                elif fit_type != 'None':
                    sort_idx = np.argsort(xs)
                    xs_s, ys_s = xs[sort_idx], ys[sort_idx]
                    x_range    = np.linspace(xs_s.min(), xs_s.max(), 300)
                    deg = (1 if fit_type == 'Linear'
                           else 2 if 'deg 2' in fit_type else 3)

                    coeffs = np.polyfit(xs_s, ys_s, deg)
                    y_fit  = np.polyval(coeffs, x_range)
                    y_pred = np.polyval(coeffs, xs_s)
                    ss_res = float(np.sum((ys_s - y_pred) ** 2))
                    ss_tot = float(np.sum((ys_s - ys_s.mean()) ** 2))
                    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

                    ax.plot(x_range, y_fit, color=color, linewidth=2.2, zorder=3)

                    if show_ci and len(xs_s) > deg + 2:
                        try:
                            from scipy.stats import t as _t
                            n_p    = len(xs_s)
                            s_err  = float(np.sqrt(ss_res / max(n_p - deg - 1, 1)))
                            x_mean = xs_s.mean()
                            ss_xx  = float(np.sum((xs_s - x_mean) ** 2))
                            t_c    = _t.ppf(0.975, n_p - deg - 1)
                            se_l   = s_err * np.sqrt(
                                1 / n_p + (x_range - x_mean) ** 2 / max(ss_xx, 1e-300)
                            )
                            ax.fill_between(x_range,
                                            y_fit - t_c * se_l,
                                            y_fit + t_c * se_l,
                                            alpha=0.14, color=color)
                        except Exception:
                            pass

                    if show_eq and deg == 1:
                        # Format the equation handling negative constants explicitly
                        slope, intercept = coeffs[0], coeffs[1]

                        if intercept < 0:
                            eq_str = f'y = {slope:.3g}x \u2212 {abs(intercept):.3g}'
                        else:
                            eq_str = f'y = {slope:.3g}x + {intercept:.3g}'

                        eq = f'{eq_str}\nR\u00b2 = {r2:.4f}'

                        ax.text(eq_x, eq_y, eq, transform=ax.transAxes,
                                fontsize=eq_size,
                                linespacing=eq_space,
                                va='top',
                                bbox=dict(boxstyle='round,pad=0.3',
                                          fc='white', alpha=0.85))

            self.set_progress(80)
            if group_col:
                ax.legend(title=group_col, frameon=False)
            _finalize_ax(ax, self)
            fig.tight_layout()
            self.output_values['plot'] = FigureData(payload=fig)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error(); return False, str(e)


# ===========================================================================
# SurvivalPlotNode
# ===========================================================================

class SurvivalPlotNode(BaseExecutionNode):
    """
    Draws Kaplan-Meier survival curves from SurvivalAnalysisNode output.

    Accepts the `km_table` output and draws survival step-function curves
    with optional 95% CI shading, censoring tick marks, and an automatic
    log-rank p-value annotation from the `log_rank` port.

    Inputs:
    - `km_table` — Kaplan-Meier table with time, survival, and group columns
    - `log_rank` — StatData with overall log-rank test result
    - `pairwise_stat` — optional pairwise comparison table

    Options:
    - *show_ci* — shade the 95% confidence interval around each curve
    - *show_censored* — draw tick marks at censoring events
    - *show_pairwise* — display pairwise log-rank comparisons on the plot
    - *palette* — colour palette for groups

    Keywords: kaplan meier, survival curve, KM plot, censored data, time-to-event, step function, log-rank, hazard, 生存曲線, 存活分析, 卡普蘭-邁耶, 截尾, 對數秩檢定
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'Survival Plot'
    PORT_SPEC      = {'inputs': ['table', 'stat', 'table'], 'outputs': ['figure']}

    def __init__(self):
        super().__init__()
        self.add_input('km_table', color=PORT_COLORS['table'])
        self.add_input('log_rank', color=PORT_COLORS['stat'])
        self.add_input('pairwise_stat', color=PORT_COLORS['table'])
        self.add_output('plot',    color=PORT_COLORS['figure'])

        self.add_checkbox('show_ci',       '', text='Show 95% CI',          state=True)
        self.add_checkbox('show_censored', '', text='Show Censoring Marks', state=True)
        self.add_checkbox('show_pairwise', '', text='Show Pairwise Stats', state=True)
        
        locs = ['upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center']
        self.add_combo_menu('pairwise_loc', 'Pairwise Stats Location', items=locs)
        self._add_float_spinbox('pairwise_x_offset', 'Pairwise X Offset', value=0.0, step=0.01)
        self._add_float_spinbox('pairwise_y_offset', 'Pairwise Y Offset', value=0.0, step=0.01)
        self._add_int_spinbox('pairwise_fontsize', 'Pairwise Font Size', value=9, min_val=4, max_val=32)
        
        self.add_combo_menu('palette', 'Color Palette',
                            items=['Set2', 'tab10', 'colorblind', 'husl', 'None'])
        self.add_text_input('x_label',    'X Label', text='Time')
        self.add_text_input('y_label',    'Y Label', text='Survival Probability')
        self.add_text_input('plot_title', 'Title',   text='Kaplan-Meier Curves')
        
        import NodeGraphQt as _nq
        H = _nq.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('fig_width',     8.0, widget_type=H)
        self.create_property('fig_height',    6.0, widget_type=H)
        self.create_property('tick_rotation', 0.0, widget_type=H)

    def evaluate(self):
        self.reset_progress()
        import seaborn as sns
        sns.set_theme(style='ticks')

        km_df = _read_table_port(self, 'km_table')
        if km_df is None:
            self.mark_error()
            return False, ("No KM table connected. "
                           "Connect the km_table output of SurvivalAnalysisNode.")

        pairwise_df   = _read_table_port(self, 'pairwise_stat')
        lr_df         = _read_table_port(self, 'log_rank')
        show_ci       = bool(self.get_property('show_ci'))
        show_censored = bool(self.get_property('show_censored'))
        show_pairwise = bool(self.get_property('show_pairwise'))
        pairwise_loc  = str(self.get_property('pairwise_loc') or 'lower left')
        pw_x = float(self.get_property('pairwise_x_offset') or 0.0)
        pw_y = float(self.get_property('pairwise_y_offset') or 0.0)
        try:
            pw_fs = int(self.get_property('pairwise_fontsize') or 9)
        except (ValueError, TypeError):
            pw_fs = 9
            
        palette       = str(self.get_property('palette') or 'Set2')
        if palette == 'None':
            palette = None

        try:
            self.set_progress(20)
            has_group = 'Group' in km_df.columns
            fig, ax   = _make_fig(self)
            colors    = sns.color_palette(palette) if palette else sns.color_palette()
            groups    = km_df['Group'].unique().tolist() if has_group else [None]

            for gi, grp in enumerate(groups):
                sub   = km_df[km_df['Group'] == grp] if has_group else km_df
                color = colors[gi % len(colors)]
                label = str(grp) if has_group else 'Overall'

                times    = sub['Time'].values
                survival = sub['Survival'].values

                ax.step(times, survival, where='post',
                        color=color, linewidth=2.2, label=label)

                if show_ci and 'Lower_95CI' in sub.columns:
                    ax.fill_between(times,
                                    sub['Lower_95CI'].values,
                                    sub['Upper_95CI'].values,
                                    step='post', alpha=0.14, color=color)

                if show_censored and 'Censored' in sub.columns:
                    cens_m = sub['Censored'].values > 0
                    if cens_m.any():
                        ax.scatter(times[cens_m], survival[cens_m],
                                   marker='+', color=color, s=55,
                                   linewidths=1.8, zorder=5)

            self.set_progress(70)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(left=0)

            if lr_df is not None and 'p-value' in lr_df.columns:
                p_val = lr_df['p-value'].iloc[0]
                if pd.notna(p_val):
                    ax.text(0.98, 0.97,
                            f'Log-rank p = {float(p_val):.4f}',
                            transform=ax.transAxes, fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3',
                                      fc='white', alpha=0.85))
                                      
            if show_pairwise and pairwise_df is not None and not pairwise_df.empty and 'group1' in pairwise_df.columns:
                from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea
                
                col1, col2, col3 = [], [], []
                for _, row in pairwise_df.iterrows():
                    p = row.get('p-adj', row.get('p-value', 1.0))
                    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    
                    col1.append(TextArea(f"{row['group1']} vs {row['group2']} ", textprops=dict(size=pw_fs)))
                    col2.append(TextArea(f"p = {p:.4g}", textprops=dict(size=pw_fs)))
                    col3.append(TextArea(stars, textprops=dict(size=pw_fs, weight='bold', color='#333333')))
                    
                if len(col1) > 9:
                    col1 = col1[:9] + [TextArea("...", textprops=dict(size=pw_fs))]
                    col2 = col2[:9] + [TextArea("...", textprops=dict(size=pw_fs))]
                    col3 = col3[:9] + [TextArea("...", textprops=dict(size=pw_fs))]
                    
                vp1 = VPacker(children=col1, align="left", pad=0, sep=4)
                vp2 = VPacker(children=col2, align="left", pad=0, sep=4)
                vp3 = VPacker(children=col3, align="left", pad=0, sep=4)
                
                table_hp = HPacker(children=[vp1, vp2, vp3], align="top", pad=0, sep=8)
                
                title = TextArea("Pairwise Log-Rank:", textprops=dict(size=pw_fs, weight='bold'))
                final_vp = VPacker(children=[title, table_hp], align="left", pad=0, sep=6)
                
                at = AnchoredOffsetbox(loc=pairwise_loc, child=final_vp, frameon=True,
                                       bbox_to_anchor=(pw_x, pw_y, 1, 1),
                                       bbox_transform=ax.transAxes,
                                       borderpad=0.5)
                at.patch.set_boxstyle("round,pad=0.3")
                at.patch.set_facecolor("white")
                at.patch.set_edgecolor("#cccccc")
                at.patch.set_alpha(0.85)
                at.set_zorder(15)
                ax.add_artist(at)

            if has_group:
                ax.legend(title='Group', frameon=False)

            _finalize_ax(ax, self)
            fig.tight_layout()
            self.output_values['plot'] = FigureData(payload=fig)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error(); return False, str(e)


# ===========================================================================
# AnglePlotNode
# ===========================================================================

class AnglePlotNode(PlotToolboxMixin, BaseExecutionNode):
    """
    Creates a polar angle distribution plot for angular data.

    Display modes:
    - *Bin Arrows* — each angular bin is drawn as a proportional arrow from the origin (length = normalised bin frequency)
    - *KDE* — smooth kernel-density fill across the defined angular range
    - *Both* — overlay KDE on top of bin arrows

    The angular range is fully user-defined via **theta_min** / **theta_max**
    (degrees). Common presets: 0--90 (fibre orientation), 0--180, 0--360
    (full circle).

    Columns:
    - **angle_col** — column containing angle values
    - **group_col** — optional column for per-group curves in distinct colours

    Input angles may be in *Degrees* or *Radians* (set via **input_unit**).

    Keywords: angle plot, polar plot, orientation, fibre angle, direction, circular, KDE, bin arrows, distribution, 0-360, 0-180, 0-90, 角度圖, 極座標, 分佈, 方向性, 纖維角, 核密度
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME      = 'Angle Distribution Plot'
    PORT_SPEC      = {'inputs': ['table'], 'outputs': ['figure']}

    _PALETTES = ['Set2', 'tab10', 'husl', 'colorblind', 'pastel', 'muted', 'Dark2']

    def __init__(self):
        self._toolbox_widgets = {}
        super().__init__()
        self.add_input('data', color=PORT_COLORS['table'])
        self.add_output('plot', color=PORT_COLORS['figure'])

        import NodeGraphQt
        H = NodeGraphQt.constants.NodePropWidgetEnum.HIDDEN.value
        self.create_property('angle_col',    '',         widget_type=H)
        self.create_property('group_col',    '',         widget_type=H)
        self.create_property('input_unit',   'Degrees',  widget_type=H)
        self.create_property('plot_mode',    'KDE',      widget_type=H)
        self.create_property('theta_min',    0.0,        widget_type=H)
        self.create_property('theta_max',    360.0,      widget_type=H)
        self.create_property('bin_width',    10.0,       widget_type=H)
        self.create_property('kde_bw_scale', 1.0,        widget_type=H)
        self.create_property('arrow_width',  0.8,        widget_type=H)
        self.create_property('arrow_head',   8.0,        widget_type=H)
        self.create_property('fill_alpha',   0.30,       widget_type=H)
        self.create_property('line_width',   2.0,        widget_type=H)
        self.create_property('palette',      'Set2',     widget_type=H)
        self.create_property('show_grid',    True,       widget_type=H)
        self.create_property('plot_title',   '',         widget_type=H)
        self.create_property('fig_width',    5.5,        widget_type=H)
        self.create_property('fig_height',   5.5,        widget_type=H)

        self._build_toolbox(520)
        # ── Data page ─────────────────────────────────────────────────────────
        self._tb_column_selector('angle_col',    'Angle Column',        'Data', '')
        self._tb_column_selector('group_col',    'Group Column (opt)',  'Data', '')
        self._tb_combo('input_unit',  'Input Unit',          'Data',
                       ['Degrees', 'Radians'])
        self._tb_combo('plot_mode',   'Plot Mode',           'Data',
                       ['KDE', 'Bin Arrows', 'Both'])
        self._tb_spinbox('theta_min',   'Theta Min (°)',     'Data',
                         0.0,  min_val=0.0,   max_val=360.0, step=5.0,  decimals=1)
        self._tb_spinbox('theta_max',   'Theta Max (°)',     'Data',
                         360.0, min_val=1.0,  max_val=360.0, step=5.0,  decimals=1)
        self._tb_spinbox('bin_width',   'Bin Width (°)',     'Data',
                         10.0, min_val=1.0,   max_val=90.0,  step=1.0,  decimals=1)
        self._tb_spinbox('kde_bw_scale','KDE BW Scale',      'Data',
                         1.0,  min_val=0.1,   max_val=20.0,  step=0.25, decimals=2)
        self._tb_spinbox('arrow_width', 'Arrow Width',       'Data',
                         0.8,  min_val=0.1,   max_val=8.0,   step=0.1,  decimals=2)
        self._tb_spinbox('arrow_head',  'Arrow Head Size',   'Data',
                         8.0,  min_val=0.0,   max_val=30.0,  step=0.5,  decimals=1)
        self._tb_spinbox('fill_alpha',  'Fill Alpha',        'Data',
                         0.30, min_val=0.0,   max_val=1.0,   step=0.05, decimals=2)
        self._tb_spinbox('line_width',  'Line Width',        'Data',
                         2.0,  min_val=0.2,   max_val=10.0,  step=0.2,  decimals=2)
        self._tb_combo('palette',    'Palette',              'Data', self._PALETTES)
        self._tb_checkbox('show_grid', 'Show Grid',          'Data', True)
        self._tb_text('plot_title',  'Title',                'Data', '')
        self._tb_add_figure_page()

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_to_range(values, unit: str, th_min: float, th_max: float):
        """Convert angles to degrees and clamp/fold into [th_min, th_max]."""
        import numpy as np
        arr = np.asarray(values, dtype=float)
        if unit == 'Radians':
            arr = np.rad2deg(arr)
        # Wrap into the target range using modulo of the span
        span = th_max - th_min
        if span <= 0:
            span = 360.0
        arr = ((arr - th_min) % span) + th_min
        return arr[np.isfinite(arr)]

    @staticmethod
    def _bin_arrows(deg_arr, bin_width, th_min, th_max):
        """Returns (bin_center_rad, normalised_proportion) for occupied bins."""
        import numpy as np
        bins     = np.arange(th_min, th_max + bin_width, bin_width)
        if len(bins) < 2:
            return np.array([]), np.array([])
        idxes    = np.clip(np.digitize(deg_arr, bins, right=True) - 1,
                           0, len(bins) - 2)
        counts   = np.zeros(len(bins) - 1)
        for i in idxes:
            counts[i] += 1
        total = counts.sum()
        if total > 0:
            counts /= total
        ctr_deg  = (bins[:-1] + bins[1:]) / 2.0
        occupied = counts > 0
        return np.deg2rad(ctr_deg[occupied]), counts[occupied]

    @staticmethod
    def _kde_density(deg_arr, bw_scale, th_min, th_max):
        """KDE with reflection at both theta boundaries → normalised to [0,1]."""
        import numpy as np
        from scipy.stats import gaussian_kde
        span = th_max - th_min
        # Reflection enforces boundary conditions without edge artefacts
        reflected = np.concatenate([
            deg_arr,
            th_min - (deg_arr - th_min),   # reflect at lower bound
            th_max + (th_max - deg_arr),   # reflect at upper bound
        ])
        kde = gaussian_kde(reflected)
        kde.set_bandwidth(kde.scotts_factor() * float(bw_scale))
        grid_deg = np.linspace(th_min, th_max, max(361, int(span * 4 + 1)))
        dens     = kde(grid_deg)
        dens     = np.clip(dens, 0, None)
        m = dens.max()
        if m > 0:
            dens /= m
        return np.deg2rad(grid_deg), dens

    # ── evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self):
        self.reset_progress()
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.lines as mlines
        import seaborn as sns

        df = _read_table_port(self, 'data')
        if df is None:
            self.mark_error(); return False, 'No data connected'
        self._tb_refresh_columns(df, 'angle_col', 'group_col')

        angle_col  = str(self.get_property('angle_col')     or '').strip()
        group_col  = str(self.get_property('group_col')     or '').strip()
        unit       = str(self.get_property('input_unit')    or 'Degrees')
        mode       = str(self.get_property('plot_mode')     or 'KDE')
        th_min     = float(self.get_property('theta_min')   or 0.0)
        th_max     = float(self.get_property('theta_max')   or 360.0)
        bin_width  = float(self.get_property('bin_width')   or 10.0)
        bw_scale   = float(self.get_property('kde_bw_scale')or 1.0)
        a_width    = float(self.get_property('arrow_width') or 0.8)
        a_head     = float(self.get_property('arrow_head')  or 8.0)
        fill_alpha = float(self.get_property('fill_alpha')  or 0.3)
        lw         = float(self.get_property('line_width')  or 2.0)
        palette    = str(self.get_property('palette')       or 'Set2')
        show_grid  = bool(self.get_property('show_grid'))
        title      = str(self.get_property('plot_title')    or '')
        fig_w      = float(self.get_property('fig_width')   or 5.5)
        fig_h      = float(self.get_property('fig_height')  or 5.5)

        if th_max <= th_min:
            self.mark_error()
            return False, 'Theta Max must be greater than Theta Min'

        # Resolve angle column
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not angle_col or angle_col not in df.columns:
            angle_col = num_cols[0] if num_cols else None
        if angle_col is None:
            self.mark_error(); return False, 'No numeric angle column found'

        # Resolve groups
        gc_valid = group_col and group_col in df.columns
        groups   = sorted(df[group_col].dropna().unique().tolist()) if gc_valid else [None]

        # Colour palette
        try:
            colours = sns.color_palette(palette, len(groups))
        except Exception:
            colours = sns.color_palette('tab10', len(groups))

        try:
            self.set_progress(20)

            fig = Figure(figsize=(fig_w, fig_h))
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111, projection='polar')
            ax.set_thetamin(th_min)
            ax.set_thetamax(th_max)
            ax.set_yticklabels([])
            ax.grid(show_grid)
            if not show_grid:
                ax.set_rticks([])

            legend_handles = []

            for gi, (grp, color) in enumerate(zip(groups, colours)):
                label   = str(grp) if grp is not None else (angle_col or 'Data')
                sub_df  = df[df[group_col] == grp] if gc_valid else df
                raw     = sub_df[angle_col].dropna().values
                if len(raw) == 0:
                    continue
                deg_arr = self._normalise_to_range(raw, unit, th_min, th_max)
                if len(deg_arr) == 0:
                    continue

                self.set_progress(20 + 60 * (gi + 1) // len(groups))

                # ── KDE ───────────────────────────────────────────────────
                if mode in ('KDE', 'Both'):
                    theta_rad, dens = self._kde_density(deg_arr, bw_scale, th_min, th_max)
                    line, = ax.plot(theta_rad, dens,
                                    color=color, linewidth=lw,
                                    label=label, zorder=3)
                    ax.fill_between(theta_rad, 0, dens,
                                    color=color, alpha=fill_alpha, zorder=2)
                    legend_handles.append(line)

                # ── Bin arrows ────────────────────────────────────────────
                if mode in ('Bin Arrows', 'Both'):
                    bin_rad, bin_cnt = self._bin_arrows(deg_arr, bin_width, th_min, th_max)
                    if bin_cnt.size == 0:
                        continue
                    max_c = bin_cnt.max()
                    norm_cnt = bin_cnt / max_c if max_c > 0 else bin_cnt
                    for th, r in zip(bin_rad, norm_cnt):
                        ax.annotate('',
                                    xytext=(0, 0), xy=(th, float(r)),
                                    xycoords='data', textcoords='data',
                                    arrowprops=dict(
                                        arrowstyle='-|>',
                                        color=color,
                                        lw=a_width,
                                        mutation_scale=a_head,
                                    ), zorder=4)
                    if mode == 'Bin Arrows':
                        legend_handles.append(
                            mlines.Line2D([], [], color=color, linewidth=a_width,
                                          label=label))

            ax.set_ylim(0, 1.05)
            if title:
                ax.set_title(title, fontweight='bold', pad=14)
            if gc_valid and legend_handles:
                ax.legend(handles=legend_handles, title=group_col, frameon=True,
                          loc='upper left', bbox_to_anchor=(1.08, 1.0))

            fig.tight_layout()
            self.output_values['plot'] = FigureData(payload=fig)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception:
            import traceback
            self.mark_error()
            return False, traceback.format_exc()


# ── Save Figure Node ─────────────────────────────────────────────────────────

class SaveFigureNode(BaseExecutionNode):
    """
    Saves a matplotlib figure to disk. Click Browse to choose file location and format.

    Inputs:
    - **figure** — FigureData to save

    Supported formats: PNG, SVG, TIFF, JPEG.
    Users can also type any path with a custom extension directly.

    Keywords: save, export, figure, plot, png, svg, tiff, write, 儲存, 匯出, 圖表
    """
    __identifier__ = 'nodes.plotting'
    NODE_NAME = 'Save Figure'
    PORT_SPEC = {'inputs': ['figure'], 'outputs': []}
    _collection_aware = True

    _EXT_FILTER = (
        'PNG Files (*.png);;'
        'SVG Files (*.svg);;'
        'TIFF Files (*.tif *.tiff);;'
        'JPEG Files (*.jpg *.jpeg);;'
        'All Files (*)'
    )

    def __init__(self):
        super().__init__()
        self.add_input('figure', color=PORT_COLORS['figure'])

        from nodes.base import NodeFileSaver
        saver = NodeFileSaver(self.view, name='file_path', label='Save Path',
                              ext_filter=self._EXT_FILTER)
        self.add_custom_widget(saver,
                               widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value)

        self._add_int_spinbox('dpi', 'DPI', value=300, min_val=72, max_val=1200)

    def evaluate(self):
        self.reset_progress()

        port = self.inputs().get('figure')
        if not port or not port.connected_ports():
            self.mark_error()
            return False, "No input connected"
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())
        if not isinstance(data, FigureData) or data.payload is None:
            self.mark_error()
            return False, "Input must be FigureData"

        file_path = str(self.get_property('file_path') or '').strip()
        if not file_path:
            self.mark_error()
            return False, "No file path specified"

        import os
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        dpi = int(self.get_property('dpi') or 300)
        fig = data.payload

        self.set_progress(30)
        try:
            fig.savefig(file_path, bbox_inches='tight', dpi=dpi)
            self.set_progress(100)
            self.mark_clean()
            return True, None
        except Exception as e:
            self.mark_error()
            return False, str(e)
