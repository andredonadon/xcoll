# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""
MAD-X style thick-lens Twiss table (TFS) of an xsuite line, for the FLUKA LineBuilder.

Everything comes from a thick line and, optionally, the CollDB. The file is written as MAD-X writes it. The KEYWORD
rules use the LHC (and HL-LHC) MAD-X class names: for other machines, KEYWORD can be wrong for drifts and multipoles.

Usage (see examples/fluka_twiss_tfs.py):
    table = twiss_table(line, colldb='CollDB.yaml', beam=1, sequence='LHCB1')
    settings = collimator_settings(table, 'CollDB.yaml', beam=1)
    add_collimator_columns(table, settings)
    write_tfs('twiss_b1.tfs', table)
    write_collgaps('collgaps_b1.dat', settings)

Conventions:
- One row per element. Aperture elements (Limit*) are left out, the slices of an element are merged back into one
  row, and consecutive drifts are merged (they are split by the apertures). Drifts are named DRIFT_<n>.
- S and the optics at the element exit, as MAD-X `twiss` (LineBuilder `twiss_file ... ref` = -1). With
  s_ref='centre', at the element centre as MAD-X `twiss, centre=true` (ref = 0): zero-length elements take the
  values after the element (after the kick), thick elements are cut at their centre, and the centre values of
  rectangular bends are moved from xtrack's straight-body frame to MAD-X's curved one.
- L is the length the element takes in the line (a thin multipole with a nominal `length` has L = 0, as in MAD-X).
- DX, DPX, DY, DPY, DQ1, DQ2 are derivatives with respect to pt, as MAD-X (xtrack's are with respect to delta).
- Strengths are xtrack's integrated strengths; TILT is rot_s_rad. Kickers (HKICKER, VKICKER, KICKER, TKICKER) have
  HKICK = -K0L, VKICK = K0SL and K0L = K0SL = 0, as MAD-X. KICK is 0 (MAD-X's own `kick` is not kept in the line).
- KEYWORD (the MAD-X type of an element class is not in the line, see `_keyword`):
      COLLIMATOR                   xcoll elements (CollDB installed) and the collimators of the colldb argument
      by MAD-X class name          drifts: bpm/bptu/bptd -> MONITOR, other b/apw/vcz -> INSTRUMENT, t/xrp ->
                                   COLLIMATOR, other classes -> PLACEHOLDER (never DRIFT: the LineBuilder looks for a
                                   FEDB model only for elements that are not DRIFT or MARKER); multipoles ms -> TKICKER
      HKICKER, VKICKER, KICKER     multipoles with a corrector calibration whose knl[0], ksl[0], or both are driven by
                                   knobs (without knobs: are not zero)
      TKICKER                      other multipoles that kick or are thick (MKD, MKI, ADT, ...)
      MULTIPOLE                    multipoles driven at higher orders
      from the xsuite type         SBEND, RBEND, QUADRUPOLE, SEXTUPOLE, OCTUPOLE, SOLENOID, RFCAVITY, CRABCAVITY,
                                   MARKER, DRIFT
  The class-name rules follow the LHC naming only (HL-LHC v1.9 B1: MAD-X's KEYWORD for all 6741 elements).
- Collimator settings (optional), at the collimator centre as the LineBuilder computes them, from the CollDB
  through xcoll (see `collimator_settings`) or from an existing collgaps file:
      add_collimator_columns       in the Twiss file: columns ID, HALFGAP, NSIG, TILT1, TILT2,
                                   LENGTH (the CollDB length; 0 on the other rows) and ANGLE = jaw rotation on the collimator rows (the LineBuilder keeps
                                   unknown columns on the elements, and uses ANGLE as the rotation of a collimator)
      write_collgaps               in a separate collgaps file (FLUKA_builder layout), which the LineBuilder reads with
                                   `collimators_settings : FILE` and FLUKA_builder with --CollG
- Header from the line's reference particle (for ions PARTICLE = ION, and MASS, CHARGE, ENERGY, PC of the ion), the
  emittances of the CollDB (or nemitt_x, nemitt_y), and the Twiss. The LineBuilder reads EX, EY, Q1, Q2.

"""

import re
import time
import warnings
import numpy as np
from collections import Counter

import xtrack as xt
from xtrack.particles import pdg


# Columns and header of the TFS file
COLUMNS = ['name', 'keyword', 's', 'l', 'angle', 'tilt', 'k0l', 'k0sl', 'k1l', 'k1sl', 'k2l', 'k2sl',
           'hkick', 'vkick', 'kick', 'x', 'y', 'px', 'py', 'betx', 'bety', 'alfx', 'alfy',
           'mux', 'muy', 'dx', 'dy', 'dpx', 'dpy']
STRING_COLUMNS = ['name', 'keyword']
COLLIMATOR_COLUMNS = ['id', 'halfgap', 'nsig', 'tilt1', 'tilt2', 'length']   # optional (add_collimator_columns):
                                                         # collimator settings; LENGTH is the CollDB length
SCALARS = {'TYPE': str, 'SEQUENCE': str, 'PARTICLE': str, 'MASS': float, 'CHARGE': float, 'ENERGY': float,
           'PC': float, 'GAMMA': float, 'EX': float, 'EY': float, 'SIGE': float, 'SIGT': float, 'NPART': float,
           'LENGTH': float, 'ALFA': float, 'Q1': float, 'Q2': float, 'DQ1': float, 'DQ2': float}

OPTICS = ['x', 'y', 'px', 'py', 'betx', 'bety', 'alfx', 'alfy', 'mux', 'muy', 'dx', 'dy', 'dpx', 'dpy']
DISPERSION = ['dx', 'dy', 'dpx', 'dpy']
STRENGTHS = ['angle', 'k0l', 'k0sl', 'k1l', 'k1sl', 'k2l', 'k2sl']
S_REF = {'exit': -1, 'centre': 0, 'entry': 1}            # LineBuilder `twiss_file ... ref`

# KEYWORD
TYPE_KEYWORD = {
    'Drift': 'DRIFT', 'Marker': 'MARKER', 'Bend': 'SBEND', 'RBend': 'RBEND', 'Quadrupole': 'QUADRUPOLE',
    'Sextupole': 'SEXTUPOLE', 'Octupole': 'OCTUPOLE', 'Multipole': 'MULTIPOLE', 'UniformSolenoid': 'SOLENOID',
    'Solenoid': 'SOLENOID', 'VariableSolenoid': 'SOLENOID', 'Cavity': 'RFCAVITY', 'CrabCavity': 'CRABCAVITY',
    'RFMultipole': 'RFMULTIPOLE', 'SRotation': 'SROTATION', 'YRotation': 'YROTATION', 'XRotation': 'XROTATION',
    'DipoleEdge': 'DIPEDGE', 'XYShift': 'TRANSLATION', 'Wire': 'WIRE',
}
DRIFT_CLASS_KEYWORDS = [                                 # MAD-X class of a drift -> KEYWORD, first match wins
    (('bpm', 'bptu', 'bptd'), 'MONITOR'),                # beam position monitors
    (('b', 'apw', 'vcz'),     'INSTRUMENT'),             # BPW, BPTQR, BTV, BCT, BSRT, BWS, BQS, APWL, VCZ, ...
    (('t', 'xrp'),            'COLLIMATOR'),             # collimators not in the CollDB: TAXS, TAXN, TCLM, XRP, ...
]
CLASS_KEYWORD_EXCEPTIONS = {                           # MAD-X classes that the prefix rules get wrong
    'bgvca003': 'PLACEHOLDER', 'bgvca004': 'PLACEHOLDER',   # beam gas vertex chambers (BGVCA001 is an INSTRUMENT)
}
SEPTUM_CLASSES = ('ms',)                                 # MSD, MSI septa (they carry a corrector calibration)
KICKERS = ['HKICKER', 'VKICKER', 'KICKER', 'TKICKER']


def twiss_table(line, *, s_ref='exit', colldb=None, beam=1, sequence=None, particle=None,
                nemitt_x=None, nemitt_y=None, sige=0., sigt=0., npart=0., verbose=True):
    """Twiss table of the line for the LineBuilder: a dict with the SCALARS (upper case) and COLUMNS (lower case).

    colldb: CollDB file whose collimators get KEYWORD COLLIMATOR, and whose emittances give EX, EY (unless nemitt_x,
    nemitt_y are given). particle: PARTICLE, by default from the reference particle of the line."""
    from ...colldb import CollimatorDatabase
    if s_ref not in S_REF:
        raise ValueError(f"Invalid s_ref '{s_ref}': use one of {list(S_REF)}!")
    collimators = set()
    if colldb is not None:
        if not isinstance(colldb, CollimatorDatabase):
            colldb = _read_colldb(colldb, beam)
        collimators = {name.lower() for name in colldb.collimator_names}
        nemitt_x = colldb.nemitt_x if nemitt_x is None else nemitt_x
        nemitt_y = colldb.nemitt_y if nemitt_y is None else nemitt_y
    if nemitt_x is None:
        raise ValueError("Need an emittance for EX and EY: give a CollDB or nemitt_x!")
    nemitt_y = nemitt_x if nemitt_y is None else nemitt_y

    t0 = time.time()
    tw = line.twiss4d(strengths=True)
    if verbose:
        print(f"Twiss: {time.time() - t0:.1f}s, Q1 = {tw.qx:.6f}, Q2 = {tw.qy:.6f}")
    tt = line.get_table()
    magnets = ['Bend', 'RBend', 'Quadrupole']
    if not np.any(np.isin(tt.element_type, magnets) | np.isin(np.asarray(tt.parent_type, dtype=str), magnets)):
        raise ValueError("The line has no thick bends or quadrupoles: the LineBuilder needs a thick line!")
    rows = _rows(tt)
    first = np.array([rr['first'] for rr in rows])
    last = np.array([rr['last'] for rr in rows])

    table = {}
    table['name'] = np.array([_madx_name(rr['name']) for rr in rows])
    table['keyword'] = _keywords(line, tt, rows, collimators, verbose=verbose)
    s_start = np.asarray(tt.s_start)[first]
    s_end = np.asarray(tt.s_end)[last]
    table['l'] = s_end - s_start
    table['s'] = {'exit': s_end, 'centre': (s_start + s_end) / 2, 'entry': s_start}[s_ref]

    # Optics
    idx = first if s_ref == 'entry' else last + 1        # exit (and zero-length elements at the centre): after it
    for col in OPTICS:
        table[col] = np.asarray(tw[col])[idx].copy()
    if s_ref == 'centre':
        _optics_at_centre(line, table, rows, verbose=verbose)
    for col in DISPERSION:
        table[col] /= tw.beta0

    # Strengths, summed over merged slices
    for col in STRENGTHS:
        cumsum = np.concatenate([[0.], np.cumsum(np.nan_to_num(np.asarray(tw[col], dtype=float)))])
        table[col] = cumsum[last + 1] - cumsum[first]
    tilt = np.asarray(tw.rot_s_rad)[first]
    table['tilt'] = np.where(np.abs(tilt) < 1e-15, 0., tilt)
    kicker = np.isin(table['keyword'], KICKERS)
    table['hkick'] = np.where(kicker, -table['k0l'], 0.)
    table['vkick'] = np.where(kicker, table['k0sl'], 0.)
    table['k0l'] = np.where(kicker, 0., table['k0l'])
    table['k0sl'] = np.where(kicker, 0., table['k0sl'])
    table['kick'] = np.zeros(len(rows))

    # Header
    ref = line.particle_ref
    mass = float(ref.mass0) / 1e9
    beta0, gamma0 = float(tw.beta0), float(tw.gamma0)
    table.update({
        'TYPE': 'TWISS', 'SEQUENCE': sequence or 'LINE', 'PARTICLE': particle or _madx_particle(ref),
        'MASS': mass, 'CHARGE': float(ref.q0), 'ENERGY': mass * gamma0, 'PC': float(tw.p0c) / 1e9,
        'GAMMA': gamma0, 'EX': nemitt_x / (beta0 * gamma0), 'EY': nemitt_y / (beta0 * gamma0),
        'SIGE': sige, 'SIGT': sigt, 'NPART': npart,
        'LENGTH': line.get_length(), 'ALFA': float(tw.momentum_compaction_factor),
        'Q1': float(tw.qx), 'Q2': float(tw.qy), 'DQ1': float(tw.dqx) / beta0, 'DQ2': float(tw.dqy) / beta0,
    })
    return table


def write_tfs(file, table, *, digits=10):
    """Write the table as MAD-X writes a TFS file: numbers with `digits` significant digits (MAD-X: 10), right-aligned;
    strings quoted, left-aligned. Re-writing a MAD-X twiss file this way gives it back byte for byte."""
    width = max(18, digits + 8)
    num = lambda val: f'{float(val):.{digits}g}'
    lines = []
    for key, kind in SCALARS.items():
        val = table[key]
        if kind is str:
            lines.append(f'@ {key:<16} %{len(str(val)):02d}s "{str(val).upper()}"')
        else:
            lines.append(f'@ {key:<16} %le {num(val):>19}')
    names = table.get('columns', COLUMNS)
    is_string = [col in STRING_COLUMNS for col in names]
    lines.append(' '.join(['*'] + [f'{col.upper():<{width}}' if ss else f'{col.upper():>{width}}'
                                   for col, ss in zip(names, is_string)]) + ' ')
    lines.append(' '.join(['$'] + [f'{"%s":<{width}}' if ss else f'{"%le":>{width}}' for ss in is_string]) + ' ')
    columns = [table[col] for col in names]
    for ii in range(len(table['name'])):
        cells = [f'"{str(cc[ii]).upper()}"'.ljust(width) + ' ' if ss else f'{num(cc[ii]):>{width}}'
                 for cc, ss in zip(columns, is_string)]
        lines.append(' ' + ' '.join(cells))
    with open(file, 'w') as fid:
        fid.write('\n'.join(lines) + '\n')


def compare_tfs(file, reference):
    """Print the differences between two TFS files, element by element (matched by NAME, drifts left out)."""
    scal, tab = _read_tfs(file)
    scal_ref, ref = _read_tfs(reference)
    print(f"\nComparison with {reference}")
    for key in ['Q1', 'Q2', 'DQ1', 'DQ2', 'ALFA', 'LENGTH', 'EX', 'EY']:
        if key in scal and key in scal_ref:
            print(f"  @ {key:7s} {scal[key]: .10g}  reference {scal_ref[key]: .10g}")
    idx_ref = {nn: ii for ii, nn in enumerate(ref['NAME'])}
    not_drift = tab['KEYWORD'] != 'DRIFT'
    pairs = np.array([(ii, idx_ref[nn]) for ii, nn in enumerate(tab['NAME']) if not_drift[ii] and nn in idx_ref])
    only_here = [nn for ii, nn in enumerate(tab['NAME']) if not_drift[ii] and nn not in idx_ref]
    names = set(tab['NAME'])
    only_ref = dict(Counter(kk for nn, kk in zip(ref['NAME'], ref['KEYWORD']) if kk != 'DRIFT' and nn not in names))
    print(f"  rows: {len(tab['NAME'])} here, {len(ref['NAME'])} in the reference; "
          f"non-drift rows matched by name: {len(pairs)}")
    print(f"  non-drift rows only here: {len(only_here)} {only_here[:6]}")
    print(f"  non-drift rows only in the reference, by KEYWORD: {only_ref}")
    ii, jj = pairs[:, 0], pairs[:, 1]
    differ = tab['KEYWORD'][ii] != ref['KEYWORD'][jj]
    print(f"  KEYWORD differs for {differ.sum()} of {len(ii)} elements (here -> reference: count, e.g.)")
    groups = {}
    for nn, kk, kk_ref in zip(tab['NAME'][ii][differ], tab['KEYWORD'][ii][differ], ref['KEYWORD'][jj][differ]):
        groups.setdefault((kk, kk_ref), []).append(nn)
    for (kk, kk_ref), nn in sorted(groups.items(), key=lambda item: -len(item[1])):
        print(f"    {kk:12s} -> {kk_ref:12s} {len(nn):4d}  {nn[:5]}")
    print(f"  {'column':7s} {'max |diff|':>11s} {'median':>9s}  at")
    for col in [cc.upper() for cc in COLUMNS + COLLIMATOR_COLUMNS if cc not in STRING_COLUMNS]:
        if col not in tab or col not in ref:
            if col.lower() not in COLLIMATOR_COLUMNS:
                print(f"  {col:7s} not in both files")
            continue
        diff = np.abs(tab[col][ii] - ref[col][jj])
        kk = int(np.argmax(diff))
        print(f"  {col:7s} {diff[kk]:11.3e} {np.median(diff):9.1e}  {tab['NAME'][ii][kk]}")


def collimator_settings(table, colldb, *, beam=1, s_ref='exit', nemitt_x=None, nemitt_y=None):
    """Settings of the CollDB collimators in the line, at the collimator centre (as the LineBuilder computes them
    from a YAML CollDB): a list of dicts with COLLGAPS_COLUMNS, in machine order.

    BETAX, BETAY from the Twiss table (the collimator is a drift), SIGX, SIGY = sqrt(beta * geometric emittance),
    HALFGAP = gap * sqrt(SIGX**2 * cos(ANGLE)**2 + SIGY**2 * sin(ANGLE)**2); a collimator set by its jaw (in m) gets
    HALFGAP = jaw. Collimators without gap and jaw are inactive (NSIG = 999, HALFGAP = parking). There is no side
    and no offset: one-sided collimators get the half gap of their jaw, and offsets or asymmetric jaws are left out
    (with a warning)."""
    from ...colldb import CollimatorDatabase
    if not isinstance(colldb, CollimatorDatabase):
        colldb = _read_colldb(colldb, beam)              # a CollDB file
    nemitt_x = colldb.nemitt_x if nemitt_x is None else nemitt_x
    nemitt_y = colldb.nemitt_y if nemitt_y is None else nemitt_y
    beta_gamma = table['PC'] / table['MASS']
    rows = {name.lower(): ii for ii, name in enumerate(table['name'])}
    names = sorted([nn for nn in colldb.collimator_names if nn.lower() in rows], key=lambda nn: rows[nn.lower()])
    missing = [nn for nn in colldb.collimator_names if nn.lower() not in rows]
    if missing:
        warnings.warn(f"Collimators of the CollDB not in the line (left out): {missing}")
    settings = []
    for ID, name in enumerate(names, start=1):
        coll = colldb[name]
        ii = rows[name.lower()]
        betx, bety = _beta_at_centre(table, ii, s_ref)
        sigx, sigy = np.sqrt(betx * nemitt_x / beta_gamma), np.sqrt(bety * nemitt_y / beta_gamma)
        angle = np.deg2rad(coll['angle'])
        sigma = np.sqrt((sigx * np.cos(angle))**2 + (sigy * np.sin(angle))**2)
        gap, jaw = _per_jaw(coll['gap']), _per_jaw(coll['jaw'], symmetric=True)
        side = {'left': 0, 'right': 1}.get(coll['side'])
        if not coll['active'] or (gap is None and jaw is None):
            nsig, halfgap = 999., coll['parking']
        elif jaw is not None:
            halfgap = abs(jaw[side]) if side is not None else (jaw[0] - jaw[1]) / 2
            nsig = halfgap / sigma
            if side is None and not np.isclose(jaw[0], -jaw[1]):
                warnings.warn(f"{name}: asymmetric jaws {jaw} written as the half gap {halfgap} (no offset).")
        else:
            nsig = gap[side] if side is not None else (gap[0] + gap[1]) / 2
            halfgap = nsig * sigma
            if side is None and gap[0] != gap[1]:
                warnings.warn(f"{name}: asymmetric gaps {gap} written as {nsig} sigma (no offset).")
        if np.any(np.asarray(coll['offset']) != 0):
            warnings.warn(f"{name}: offset {coll['offset']} left out (no offset column).")
        if abs(coll['length'] - table['l'][ii]) > 1e-4:
            warnings.warn(f"{name}: length {coll['length']} m in the CollDB but {table['l'][ii]} m in the line (LENGTH is "
                 f"the CollDB one; the LineBuilder checks it against L only when L > 0).")
        tilt = np.deg2rad(_per_jaw(coll['tilt']) or [0., 0.])
        settings.append({'ID': ID, 'NAME': name.upper(), 'ANGLE': angle, 'BETAX': betx, 'BETAY': bety,
                         'HALFGAP': halfgap, 'MATERIAL': str(coll['material']).upper(), 'LENGTH': coll['length'],
                         'SIGX': sigx, 'SIGY': sigy, 'TILT1': tilt[0], 'TILT2': tilt[1], 'NSIG': nsig})
    return settings


def write_collgaps(file, settings):
    """Write collimator settings in the collgaps format of FLUKA_builder.py (--CollG), which the LineBuilder also
    reads (inputcard: `collimators_settings : <file>`; it uses ID NAME ANGLE HALFGAP LENGTH NSIG TILT1 TILT2)."""
    # No comment lines: the LineBuilder first tries the file as YAML (a leading '*' is a YAML alias), and takes the
    # first '#' line as the column header.
    lines = ['# ID NAME ANGLE[rad] BETAX[m] BETAY[m] HALFGAP[m] MATERIAL LENGTH[m] SIGX[m] SIGY[m] TILT1[rad] '
             'TILT2[rad] NSIG']
    for cc in settings:
        numbers = [f"{cc[key]:23.16e}" for key in ["ANGLE", "BETAX", "BETAY", "HALFGAP"]]
        numbers += [f"{cc['MATERIAL']:<6s}"]
        numbers += [f"{cc[key]:23.16e}" for key in ["LENGTH", "SIGX", "SIGY", "TILT1", "TILT2", "NSIG"]]
        lines.append(f"{int(cc['ID']):6d} {cc['NAME']:<20s} " + ' '.join(numbers))
    with open(file, 'w') as fid:
        fid.write('\n'.join(lines) + '\n')


def read_collgaps(file):
    """Collimator settings from a collgaps file (columns found by the '#' header line; units in [] dropped)."""
    settings, names = [], None
    with open(file) as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            if line.startswith('#'):
                if names is None:
                    names = re.sub(r'\[[^\]]*\]', '', line[1:]).upper().split()
                continue
            values = dict(zip(names, line.split()))
            settings.append({key: (val if key in ['NAME', 'MATERIAL'] else float(val)) for key, val in values.items()})
    if names is None or not {'NAME', 'ANGLE', 'HALFGAP'} <= set(names):
        raise ValueError(f"{file} is not a collgaps file: no '#' header with NAME, ANGLE, HALFGAP!")
    return settings


def add_collimator_columns(table, settings):
    """Put the collimator settings in the Twiss table, on the collimator rows: the columns COLLIMATOR_COLUMNS
    (0 on the other rows: ID, HALFGAP, NSIG, TILT1, TILT2, and LENGTH = the CollDB length, which the LineBuilder
    uses for the collimators while L stays the length in the line), and ANGLE = the jaw rotation (radians), as the
    LineBuilder does when it reads a collimators_settings file (MAD-X has ANGLE = 0 for collimators)."""
    rows = {name.upper(): ii for ii, name in enumerate(table['name'])}
    for col in COLLIMATOR_COLUMNS:
        table[col] = np.zeros(len(table['name']))
    missing = []
    for cc in settings:
        ii = rows.get(cc['NAME'].upper())
        if ii is None:
            missing.append(cc['NAME'])
            continue
        table['angle'][ii] = cc['ANGLE']
        for col in COLLIMATOR_COLUMNS:
            table[col][ii] = cc.get(col.upper(), 0.)
    if missing:
        warnings.warn(f"Collimators not in the line (left out): {missing}")
    table['columns'] = COLUMNS + COLLIMATOR_COLUMNS
    return table


def _per_jaw(value, symmetric=False):
    """A CollDB value as [left, right], or None. A single jaw position (symmetric=True) means +jaw and -jaw."""
    if value is None:
        return None
    if np.ndim(value) == 0:
        return [value, -value] if symmetric else [value, value]
    return list(value)


def _beta_at_centre(table, ii, s_ref):
    """Beta functions at the centre of row ii (a collimator: a drift for the optics), from the values at s_ref."""
    ds = {'exit': -0.5, 'centre': 0, 'entry': 0.5}[s_ref] * table['l'][ii]
    betas = []
    for plane in 'xy':
        beta, alfa = table[f'bet{plane}'][ii], table[f'alf{plane}'][ii]
        gamma = (1 + alfa**2) / beta
        betas.append(beta - 2 * ds * alfa + ds**2 * gamma)
    return betas

# Rows and KEYWORD
# ================

def _rows(tt):
    """Rows of the TFS file over the line table: [{'first', 'last', 'name', 'kind'}]. Aperture elements are left out;
    the slices of an element (kind 'slices') and consecutive drifts (kind 'drift') are merged into one row."""
    parents = list(tt.parent_name)
    sliced = {parent for parent in parents if parent is not None}
    rows = []
    for ii, (name, etype, parent) in enumerate(zip(tt.name, tt.element_type, parents)):
        if name == '_end_point' or etype == '' or etype.startswith('Limit'):
            continue
        marker = re.match(r'(.*)_(entry|exit)$', name)
        if etype == 'Marker' and marker and marker.group(1) in sliced:
            continue                                     # markers that xtrack adds around a sliced element
        if parent is not None:
            kind, name = 'slices', parent
        elif etype == 'Drift' and name.startswith('||drift'):
            kind = 'drift'
        else:
            kind = 'element'
        previous = rows[-1] if rows else None
        same_element = kind == 'slices' and previous and previous['kind'] == 'slices' and previous['name'] == name
        next_drift = kind == 'drift' and previous and previous['kind'] == 'drift'
        if same_element or next_drift:
            previous['last'] = ii
        else:
            rows.append({'first': ii, 'last': ii, 'name': name, 'kind': kind})
    return rows


def _madx_name(name):
    return re.sub(r'^\|\|drift_', 'drift_', name).replace('||', '')


def _keywords(line, tt, rows, collimators, verbose=True):
    """MAD-X KEYWORD of each row, from the line (and the CollDB collimators). Prints the rules used."""
    from ...beam_elements import block_classes
    xcoll_types = {cls.__name__ for cls in block_classes}
    knobs = _knob_targets(line)
    etypes, parent_types = list(tt.element_type), list(tt.parent_type)
    prototypes, isthick_all = list(tt.base_prototype), list(tt.isthick)
    used = {}
    keywords = []
    for rr in rows:
        ii = rr['first']
        etype, prototype, isthick = etypes[ii], prototypes[ii], bool(isthick_all[ii])
        try:
            element = line.env[rr['name']] if rr['kind'] == 'slices' else line[rr['name']]   # a view
        except Exception:
            element = None
        if rr['kind'] == 'slices':
            etype, isthick = parent_types[ii] or etype, True
            if prototype is None and element is not None:
                prototype = getattr(element, 'prototype', None)
        keyword, rule = _keyword(rr, etype, prototype, isthick, element, knobs, collimators, xcoll_types)
        keywords.append(keyword)
        count, classes = used.get((rule, keyword), (0, set()))
        used[(rule, keyword)] = (count + 1, classes | {str(prototype or etype)})
    if verbose:
        print("KEYWORD (rule -> keyword: elements, classes):")
        for (rule, keyword), (count, classes) in sorted(used.items()):
            examples = f", e.g. {sorted(classes)[:6]}" if rule.startswith(('drift', 'xcoll')) else ''
            print(f"  {rule:34s} -> {keyword:12s} {count}{examples}")
    return np.array(keywords)


def _keyword(row, etype, prototype, isthick, element, knobs, collimators, xcoll_types):
    """KEYWORD of one row, and the rule used."""
    prototype = None if prototype is None else str(prototype).lower()
    if etype in xcoll_types or row['name'].lower() in collimators:
        return 'COLLIMATOR', 'xcoll element or CollDB'
    if etype == 'Multipole':
        if prototype is not None and prototype.startswith(SEPTUM_CLASSES):
            return 'TKICKER', f'multipole of class {"/".join(SEPTUM_CLASSES)}*'
        return _multipole_keyword(row['name'], element, knobs, isthick), 'multipole: knobs and calibration'
    if etype in ['Drift', 'DriftSlice'] and row['kind'] != 'drift' and prototype is not None:
        if prototype in CLASS_KEYWORD_EXCEPTIONS:
            return CLASS_KEYWORD_EXCEPTIONS[prototype], 'drift of an exception class'
        for prefixes, keyword in DRIFT_CLASS_KEYWORDS:
            if prototype.startswith(prefixes):
                return keyword, f'drift of class {"/".join(prefixes)}*'
        return 'PLACEHOLDER', 'drift of another class'
    return TYPE_KEYWORD.get(etype, etype.upper()), 'xsuite type'


def _multipole_keyword(name, element, knobs, isthick):
    """KEYWORD of a multipole: from the strengths driven by knobs (without knobs: the non-zero strengths), and
    whether it carries a corrector calibration (extra['calib'])."""
    driven = knobs.get(name)
    if not driven:
        driven = {(kind, order) for kind in ['knl', 'ksl']
                  for order, val in enumerate(np.asarray(getattr(element, kind, [0.]))) if val != 0}
    if any(order > 0 for _, order in driven):
        return 'MULTIPOLE'
    horizontal, vertical = ('knl', 0) in driven, ('ksl', 0) in driven
    if 'calib' not in (getattr(element, 'extra', None) or {}) and (horizontal or vertical or isthick):
        return 'TKICKER'                                 # kickers without a corrector calibration (MKD, MKI, ADT)
    if horizontal and vertical:
        return 'KICKER'
    if horizontal:
        return 'HKICKER'
    if vertical:
        return 'VKICKER'
    return 'MULTIPOLE'


def _knob_targets(line):
    """{element name: {('knl' or 'ksl', order)}}: the multipole strengths driven by knobs."""
    try:
        manager = line._xdeps_manager
    except Exception:
        manager = getattr(getattr(line, 'env', None), '_xdeps_manager', None)
    if manager is None:
        return {}
    targets = {}
    for target, _ in manager.dump():
        match = re.search(r"\['([^']+)'\]\.(knl|ksl)\[(\d+)\]", str(target))
        if match:
            targets.setdefault(match.group(1), set()).add((match.group(2), int(match.group(3))))
    return targets


# Optics at the element centre
# ============================

def _optics_at_centre(line, table, rows, verbose=True):
    """Thick elements: the optics at their centre, on a copy of the line cut there (zero-length elements keep the
    values after the element, as MAD-X). Rectangular bends are moved to MAD-X's curved frame."""
    thick = table['l'] > 0
    s_centre = table['s'][thick]
    t0 = time.time()
    cut = line.copy(shallow=True)
    cut.cut_at_s(np.unique(s_centre), s_tol=1e-6)
    tw = cut.twiss4d()
    idx = np.searchsorted(tw.s, s_centre - 1e-7)
    if np.any(np.abs(tw.s[idx] - s_centre) > 1e-6):
        raise RuntimeError("Some element centres were not found after cutting the line!")
    for col in OPTICS:
        table[col][thick] = np.asarray(tw[col])[idx]
    if verbose:
        print(f"Twiss at {thick.sum()} element centres: {time.time() - t0:.1f}s")
    for ii in np.flatnonzero(thick & (table['keyword'] == 'RBEND')):
        bend = line[rows[ii]['name']]
        model = getattr(bend, 'rbend_model', None)
        if model in ['straight-body', 'adaptive'] and not getattr(bend, 'rbend_compensate_sagitta', False):
            table['x'][ii] += bend.rbend_shift
            table['px'][ii] += bend.rbend_angle_diff / 2
        elif model not in [None, 'curved-body']:
            warnings.warn(f"Rectangular bend {rows[ii]['name']} with rbend_model '{model}' and compensated sagitta: "
                 f"its centre values are in xtrack's frame.")


# Inputs
# ======

def _read_colldb(file, beam):
    """One beam of a CollDB, as read by xcoll."""
    from ...colldb import CollimatorDatabase
    reader = CollimatorDatabase.from_yaml if str(file).endswith(('.yaml', '.yml')) \
             else CollimatorDatabase.from_json
    try:
        return reader(file, beam=beam, ignore_crystals=False)
    except Exception:
        return reader(file, beam=beam, ignore_crystals=True)


def _madx_particle(particle_ref):
    """MAD-X name of the reference particle: PROTON, ANTIPROTON, ELECTRON, POSITRON, or ION (nuclei)."""
    pdg_id = int(np.atleast_1d(particle_ref.pdg_id)[0])
    charge = float(np.atleast_1d(particle_ref.q0)[0])
    if pdg_id == 0:                                      # no PDG id: from the mass and the charge
        mass = float(np.atleast_1d(particle_ref.mass0)[0])
        if abs(mass - xt.PROTON_MASS_EV) < 1e3:
            return 'PROTON' if charge > 0 else 'ANTIPROTON'
        if abs(mass - xt.ELECTRON_MASS_EV) < 1:
            return 'POSITRON' if charge > 0 else 'ELECTRON'
        return 'ION'
    for is_particle, name in [(pdg.is_proton, 'PROTON'), (pdg.is_antiproton, 'ANTIPROTON'),
                              (pdg.is_electron, 'ELECTRON'), (pdg.is_positron, 'POSITRON'), (pdg.is_ion, 'ION')]:
        if is_particle(pdg_id):
            return name
    return pdg.get_name_from_pdg_id(pdg_id, long_name=False, subscripts=False).upper()


def _read_tfs(file):
    """Scalars and columns of a TFS file."""
    scalars, names, types, rows = {}, [], [], []
    with open(file) as fid:
        for line in fid:
            if line.startswith('@'):
                _, key, kind, val = line.split(None, 3)
                scalars[key] = val.strip().strip('"') if kind.endswith('s') else float(val)
            elif line.startswith('*'):
                names = line.split()[1:]
            elif line.startswith('$'):
                types = line.split()[1:]
            elif line.strip():
                rows.append([tok.strip('"') for tok in re.findall(r'"[^"]*"|\S+', line)])
    columns = {}
    for ii, (name, kind) in enumerate(zip(names, types)):
        values = [row[ii] for row in rows]
        columns[name] = np.array(values) if kind == '%s' else np.array(values, dtype=float)
    return scalars, columns
