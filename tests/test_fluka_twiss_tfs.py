# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import pytest
import numpy as np
from pathlib import Path

import xtrack as xt
import xcoll as xc
from xcoll.scattering_routines.fluka.twiss_tfs import (COLUMNS, COLLIMATOR_COLUMNS, twiss_table, write_tfs,
            collimator_settings, write_collgaps, read_collgaps, add_collimator_columns, _madx_particle, _read_tfs)


path = Path(__file__).parent / 'data'


@pytest.mark.xcother
def test_twiss_tfs_hllhc(tmp_path):
    line = xt.load(path / 'sequence_hllhc19_b1_thick.json')
    colldb = xc.CollimatorDatabase.from_yaml(Path(__file__).parents[1] / 'examples' / 'colldbs'
                                             / 'hllhc_tight_15cm.yaml', beam=1)
    table = twiss_table(line, colldb=colldb, sequence='LHCB1', verbose=False)

    # Header
    beta_gamma = table['PC'] / table['MASS']
    assert table['TYPE'] == 'TWISS'
    assert table['SEQUENCE'] == 'LHCB1'
    assert table['PARTICLE'] == 'PROTON'
    assert np.isclose(table['EX'], colldb.nemitt_x / beta_gamma)
    assert np.isclose(table['EY'], colldb.nemitt_y / beta_gamma)
    assert np.isclose(table['LENGTH'], line.get_length())

    # Rows: no apertures, no '||' names, S increasing up to the length (at the element exit)
    assert not any(nn.startswith('limit') or '||' in nn for nn in table['name'])
    assert np.all(np.diff(table['s']) >= -1e-9)
    assert np.isclose(table['s'][-1], table['LENGTH'])
    assert np.isclose(table['l'].sum(), table['LENGTH'])

    # All CollDB collimators of the line are COLLIMATOR
    rows = {nn: ii for ii, nn in enumerate(table['name'])}
    in_line = [nn for nn in colldb.collimator_names if nn in rows]
    assert len(in_line) > 0
    assert all(table['keyword'][rows[nn]] == 'COLLIMATOR' for nn in in_line)
    keywords = set(table['keyword'])
    assert {'SBEND', 'RBEND', 'QUADRUPOLE', 'SEXTUPOLE', 'MONITOR', 'HKICKER', 'VKICKER', 'TKICKER'} <= keywords

    # Collimator settings: half gap = nsig * beam size at the collimator centre
    settings = collimator_settings(table, colldb)
    assert [cc['NAME'] for cc in settings] == sorted([nn.upper() for nn in in_line], key=lambda nn: rows[nn.lower()])
    for cc in settings:
        coll = colldb[cc['NAME'].lower()]
        sigma = np.sqrt((cc['SIGX'] * np.cos(cc['ANGLE']))**2 + (cc['SIGY'] * np.sin(cc['ANGLE']))**2)
        if not coll['active'] or (coll['gap'] is None and coll['jaw'] is None):
            assert cc['NSIG'] == 999.                            # inactive: half gap = parking
        else:
            assert np.isclose(cc['HALFGAP'], cc['NSIG'] * sigma)  # from the gap, or NSIG from the jaw

    # collgaps file round trip
    write_collgaps(tmp_path / 'collgaps.dat', settings)
    settings_read = read_collgaps(tmp_path / 'collgaps.dat')
    assert len(settings_read) == len(settings)
    for cc, cc_read in zip(settings, settings_read):
        for key, val in cc.items():
            if isinstance(val, str):
                assert cc_read[key] == val
            else:
                assert np.isclose(cc_read[key], val, rtol=1e-14, atol=0)

    # TFS file round trip, with the collimator columns
    add_collimator_columns(table, settings)
    write_tfs(tmp_path / 'twiss.tfs', table)
    scalars, columns = _read_tfs(tmp_path / 'twiss.tfs')
    assert list(columns) == [cc.upper() for cc in COLUMNS + COLLIMATOR_COLUMNS]
    assert np.isclose(scalars['Q1'], table['Q1'], rtol=1e-9)
    assert np.all(columns['NAME'] == np.char.upper(table['name']))
    assert np.all(columns['KEYWORD'] == table['keyword'])
    for col in COLUMNS + COLLIMATOR_COLUMNS:
        if col not in ['name', 'keyword']:
            assert np.allclose(columns[col.upper()], table[col], rtol=1e-9, atol=1e-15)
    for cc in settings:
        ii = rows[cc['NAME'].lower()]
        assert columns['ANGLE'][ii] == pytest.approx(cc['ANGLE'], rel=1e-9)
        assert columns['HALFGAP'][ii] == pytest.approx(cc['HALFGAP'], rel=1e-9)


@pytest.mark.xcother
def test_twiss_tfs_thick_sps():
    env = xt.load(path / 'sequence_sps_q20_inj.json')
    line = env[list(env.lines)[0]] if isinstance(env, xt.Environment) else env
    table = twiss_table(line, nemitt_x=2.5e-6, verbose=False)
    keywords = set(table['keyword'])
    assert {'QUADRUPOLE', 'RBEND'} <= keywords
    assert np.isclose(table['l'].sum(), line.get_length())
    quads = table['keyword'] == 'QUADRUPOLE'
    assert np.all(table['l'][quads] > 0) and np.count_nonzero(table['k1l'][quads]) > 0.9 * quads.sum()
    tw = line.twiss4d()
    assert np.isclose(table['Q1'], tw.qx) and np.isclose(table['Q2'], tw.qy)
    assert np.isclose(table['DQ1'], tw.dqx / tw.beta0) and np.isclose(table['DQ2'], tw.dqy / tw.beta0)
    # Optics at the element exit: the values of xtrack's Twiss at the end of each element
    idx = np.searchsorted(tw.s, table['s'][quads] - 1e-9)
    assert np.allclose(table['betx'][quads], tw.betx[idx], rtol=1e-10)


@pytest.mark.xcother
def test_twiss_tfs_rows():
    # A small ring: a sliced quadrupole, drifts split by an aperture, a thin kicker
    env = xt.Environment()
    env.particle_ref = xt.Particles(p0c=7e12)
    env.new('qf', xt.Quadrupole, length=1, k1=0.01)
    env.new('qd', xt.Quadrupole, length=1, k1=-0.01)
    env.new('ap', xt.LimitEllipse, a=0.1, b=0.1)
    env.new('mk', xt.Multipole, knl=[1e-6])
    line = env.new_line(length=20, components=[env.place('qf', at=0.5), env.place('ap', at=5),
                                               env.place('mk', at=9), env.place('qd', at=10.5)])
    line.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=None),
                                                  xt.Strategy(slicing=xt.Teapot(4), name='qf')])
    table = twiss_table(line, nemitt_x=2.5e-6, verbose=False)
    assert list(table['keyword']) == ['QUADRUPOLE', 'DRIFT', 'TKICKER', 'DRIFT', 'QUADRUPOLE', 'DRIFT']
    assert list(table['name'][[0, 2, 4]]) == ['qf', 'mk', 'qd']
    assert np.allclose(table['l'], [1, 8, 0, 1, 1, 9])
    assert np.allclose(table['s'], [1, 9, 9, 10, 11, 20])
    assert np.isclose(table['k1l'][0], 0.01)                     # the four slices summed
    assert np.isclose(table['k1l'][4], -0.01)
    assert table['hkick'][2] == -1e-6                            # kicker: HKICK = -K0L, K0L = 0
    assert table['k0l'][2] == 0
    assert np.isclose(table['EY'], table['EX'])                  # nemitt_y defaults to nemitt_x


@pytest.mark.xcother
def test_twiss_tfs_thin_line():
    line = xt.load(path / 'sequence_lhc_run3_b1.json').lhcb1
    with pytest.raises(ValueError, match="needs a thick line"):
        twiss_table(line, nemitt_x=2.5e-6, verbose=False)


@pytest.mark.xcother
def test_twiss_tfs_particle():
    assert _madx_particle(xt.Particles('proton', p0c=7e12)) == 'PROTON'
    assert _madx_particle(xt.Particles('Pb208', p0c=7e12 * 82, q0=82)) == 'ION'
    assert _madx_particle(xt.Particles('electron', p0c=1e9)) == 'ELECTRON'
    assert _madx_particle(xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, p0c=7e12)) == 'PROTON'   # no PDG id
