import json
from pathlib import Path
import numpy as np

import xobjects as xo
import xpart as xp
import xcoll as xc
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_horizontal_parallel(test_context):
    jaw_L, jaw_R, _, _, cox, _, L, coll = _make_absorber(_context=test_context)
    part, x, _, _, _ = _generate_particles(_context=test_context)
    coll.track(part)
    part.move(_context=xo.ContextCpu())
    part.sort(interleave_lost_particles=True)
    # As the angles are zero, only particles that started in front of the jaw are lost
    mask_hit = (x >= jaw_L) | (x <= jaw_R)
    lost = np.unique(part.state[mask_hit])
    alive = np.unique(part.state[~mask_hit])
    assert len(lost)==1 and lost[0]==-340
    assert len(alive)==1 and alive[0]==1
    s_lost = np.unique(part.s[mask_hit])
    s_alive = np.unique(part.s[~mask_hit])
    assert len(s_lost)==1 and s_lost[0]==0
    assert len(s_alive)==1 and s_alive[0]==L
    assert np.allclose(part.x, x, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_horizontal(test_context):
    jaw_L, jaw_R, _, _, cox, _, L, coll = _make_absorber(_context=test_context)
    part, x, _, xp, _ = _generate_particles(four_dim=True, _context=test_context)
    coll.track(part)
    part.move(_context=xo.ContextCpu())
    part.sort(interleave_lost_particles=True)
    # Particles in front of the jaw are lost, ...
    mask_hit_front = (x >= jaw_L + cox) | (x <= jaw_R + cox)
    # but also those in the opening with an angle that would kick them on the jaw halfway
    mask_hit_angle_L = ~mask_hit_front & (x + xp*L >= jaw_L + cox)
    mask_hit_angle_R = ~mask_hit_front & (x + xp*L <= jaw_R + cox)
    mask_hit = mask_hit_front | mask_hit_angle_L | mask_hit_angle_R
    lost = np.unique(part.state[mask_hit])
    alive = np.unique(part.state[~mask_hit])
    assert len(lost)==1 and lost[0]==-340
    assert len(alive)==1 and alive[0]==1
    s_lost_front = np.unique(part.s[mask_hit_front])
    s_alive = np.unique(part.s[~mask_hit])
    assert len(s_lost_front)==1 and s_lost_front[0]==0
    assert len(s_alive)==1 and s_alive[0]==L
    assert np.allclose(part.x[mask_hit_front], x[mask_hit_front], atol=1e-15, rtol=0)
    assert np.allclose(part.x[~mask_hit],      x[~mask_hit] + xp[~mask_hit]*L, atol=1e-15, rtol=0)
    s_hit_L = (jaw_L + cox - x[mask_hit_angle_L]) / xp[mask_hit_angle_L]
    assert np.allclose(part.s[mask_hit_angle_L], s_hit_L, atol=1e-13, rtol=0)
    assert np.allclose(part.x[mask_hit_angle_L], jaw_L + cox, atol=1e-15, rtol=0)
    s_hit_R = (jaw_R + cox - x[mask_hit_angle_R]) / xp[mask_hit_angle_R]
    assert np.allclose(part.s[mask_hit_angle_R], s_hit_R, atol=1e-13, rtol=0)
    assert np.allclose(part.x[mask_hit_angle_R], jaw_R + cox, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_horizontal_with_tilts(test_context):
    jaw_L, jaw_R, jaw_LD, jaw_RD, cox, _, L, coll = _make_absorber(tilts=[0.0005, -0.00015], _context=test_context)
    part, x, _, xp, _ = _generate_particles(four_dim=True, _context=test_context)
    coll.track(part)
    part.move(_context=xo.ContextCpu())
    part.sort(interleave_lost_particles=True)
    # Particles in front of the jaw are lost, ...
    mask_hit_front = (x >= jaw_L + cox) | (x <= jaw_R + cox)
    # but also those in the opening with an angle that would kick them on the jaw halfway
    mask_hit_angle_L = ~mask_hit_front & (x + xp*L >= jaw_LD + cox)
    mask_hit_angle_R = ~mask_hit_front & (x + xp*L <= jaw_RD + cox)
    mask_hit = mask_hit_front | mask_hit_angle_L | mask_hit_angle_R
    lost = np.unique(part.state[mask_hit])
    alive = np.unique(part.state[~mask_hit])
    assert len(lost)==1 and lost[0]==-340
    assert len(alive)==1 and alive[0]==1
    s_lost_front = np.unique(part.s[mask_hit_front])
    s_alive = np.unique(part.s[~mask_hit])
    assert len(s_lost_front)==1 and s_lost_front[0]==0
    assert len(s_alive)==1 and s_alive[0]==L
    assert np.allclose(part.x[mask_hit_front], x[mask_hit_front], atol=1e-15, rtol=0)
    assert np.allclose(part.x[~mask_hit],      x[~mask_hit] + xp[~mask_hit]*L, atol=1e-15, rtol=0)
    s_hit_L = (jaw_L + cox - x[mask_hit_angle_L]) / ( xp[mask_hit_angle_L] - (jaw_LD-jaw_L)/L)
    assert np.allclose(part.s[mask_hit_angle_L], s_hit_L, atol=1e-13, rtol=0)
    assert np.allclose(part.x[mask_hit_angle_L], jaw_L + cox + (jaw_LD-jaw_L)/L*s_hit_L, atol=1e-15, rtol=0)
    s_hit_R = (jaw_R + cox - x[mask_hit_angle_R]) / ( xp[mask_hit_angle_R] - (jaw_RD-jaw_R)/L)
    assert np.allclose(part.s[mask_hit_angle_R], s_hit_R, atol=1e-13, rtol=0)
    assert np.allclose(part.x[mask_hit_angle_R], jaw_R + cox + (jaw_RD-jaw_R)/L*s_hit_R, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_vertical_parallel(test_context):
    jaw_L, jaw_R, _, _, _, coy, L, coll = _make_absorber(angle=90, _context=test_context)
    part, _, y, _, _ = _generate_particles(_context=test_context)
    coll.track(part)
    part.move(_context=xo.ContextCpu())
    part.sort(interleave_lost_particles=True)
    # As the angles are zero, only particles that started in front of the jaw are lost
    mask_hit = (y >= jaw_L + coy) | (y <= jaw_R + coy)
    lost = np.unique(part.state[mask_hit])
    alive = np.unique(part.state[~mask_hit])
    assert len(lost)==1 and lost[0]==-340
    assert len(alive)==1 and alive[0]==1
    s_lost = np.unique(part.s[mask_hit])
    s_alive = np.unique(part.s[~mask_hit])
    assert len(s_lost)==1 and s_lost[0]==0
    assert len(s_alive)==1 and s_alive[0]==L
    assert np.allclose(part.y, y, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_vertical(test_context):
    jaw_L, jaw_R, _, _, _, coy, L, coll = _make_absorber(angle=90, _context=test_context)
    part, _, y, _, yp = _generate_particles(four_dim=True, _context=test_context)
    coll.track(part)
    part.move(_context=xo.ContextCpu())
    part.sort(interleave_lost_particles=True)
    # Particles in front of the jaw are lost, ...
    mask_hit_front = (y >= jaw_L + coy) | (y <= jaw_R + coy)
    # but also those in the opening with an angle that would kick them on the jaw halfway
    mask_hit_angle_L = ~mask_hit_front & (y + yp*L >= jaw_L + coy)
    mask_hit_angle_R = ~mask_hit_front & (y + yp*L <= jaw_R + coy)
    mask_hit = mask_hit_front | mask_hit_angle_L | mask_hit_angle_R
    lost = np.unique(part.state[mask_hit])
    alive = np.unique(part.state[~mask_hit])
    assert len(lost)==1 and lost[0]==-340
    assert len(alive)==1 and alive[0]==1
    s_lost_front = np.unique(part.s[mask_hit_front])
    s_alive = np.unique(part.s[~mask_hit])
    assert len(s_lost_front)==1 and s_lost_front[0]==0
    assert len(s_alive)==1 and s_alive[0]==L
    assert np.allclose(part.y[mask_hit_front], y[mask_hit_front], atol=1e-15, rtol=0)
    assert np.allclose(part.y[~mask_hit],      y[~mask_hit] + yp[~mask_hit]*L, atol=1e-15, rtol=0)
    s_hit_L = (jaw_L + coy - y[mask_hit_angle_L]) / yp[mask_hit_angle_L]
    assert np.allclose(part.s[mask_hit_angle_L], s_hit_L, atol=1e-12, rtol=0)
    assert np.allclose(part.y[mask_hit_angle_L], jaw_L + coy, atol=1e-15, rtol=0)
    s_hit_R = (jaw_R + coy - y[mask_hit_angle_R]) / yp[mask_hit_angle_R]
    assert np.allclose(part.s[mask_hit_angle_R], s_hit_R, atol=1e-12, rtol=0)
    assert np.allclose(part.y[mask_hit_angle_R], jaw_R + coy, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_vertical_with_tilts(test_context):
    jaw_L, jaw_R, jaw_LD, jaw_RD, _, coy, L, coll = _make_absorber(angle=90, tilts=[0.0005, -0.00015], _context=test_context)
    part, _, y, _, yp = _generate_particles(four_dim=True, _context=test_context)
    coll.track(part)
    part.move(_context=xo.ContextCpu())
    part.sort(interleave_lost_particles=True)
    # Particles in front of the jaw are lost, ...
    mask_hit_front = (y >= jaw_L + coy) | (y <= jaw_R + coy)
    # but also those in the opening with an angle that would kick them on the jaw halfway
    mask_hit_angle_L = ~mask_hit_front & (y + yp*L >= jaw_LD + coy)
    mask_hit_angle_R = ~mask_hit_front & (y + yp*L <= jaw_RD + coy)
    mask_hit = mask_hit_front | mask_hit_angle_L | mask_hit_angle_R
    lost = np.unique(part.state[mask_hit])
    alive = np.unique(part.state[~mask_hit])
    assert len(lost)==1 and lost[0]==-340
    assert len(alive)==1 and alive[0]==1
    s_lost_front = np.unique(part.s[mask_hit_front])
    s_alive = np.unique(part.s[~mask_hit])
    assert len(s_lost_front)==1 and s_lost_front[0]==0
    assert len(s_alive)==1 and s_alive[0]==L
    assert np.allclose(part.y[mask_hit_front], y[mask_hit_front], atol=1e-15, rtol=0)
    assert np.allclose(part.y[~mask_hit],      y[~mask_hit] + yp[~mask_hit]*L, atol=1e-15, rtol=0)
    s_hit_L = (jaw_L + coy - y[mask_hit_angle_L]) / ( yp[mask_hit_angle_L] - (jaw_LD-jaw_L)/L)
    assert np.allclose(part.s[mask_hit_angle_L], s_hit_L, atol=1e-13, rtol=0)
    assert np.allclose(part.y[mask_hit_angle_L], jaw_L + coy + (jaw_LD-jaw_L)/L*s_hit_L, atol=1e-15, rtol=0)
    s_hit_R = (jaw_R + coy - y[mask_hit_angle_R]) / ( yp[mask_hit_angle_R] - (jaw_RD-jaw_R)/L)
    assert np.allclose(part.s[mask_hit_angle_R], s_hit_R, atol=1e-13, rtol=0)
    assert np.allclose(part.y[mask_hit_angle_R], jaw_R + coy + (jaw_RD-jaw_R)/L*s_hit_R, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_angled_parallel(test_context):
    angle=17.8
    jaw_L, jaw_R, _, _, cox, _, L, coll = _make_absorber(angle=angle, rotate_co=True, _context=test_context)
    part, x, _, _, _ = _generate_particles(angle=angle, _context=test_context)
    coll.track(part)
    part.move(_context=xo.ContextCpu())
    part.sort(interleave_lost_particles=True)
    # As the angles are zero, only particles that started in front of the jaw are lost
    mask_hit = (x >= jaw_L + cox) | (x <= jaw_R + cox)
    lost = np.unique(part.state[mask_hit])
    alive = np.unique(part.state[~mask_hit])
    assert len(lost)==1 and lost[0]==-340
    assert len(alive)==1 and alive[0]==1
    s_lost = np.unique(part.s[mask_hit])
    s_alive = np.unique(part.s[~mask_hit])
    assert len(s_lost)==1 and s_lost[0]==0
    assert len(s_alive)==1 and s_alive[0]==L
    part_x_rot = part.x * np.cos(angle/180.*np.pi) + part.y * np.sin(angle/180.*np.pi)
    assert np.allclose(part_x_rot, x, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_angled(test_context):
    angle=17.8
    jaw_L, jaw_R, _, _, cox, _, L, coll = _make_absorber(angle=angle, rotate_co=True, _context=test_context)
    part, x, _, xp, _ = _generate_particles(four_dim=True, angle=angle, _context=test_context)
    coll.track(part)
    part.move(_context=xo.ContextCpu())
    part.sort(interleave_lost_particles=True)
    # Particles in front of the jaw are lost, ...
    mask_hit_front = (x >= jaw_L + cox) | (x <= jaw_R + cox)
    # but also those in the opening with an angle that would kick them on the jaw halfway
    mask_hit_angle_L = ~mask_hit_front & (x + xp*L >= jaw_L + cox)
    mask_hit_angle_R = ~mask_hit_front & (x + xp*L <= jaw_R + cox)
    mask_hit = mask_hit_front | mask_hit_angle_L | mask_hit_angle_R
    lost = np.unique(part.state[mask_hit])
    alive = np.unique(part.state[~mask_hit])
    assert len(lost)==1 and lost[0]==-340
    assert len(alive)==1 and alive[0]==1
    s_lost_front = np.unique(part.s[mask_hit_front])
    s_alive = np.unique(part.s[~mask_hit])
    assert len(s_lost_front)==1 and s_lost_front[0]==0
    assert len(s_alive)==1 and s_alive[0]==L
    part_x_rot = part.x * np.cos(angle/180.*np.pi) + part.y * np.sin(angle/180.*np.pi)
    assert np.allclose(part_x_rot[mask_hit_front], x[mask_hit_front], atol=1e-15, rtol=0)
    assert np.allclose(part_x_rot[~mask_hit],      x[~mask_hit] + xp[~mask_hit]*L, atol=1e-15, rtol=0)
    s_hit_L = (jaw_L + cox - x[mask_hit_angle_L]) / xp[mask_hit_angle_L]
    assert np.allclose(part.s[mask_hit_angle_L], s_hit_L, atol=1e-13, rtol=0)
    assert np.allclose(part_x_rot[mask_hit_angle_L], jaw_L + cox, atol=1e-15, rtol=0)
    s_hit_R = (jaw_R + cox - x[mask_hit_angle_R]) / xp[mask_hit_angle_R]
    assert np.allclose(part.s[mask_hit_angle_R], s_hit_R, atol=1e-13, rtol=0)
    assert np.allclose(part_x_rot[mask_hit_angle_R], jaw_R + cox, atol=1e-15, rtol=0)


@for_all_test_contexts
def test_angled_with_tilts(test_context):
    angle=17.8
    jaw_L, jaw_R, jaw_LD, jaw_RD, cox, _, L, coll = _make_absorber(angle=angle, tilts=[0.0005, -0.00015], rotate_co=True, _context=test_context)
    part, x, _, xp, _ = _generate_particles(four_dim=True,angle=angle, _context=test_context)
    coll.track(part)
    part.move(_context=xo.ContextCpu())
    part.sort(interleave_lost_particles=True)
    # Particles in front of the jaw are lost, ...
    mask_hit_front = (x >= jaw_L + cox) | (x <= jaw_R + cox)
    # but also those in the opening with an angle that would kick them on the jaw halfway
    mask_hit_angle_L = ~mask_hit_front & (x + xp*L >= jaw_LD + cox)
    mask_hit_angle_R = ~mask_hit_front & (x + xp*L <= jaw_RD + cox)
    mask_hit = mask_hit_front | mask_hit_angle_L | mask_hit_angle_R
    lost = np.unique(part.state[mask_hit])
    alive = np.unique(part.state[~mask_hit])
    assert len(lost)==1 and lost[0]==-340
    assert len(alive)==1 and alive[0]==1
    s_lost_front = np.unique(part.s[mask_hit_front])
    s_alive = np.unique(part.s[~mask_hit])
    assert len(s_lost_front)==1 and s_lost_front[0]==0
    assert len(s_alive)==1 and s_alive[0]==L
    part_x_rot = part.x * np.cos(angle/180.*np.pi) + part.y * np.sin(angle/180.*np.pi)
    assert np.allclose(part_x_rot[mask_hit_front], x[mask_hit_front], atol=1e-15, rtol=0)
    assert np.allclose(part_x_rot[~mask_hit],      x[~mask_hit] + xp[~mask_hit]*L, atol=1e-15, rtol=0)
    s_hit_L = (jaw_L + cox - x[mask_hit_angle_L]) / ( xp[mask_hit_angle_L] - (jaw_LD-jaw_L)/L)
    assert np.allclose(part.s[mask_hit_angle_L], s_hit_L, atol=1e-13, rtol=0)
    assert np.allclose(part_x_rot[mask_hit_angle_L], jaw_L + cox + (jaw_LD-jaw_L)/L*s_hit_L, atol=1e-15, rtol=0)
    s_hit_R = (jaw_R + cox - x[mask_hit_angle_R]) / ( xp[mask_hit_angle_R] - (jaw_RD-jaw_R)/L)
    assert np.allclose(part.s[mask_hit_angle_R], s_hit_R, atol=1e-13, rtol=0)
    assert np.allclose(part_x_rot[mask_hit_angle_R], jaw_R + cox + (jaw_RD-jaw_R)/L*s_hit_R, atol=1e-15, rtol=0)


def _make_absorber(angle=0, tilts=[0,0], rotate_co=False, _context=None):
    if _context is None:
        _context = xo.ContextCpu()
    jaws = [0.03, -0.01]
    co = [0.0075, -0.089] # TODO: SIMONE, trenger vi dette? 
    L = 0.873
    coll = xc.BlackAbsorber(length=L, angle=angle, jaw=jaws, tilt=tilts, _context=_context)
    if rotate_co:
        cox = co[0] * np.cos(angle/180.*np.pi) + co[1] * np.sin(angle/180.*np.pi)
        coy = co[0] * np.sin(angle/180.*np.pi) + co[1] * np.cos(angle/180.*np.pi)
    else:
        cox = co[0]
        coy = co[1]
    return coll.jaw_LU, coll.jaw_RU, coll.jaw_LD, coll.jaw_RD, cox, coy, L, coll


def _generate_particles(four_dim=False, angle=0, _context=None):
    if _context is None:
        _context = xo.ContextCpu()
    # Make particles
    n_part = 50000
    x = np.random.uniform(-0.1, 0.1, n_part)
    y = np.random.uniform(-0.1, 0.1, n_part)
    if four_dim:
        px = np.random.uniform(-0.1, 0.1, n_part)
        py = np.random.uniform(-0.1, 0.1, n_part)
    else:
        px = 0
        py = 0
    ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, _context=_context)
    part = xp.build_particles(x=x, y=y, px=px, py=py, particle_ref=ref, _context=_context)
    part_init = part.copy()
    part_init.move(_context=xo.ContextCpu())
    x_rot = part_init.x * np.cos(angle/180.*np.pi) + part_init.y * np.sin(angle/180.*np.pi)
    y_rot = part_init.x * np.sin(angle/180.*np.pi) + part_init.y * np.cos(angle/180.*np.pi)
    xp_rot = part_init.px * part_init.rpp * np.cos(angle/180.*np.pi) + part_init.py * part_init.rpp * np.sin(angle/180.*np.pi)
    yp_rot = part_init.px * part_init.rpp * np.sin(angle/180.*np.pi) + part_init.py * part_init.rpp * np.cos(angle/180.*np.pi)
    return part, x_rot, y_rot, xp_rot, yp_rot
