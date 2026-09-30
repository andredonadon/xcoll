import time
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import xtrack as xt
import xcoll as xc
import xpart as xp
from xtrack.particles import pdg


path_in = Path(__file__).parent
path_out = Path.cwd()

beam = 1
ip = 'ip1'
species_b1 = 'Pb208'          # tracked beam: name ('proton', 'Pb208', 'O16', ...) or PDG ID (2212, 1000822080, ...)
species_b2 = 'Pb208'          # other beam, same format; can differ (p-Pb, Pb-p)
energy_per_charge = 6.8e12    # p0c per unit charge [eV], same rigidity for both beams
interaction = 'inelastic'     # 'inelastic', 'elastic', 'emd'
num_particles = 100
sigma_z = 8e-2
seed = 5656565
dump = False

INTERACTION_CODE = {'inelastic': 1.0, 'elastic': 10.0, 'emd': 100.0}


def other_beam_at_ip(tw, ip, line_b2=None):
    # The FLUKA source expects the optics and closed orbit of the other beam in its
    # own reference frame (s along its motion, x mirrored). Passing the tracked
    # beam's twiss instead mirrors its crossing angle and removes the crossing.
    if line_b2 is not None:
        # Counter-rotating line, in its own frame
        row = line_b2.twiss().rows[ip]
        return {kk: row[kk][0] for kk in ['x', 'px', 'y', 'py', 'betx', 'bety',
                                          'alfx', 'alfy', 'dx', 'dpx', 'dy', 'dpy']}
    # Fallback: mirror the tracked beam through the IP, assuming the two beams are
    # mirror images of each other (LHC design; the crossing angles of the real b1/b2
    # lines agree within 0.3% at IP1/2/5/8). In the lab the other beam's orbit is
    # the transverse reflection of this one; going to its own frame (s -> -s, x -> -x)
    # gives x2 = x1, px2 = -px1, y2 = -y1, py2 = py1 (dispersion transforms like the
    # orbit). Beta and alpha are copied (alpha only enters FLUKA as alpha**2).
    row = tw.rows[ip]
    return {'x':     row.x[0],   'px':   -row.px[0],
            'y':    -row.y[0],   'py':    row.py[0],
            'betx':  row.betx[0], 'bety': row.bety[0],
            'alfx':  row.alfx[0], 'alfy': row.alfy[0],
            'dx':    row.dx[0],  'dpx':  -row.dpx[0],
            'dy':   -row.dy[0],  'dpy':   row.dpy[0]}


def FlukaBeamBeamSource(line, colldb, int_type, ip, num_particles, pdg_id_b1, p0c_b1, pdg_id_b2,
                        sigma_z = 0, sigma_dpp = None, line_b2 = None, seed = None):

    if seed is not None:
        xc.fluka.engine.seed = seed
        np.random.seed(seed)

    tw = line.twiss()
    b2 = other_beam_at_ip(tw, ip, line_b2=line_b2)

    # Beam-2 direction for FLUKA without optics, from its orbit at the IP (FLUKA's lab
    # momentum is (-x', +y') in beam 2's own frame): theta2 >= 0 (angle to -z), phi_ir the
    # plane angle in degrees. Includes the sign of the crossing and the spectrometer bumps.
    theta2 = math.atan(math.hypot(b2['px'], b2['py']))
    phi_ir = math.degrees(math.atan2(b2['py'], -b2['px']))

    ref_b1 = xt.Particles.reference_from_pdg_id(pdg_id=pdg_id_b1, p0c=p0c_b1)
    betagamma_b1 = ref_b1.beta0[0] * ref_b1.gamma0[0]

    # Spare room in the particle arrays for the secondaries (ion fragments need more)
    is_ion = round(ref_b1.mass0 / 931.49410242e6) > 1
    relative_capacity = 400 if is_ion else 20
    _capacity = num_particles * relative_capacity

    # Other beam at the same rigidity
    _, A_b2, Z_b2, _ = pdg.get_properties_from_pdg_id(pdg.get_pdg_id_from_name(pdg_id_b2))
    ref_b2 = xt.Particles.reference_from_pdg_id(pdg_id=pdg_id_b2, p0c=p0c_b1 / ref_b1.q0 * Z_b2)
    betagamma_b2 = ref_b2.beta0[0] * ref_b2.gamma0[0]

    # Beam-2 divergence at the IP (only used by FLUKA when no optics are given)
    sigma_p_x2 = (colldb.nemitt_x / betagamma_b2 * (1 + b2['alfx']**2) / b2['betx'])**0.5
    sigma_p_y2 = (colldb.nemitt_y / betagamma_b2 * (1 + b2['alfy']**2) / b2['bety'])**0.5

    # FLUKA samples the collision position with the luminous-region length, not the
    # bunch length: for two Gaussian bunches of rms length sigma_z crossing with
    # half-angles theta_x/y (Piwinski, hourglass neglected)
    #   sigma_lum = sigma_z / sqrt(2) / sqrt(1 + (theta_x sigma_z/sigma_x)^2 + (theta_y sigma_z/sigma_y)^2)
    # with sigma_x/y the effective size sqrt((sigma_1^2 + sigma_2^2) / 2) of the two beams
    # (they differ for p-ion collisions). The half-angles follow from both beams' IP angles
    # (the other beam in its own frame). Geometric emittances with each species' beta*gamma
    # (the line's own reference particle may be a proton; the optics are the same at equal rigidity).
    sigma_x_b1 = (tw.rows[ip].betx[0] * colldb.nemitt_x / betagamma_b1)**0.5
    sigma_y_b1 = (tw.rows[ip].bety[0] * colldb.nemitt_y / betagamma_b1)**0.5
    sigma_x_b2 = (b2['betx'] * colldb.nemitt_x / betagamma_b2)**0.5
    sigma_y_b2 = (b2['bety'] * colldb.nemitt_y / betagamma_b2)**0.5
    sigma_x_eff = ((sigma_x_b1**2 + sigma_x_b2**2) / 2)**0.5
    sigma_y_eff = ((sigma_y_b1**2 + sigma_y_b2**2) / 2)**0.5
    theta_x = 0.5 * (tw.rows[ip].px[0] - b2['px'])
    theta_y = 0.5 * (tw.rows[ip].py[0] + b2['py'])
    piwinski_sq = (theta_x * sigma_z / sigma_x_eff)**2 + (theta_y * sigma_z / sigma_y_eff)**2
    sigma_z_lum = sigma_z / math.sqrt(2) / math.sqrt(1 + piwinski_sq)
    print(f"Luminous region: sigma_z_lum = {sigma_z_lum*1e2:.2f} cm (bunch sigma_z = {sigma_z*1e2:.2f} cm, "
          f"half crossing = ({theta_x*1e6:.1f}, {theta_y*1e6:.1f}) urad, Piwinski = {math.sqrt(piwinski_sq):.3f})")

    # FLUKA samples the collision within an insertion of length L around its centre and
    # expects the particles at IP - L/2: take L = +-4 sigma_z_lum (capped by the dummy
    # length, so the collisions stay inside the FLUKA IP pipe) and drift by -L/2 below.
    dummy_length = 0.502
    insertion_length = min(2 * 4 * sigma_z_lum, dummy_length)

    # Longitudinal coordinates of the tracked bunch, matched to the RF bucket for sigma_z.
    # Beam-2 momentum spread: same matching with beam 2's own species on its line (or on the
    # tracked line if no beam-2 line is given), unless sigma_dpp is given explicitly.
    zeta, delta = xp.generate_longitudinal_coordinates(
        num_particles=num_particles, distribution='gaussian', sigma_z=sigma_z,
        line=line, particle_ref=ref_b1)
    if sigma_dpp is None:
        if line_b2 is not None:
            _, delta_b2 = xp.generate_longitudinal_coordinates(
                num_particles=10000, distribution='gaussian', sigma_z=sigma_z,
                line=line_b2, particle_ref=ref_b2)
            source = 'matched on line_b2'
        else:
            _, delta_b2 = xp.generate_longitudinal_coordinates(
                num_particles=10000, distribution='gaussian', sigma_z=sigma_z,
                line=line, particle_ref=ref_b1)
            source = 'matched on the tracked line'
        sigma_dpp = float(np.std(delta_b2))
        print(f"Beam-2 momentum spread: {sigma_dpp:.2e} ({source}, sigma_z = {sigma_z*1e2:.1f} cm)")


    bb_int = {
            'int_type': int_type,
            'theta2': theta2, # Angle between b2 and -Z
            'xs': phi_ir,
            'sigma_p_x2': sigma_p_x2,
            'sigma_p_y2': sigma_p_y2,
            'Z': Z_b2,
            'A': A_b2,
            'betx': b2['betx'],
            'bety': b2['bety'],
            'dx': b2['dx'],
            'dy': b2['dy'],
            'alfx': b2['alfx'],
            'alfy': b2['alfy'],
            'dpx': b2['dpx'],
            'dpy': b2['dpy'],
            'rms_emx': colldb.nemitt_x, #/(tw.beta0* tw.gamma0),
            'rms_emy': colldb.nemitt_y, #/(tw.beta0* tw.gamma0)
            'offset': [b2['x'], b2['px'], b2['y'], b2['py']],
            'sigma_z': sigma_z_lum * 1e2,  # collision-position spread [cm]
            'insertion_length': insertion_length * 1e2,  # [cm]
            'sigma_dpp': sigma_dpp
            }

    coll_dummy = xc.FlukaCollimator(assembly="lhc_ippipe", length=dummy_length, jaw=0.001)

    xc.fluka.engine.particle_ref = xt.Particles.reference_from_pdg_id(pdg_id=pdg_id_b1, p0c=p0c_b1)
    xc.fluka.engine.minimum_free_length_fortran_array = relative_capacity
    xc.fluka.engine.relative_length_fortran_array = relative_capacity

    # FLUKA generates the collision in the rest frame of the other beam's nucleus, i.e.
    # at ~2 gamma^2 m per nucleon (~1.5e7 GeV/n for Pb-Pb at 6.8 Z TeV). By default the
    # high-energy model (DPMJET, used for ion-ion) is only initialised up to the BEAM-card
    # (fixed-target) energy and crashes. PHYSICS/LIMITS sets the maximum pp CMS momentum,
    # which for colliding beams is the beam momentum per nucleon (+ margin).
    part_ref = xc.fluka.engine.particle_ref
    A_b1 = max(1, round(part_ref.mass0 / 931.49410242e6))         # mass / atomic mass unit
    p_n_b1 = p0c_b1 / A_b1 / 1e9                                   # [GeV/c per nucleon]
    p_n_b2 = p0c_b1 / part_ref.q0 * Z_b2 / A_b2 / 1e9              # same rigidity
    p_cms_max = 1.1 * max(p_n_b1, p_n_b2)
    limits_card = f"PHYSICS   {p_cms_max:10.1f}{'':50s}LIMITS"

    xc.fluka.engine.start(elements=coll_dummy, clean=False , verbose=True, include_showers=False, return_ions=True, bb_int=bb_int, touches=False,
                          extra_physics_cards=[limits_card])


    line.build_tracker()

    # Matched Gaussian bunch at the IP itself (the matched-bunch generator matches at the
    # start of the line, i.e. IP1 for the LHC lines, whatever `ip` is)
    tt = line.get_table()
    x_norm, px_norm = xp.generate_2D_gaussian(num_particles)
    y_norm, py_norm = xp.generate_2D_gaussian(num_particles)
    part_init = line.build_particles(x_norm=x_norm, px_norm=px_norm,
                            y_norm=y_norm, py_norm=py_norm,
                            zeta=zeta, delta=delta,
                            nemitt_x=colldb.nemitt_x,
                            nemitt_y=colldb.nemitt_y,
                            at_element=ip,
                            match_at_s=tt['s', ip],
                            particle_ref=xc.fluka.engine.particle_ref,
                            _capacity=_capacity)
    part = part_init.copy()

    s_ip = part.s[part.state > 0][0]
    xt.Drift(length=-insertion_length / 2, model="exact").track(part)

    print(f"Tracking {num_particles} particles (FLUKA)...     ", end='')
    start = time.time()
    coll_dummy.track(part)
    print(f"Done in {round(time.time()-start, 3)}s.")

    # Back to the IP (pure drift around the IP), to restart the tracking from there
    alive = part.state > 0
    if np.any(alive):
        xt.Drift(length=s_ip - part.s[alive][0], model="exact").track(part)
    xc.fluka.engine.stop(clean=False)

    line.discard_tracker()

    return part


start_time = time.time()

# Path to inputs
path_out.mkdir(exist_ok=True)

line = xt.load(path_in / 'machines' / f'levelling.23_b{beam}.json')
# Counter-rotating beam (in its own frame), for the optics/orbit of the other beam at the IP.
# Set to None to mirror the tracked beam instead (assumes the beams are mirror images).
line_b2 = xt.load(path_in / 'machines' / f'levelling.23_b{3 - beam}.json')
colldb = xc.CollimatorDatabase.from_yaml(path_in / "colldbs" / f'levelling.23.yaml', beam=beam)

q_b1 = pdg.get_properties_from_pdg_id(pdg.get_pdg_id_from_name(species_b1))[0]

part = FlukaBeamBeamSource(
    line=line,
    line_b2=line_b2,
    colldb=colldb,
    int_type=INTERACTION_CODE[interaction],
    ip=ip,
    num_particles=num_particles,
    pdg_id_b1=species_b1,
    p0c_b1=energy_per_charge * q_b1,
    pdg_id_b2=species_b2,
    sigma_z=sigma_z,
    sigma_dpp=None,
    seed=seed
)


part_alive = part.filter(part.state == 1)
values, counts = np.unique(part_alive.pdg_id, return_counts=True)

# Bar chart with string labels
labels = [str(v) for v in values]

plt.figure(figsize=(6,4))
plt.yscale('log')
plt.bar(labels, counts, color='skyblue', edgecolor='black')
plt.xlabel('Particles')
plt.ylabel('Counts')
plt.tight_layout()
plt.xticks(rotation=90)
        
plt.show()

if dump:
    import pickle
    part_pkl_init = path_out / f'part_init.pkl'
    part_dpp_cut_init = part.filter(part.pdg_id != -999999999)
    with open(part_pkl_init, 'wb') as fid_out:
        pickle.dump(part_dpp_cut_init.to_dict(), fid_out)

