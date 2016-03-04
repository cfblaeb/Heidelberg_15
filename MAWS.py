import simtk.openmm as mm
from simtk.openmm import app
from simtk import unit
from pexpect import spawn
from mpmath import mp as math
import numpy as np
import argparse
from time import sleep

forcefield_name = "leaprc.ff12SB"


def constrainPO3(topology, system):
    C3s = []
    O3s = []
    Ps = []
    C5s = []
    for elem in topology.atoms():
        if elem.name == "C3'":
            C3s.append(elem.index)
        elif elem.name == "O3'":
            O3s.append(elem.index)
        elif elem.name == "P":
            Ps.append(elem.index)
        elif elem.name == "C5'":
            C5s.append(elem.index)

    force = mm.CustomTorsionForce('-k0*(theta^2)')
    force.setForceGroup(2)
    force.addPerTorsionParameter('k0')

    for i in range(len(Ps)):
        force.addTorsion(C3s[i], O3s[i], Ps[i], C5s[i + 1], [20])
    system.addForce(force)
    print(C3s, O3s, Ps, C5s)


def constrainPO5(topology, system):
    C5s = []
    O5s = []
    Ps = []
    O3s = []
    for elem in topology.atoms():
        if elem.name == "O3'":
            O3s.append(elem.index)
        elif elem.name == "O5'":
            O5s.append(elem.index)
        elif elem.name == "P":
            Ps.append(elem.index)
        elif elem.name == "C5'":
            C5s.append(elem.index)
        else:
            pass
    force = mm.CustomTorsionForce('-k0*(theta^2)')
    force.setForceGroup(2)
    force.addPerTorsionParameter('k0')
    for i in range(len(Ps)):
        force.addTorsion(C5s[i + 1], O5s[i + 1], Ps[i], O3s[i], [20])
    system.addForce(force)
    print(C5s, O5s, Ps, O3s)


def constrainC5O5(topology, system):
    C5s = []
    O5s = []
    Ps = []
    H5s = []
    for elem in topology.atoms():
        if elem.name == "C5'":
            C5s.append(elem.index)
        elif elem.name == "O5'":
            O5s.append(elem.index)
        elif elem.name == "P":
            Ps.append(elem.index)
        elif elem.name == "C4'":
            H5s.append(elem.index)
        else:
            pass
    force = mm.CustomTorsionForce('-k0*(theta^2)')
    force.setForceGroup(2)
    force.addPerTorsionParameter('k0')
    for i in range(len(Ps)):
        force.addTorsion(H5s[i + 1], C5s[i + 1], O5s[i + 1], Ps[i], [20])
    system.addForce(force)
    print(C5s, O5s, Ps, H5s)


def get_aptamer(ligand_range, positions):
    return positions[ligand_range[1] - 1:]


def get_ligand(topology):
    ligand_indices = []
    for a in topology.atoms():
        if a.residue.name not in ["DGN", "DAN", "DTN", "DCN", "DG", "DA", "DT", "DC", "DG5", "DA5", "DT5", "DC5", "DG3", "DA3", "DT3", "DC3"]:
            ligand_indices.append(a.index)
    return ligand_indices


def get_ligand_range(topology):
    return [get_ligand(topology)[0], len(get_ligand(topology))]


def get_offset(positions_old, positions):
    vec_a = (positions[len(positions_old) - 1] - positions[len(positions_old) - 2])
    vec_b = (positions_old[len(positions_old) - 1] - positions_old[-2])
    alpha = math.acos(sum([alem.value_in_unit(unit.angstroms) * blem.value_in_unit(unit.angstroms) for alem, blem in zip(vec_a, vec_b)]) / (np.linalg.norm(vec_a.value_in_unit(unit.angstroms)) * np.linalg.norm(vec_b.value_in_unit(unit.angstroms))))
    alpha_t = 0.
    d_alpha = alpha_t - alpha
    axis = np.cross(vec_a.value_in_unit(unit.angstroms), vec_b.value_in_unit(unit.angstroms)) / np.linalg.norm(np.cross(vec_a.value_in_unit(unit.angstroms), vec_b.value_in_unit(unit.angstroms)))
    offset = positions_old[-1] - positions[len(positions_old) - 1]
    return d_alpha, axis, offset, vec_a, vec_b


def position_aptamer(positions_old, positions):
    alpha, axis, offset, vec_a, vec_b = get_offset(positions_old, positions)
    ps = positions
    phi_2 = (alpha / 2).real
    x, y, z = axis
    s = np.math.sin(phi_2)
    c = np.math.cos(phi_2)
    rot = np.array([[2 * (np.power(x, 2) - 1) * np.power(s, 2) + 1, 2 * x * y * np.power(s, 2) - 2 * z * c * s,
                     2 * x * z * np.power(s, 2) + 2 * y * c * s],
                    [2 * x * y * np.power(s, 2) + 2 * z * c * s, 2 * (np.power(y, 2) - 1) * np.power(s, 2) + 1,
                     2 * z * y * np.power(s, 2) - 2 * x * c * s],
                    [2 * x * z * np.power(s, 2) - 2 * y * c * s, 2 * z * y * np.power(s, 2) + 2 * x * c * s,
                     2 * (np.power(z, 2) - 1) * np.power(s, 2) + 1]])

    for j in range(len(positions_old) - 2, len(positions)):
        roted = np.dot(np.array(positions[j].value_in_unit(unit.angstrom)), rot)
        ps[j] = mm.Vec3(roted[0], roted[1], roted[2]) * unit.angstrom
    drift = positions_old[-1] - ps[len(positions_old) - 1]
    for j in range(len(positions_old) - 2, len(positions)):
        ps[j] += drift + vec_b.value_in_unit(unit.angstroms) / np.linalg.norm(
            vec_b.value_in_unit(unit.angstroms)) * .6 * unit.angstroms
    positions_new = positions_old[:-1] + ps[len(positions_old) - 1:]
    return positions_new


def get_offset_five(positions_old, positions, ligand_length):
    vec_a = (positions[ligand_length + (len(positions) - len(positions_old)) + 1] - positions[
        ligand_length + (len(positions) - len(positions_old))])
    vec_b = (positions_old[ligand_length + 1] - positions_old[ligand_length])
    alpha = math.acos(sum([alem.value_in_unit(unit.angstroms) * blem.value_in_unit(unit.angstroms) for alem, blem in
                           zip(vec_a, vec_b)]) / (np.linalg.norm(vec_a.value_in_unit(unit.angstroms)) * np.linalg.norm(vec_b.value_in_unit(unit.angstroms))))
    alpha_t = 0.
    # alpha_t = math.pi-113.3*math.pi/360.
    d_alpha = alpha_t - alpha
    axis = np.cross(vec_a.value_in_unit(unit.angstroms), vec_b.value_in_unit(unit.angstroms)) / np.linalg.norm(np.cross(vec_a.value_in_unit(unit.angstroms), vec_b.value_in_unit(unit.angstroms)))
    offset = positions_old[-1] - positions[len(positions_old) - 1]
    return d_alpha, axis, offset, vec_a, vec_b


def var_to_ratio(var):
    var = np.array(var)
    res = var / var.sum()
    return res


def uniform_strat(x_var, y_var, z_var, i_var, j_var, k_var, phi_var, size, phi_size):
    x = var_to_ratio(x_var)
    y = var_to_ratio(y_var)
    z = var_to_ratio(z_var)
    i = var_to_ratio(i_var)
    j = var_to_ratio(j_var)
    k = var_to_ratio(k_var)
    phi = var_to_ratio(phi_var)
    size = np.array(size)

    stepx = (size[1] - size[0]) / len(x)
    stepy = (size[1] - size[0]) / len(y)
    stepz = (size[1] - size[0]) / len(z)
    stepi = (phi_size[1] - phi_size[0]) / len(i)
    stepj = (phi_size[1] - phi_size[0]) / len(j)
    stepk = (phi_size[1] - phi_size[0]) / len(k)
    stepphi = (phi_size[1] - phi_size[0]) / len(phi)

    distx = np.random.choice(np.array([np.random.uniform(size[0] + (l - 1) * stepx, size[0] + l * stepx) for l in range(1, len(x) + 1)]), p=np.append(x[:-1], [1 - sum(x[:-1])]))
    disty = np.random.choice(np.array([np.random.uniform(size[0] + (l - 1) * stepy, size[0] + l * stepy) for l in range(1, len(y) + 1)]), p=np.append(y[:-1], [1 - sum(y[:-1])]))
    distz = np.random.choice(np.array([np.random.uniform(size[0] + (l - 1) * stepz, size[0] + l * stepz) for l in range(1, len(z) + 1)]), p=np.append(z[:-1], [1 - sum(z[:-1])]))
    disti = np.random.choice(np.array([np.random.uniform(phi_size[0] + (l - 1) * stepi, phi_size[0] + l * stepi) for l in range(1, len(i) + 1)]), p=np.append(i[:-1], [1 - sum(i[:-1])]))
    distj = np.random.choice(np.array([np.random.uniform(phi_size[0] + (l - 1) * stepj, phi_size[0] + l * stepj) for l in range(1, len(j) + 1)]), p=np.append(j[:-1], [1 - sum(j[:-1])]))
    distk = np.random.choice(np.array([np.random.uniform(phi_size[0] + (l - 1) * stepk, phi_size[0] + l * stepk) for l in range(1, len(k) + 1)]), p=np.append(k[:-1], [1 - sum(k[:-1])]))
    distphi = np.random.choice(np.array([np.random.uniform(phi_size[0] + (l - 1) * stepphi, phi_size[0] + l * stepphi) for l in range(1, len(phi) + 1)]), p=np.append(phi[:-1], [1 - sum(phi[:-1])]))

    return distx, disty, distz, disti, distj, distk, distphi


class Aptamer:
    def __init__(self, inner_forcefield_name, ligand_mol2_path):
        self.process = spawn('tleap -f' + inner_forcefield_name)
        self.process.expect('>')
        self.process.sendline('source leaprc.gaff')
        self.process.expect('>')
        self.process.sendline("set default PBradii mbondi2")
        self.process.expect('>')
        self.process.sendline("ligand = load" + _FORMAT + " " + ligand_mol2_path)
        if _HYBRID:
            self.process.expect('>')
            self.process.sendline("loadamberparams " + _HYBRID)
        self.geometry = []
        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0]

    def command(self, command_text):
        self.process.sendline(command_text)
        self.process.expect('>', timeout=None)

    def unify(self, identifier):
        self.command("union = combine { ligand " + identifier + " }")

    def sequence(self, identifier, string_of_residues):
        inputstring = "{"+string_of_residues+"}"
        self.command(identifier+" = sequence "+inputstring)


def get_PO3(positions_old, positions):
    pos = positions
    vec_a = (positions[len(positions_old) - 1] - positions[len(positions_old) - 2])
    x, y, z = vec_a.value_in_unit(unit.angstroms)
    x, y, z = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
    shift_forward = mm.Vec3(0, 0, 0) * unit.angstroms - positions[len(positions_old) - 1]
    phi_2 = np.random.uniform(-np.math.pi / 2, np.math.pi / 2)
    s = np.math.sin(phi_2)
    c = np.math.cos(phi_2)
    rot = np.array([[2 * (np.power(x, 2) - 1) * np.power(s, 2) + 1, 2 * x * y * np.power(s, 2) - 2 * z * c * s,
                     2 * x * z * np.power(s, 2) + 2 * y * c * s],
                    [2 * x * y * np.power(s, 2) + 2 * z * c * s, 2 * (np.power(y, 2) - 1) * np.power(s, 2) + 1,
                     2 * z * y * np.power(s, 2) - 2 * x * c * s],
                    [2 * x * z * np.power(s, 2) - 2 * y * c * s, 2 * z * y * np.power(s, 2) + 2 * x * c * s,
                     2 * (np.power(z, 2) - 1) * np.power(s, 2) + 1]])

    for j in range(len(positions_old) - 1, len(positions)):
        pos[j] += shift_forward

    for j in range(len(positions_old) - 1, len(positions)):
        roted = np.dot(np.array(pos[j].value_in_unit(unit.angstrom)), rot)
        pos[j] = mm.Vec3(roted[0], roted[1], roted[2]) * unit.angstrom
        pos[j] -= shift_forward

    positions_new = pos
    return positions_new


def get_PO5(positions_old, positions):
    pos = positions
    vec_a = (positions[len(positions_old) + 2] - positions[len(positions_old) - 1])
    x, y, z = vec_a.value_in_unit(unit.angstroms)
    x, y, z = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
    shift_forward = mm.Vec3(0, 0, 0) * unit.angstroms - positions[len(positions_old) + 2]
    phi_2 = np.random.uniform(-np.math.pi / 2, np.math.pi / 2)
    s = np.math.sin(phi_2)
    c = np.math.cos(phi_2)
    rot = np.array([[2 * (np.power(x, 2) - 1) * np.power(s, 2) + 1, 2 * x * y * np.power(s, 2) - 2 * z * c * s,
                     2 * x * z * np.power(s, 2) + 2 * y * c * s],
                    [2 * x * y * np.power(s, 2) + 2 * z * c * s, 2 * (np.power(y, 2) - 1) * np.power(s, 2) + 1,
                     2 * z * y * np.power(s, 2) - 2 * x * c * s],
                    [2 * x * z * np.power(s, 2) - 2 * y * c * s, 2 * z * y * np.power(s, 2) + 2 * x * c * s,
                     2 * (np.power(z, 2) - 1) * np.power(s, 2) + 1]])

    for j in range(len(positions_old) + 2, len(positions)):
        pos[j] += shift_forward

    for j in range(len(positions_old) + 2, len(positions)):
        # pos[j] += drift
        roted = np.dot(np.array(pos[j].value_in_unit(unit.angstrom)), rot)
        pos[j] = mm.Vec3(roted[0], roted[1], roted[2]) * unit.angstrom
        pos[j] -= shift_forward

    positions_new = pos
    return positions_new


def get_C5O5(positions_old, positions):
    pos = positions
    vec_a = (positions[len(positions_old) + 3] - positions[len(positions_old) + 2])
    x, y, z = vec_a.value_in_unit(unit.angstroms)
    x, y, z = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
    shift_forward = mm.Vec3(0, 0, 0) * unit.angstroms - positions[len(positions_old) + 3]
    phi_2 = np.random.uniform(-np.math.pi / 2, np.math.pi / 2)
    s = np.math.sin(phi_2)
    c = np.math.cos(phi_2)
    rot = np.array([[2 * (np.power(x, 2) - 1) * np.power(s, 2) + 1, 2 * x * y * np.power(s, 2) - 2 * z * c * s,
                     2 * x * z * np.power(s, 2) + 2 * y * c * s],
                    [2 * x * y * np.power(s, 2) + 2 * z * c * s, 2 * (np.power(y, 2) - 1) * np.power(s, 2) + 1,
                     2 * z * y * np.power(s, 2) - 2 * x * c * s],
                    [2 * x * z * np.power(s, 2) - 2 * y * c * s, 2 * z * y * np.power(s, 2) + 2 * x * c * s,
                     2 * (np.power(z, 2) - 1) * np.power(s, 2) + 1]])

    for j in range(len(positions_old) + 3, len(positions)):
        pos[j] += shift_forward

    for j in range(len(positions_old) + 3, len(positions)):
        # pos[j] += drift
        roted = np.dot(np.array(pos[j].value_in_unit(unit.angstrom)), rot)
        pos[j] = mm.Vec3(roted[0], roted[1], roted[2]) * unit.angstrom
        pos[j] -= shift_forward

    positions_new = pos
    return positions_new


def get_C5(positions_old, positions):
    pos = positions
    vec_a = (positions[len(positions_old) + 6] - positions[len(positions_old) + 3])
    x, y, z = vec_a.value_in_unit(unit.angstroms)
    x, y, z = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
    shift_forward = mm.Vec3(0, 0, 0) * unit.angstroms - positions[len(positions_old) + 6]
    phi_2 = np.random.uniform(-np.math.pi / 2, np.math.pi / 2)
    s = np.math.sin(phi_2)
    c = np.math.cos(phi_2)
    rot = np.array([[2 * (np.power(x, 2) - 1) * np.power(s, 2) + 1, 2 * x * y * np.power(s, 2) - 2 * z * c * s,
                     2 * x * z * np.power(s, 2) + 2 * y * c * s],
                    [2 * x * y * np.power(s, 2) + 2 * z * c * s, 2 * (np.power(y, 2) - 1) * np.power(s, 2) + 1,
                     2 * z * y * np.power(s, 2) - 2 * x * c * s],
                    [2 * x * z * np.power(s, 2) - 2 * y * c * s, 2 * z * y * np.power(s, 2) + 2 * x * c * s,
                     2 * (np.power(z, 2) - 1) * np.power(s, 2) + 1]])

    for j in range(len(positions_old) + 6, len(positions)):
        pos[j] += shift_forward

    for j in range(len(positions_old) + 6, len(positions)):
        # pos[j] += drift
        roted = np.dot(np.array(pos[j].value_in_unit(unit.angstrom)), rot)
        pos[j] = mm.Vec3(roted[0], roted[1], roted[2]) * unit.angstrom
        pos[j] -= shift_forward

    positions_new = pos
    return positions_new


def get_base(positions_old, positions):
    pos = positions
    vec_a = (positions[len(positions_old) + 11] - positions[len(positions_old) + 9])
    x, y, z = vec_a.value_in_unit(unit.angstroms)
    x, y, z = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
    shift_forward = mm.Vec3(0, 0, 0) * unit.angstroms - positions[len(positions_old) + 11]
    phi_2 = np.random.uniform(-np.math.pi / 2, np.math.pi / 2)
    s = np.math.sin(phi_2)
    c = np.math.cos(phi_2)
    rot = np.array([[2 * (np.power(x, 2) - 1) * np.power(s, 2) + 1, 2 * x * y * np.power(s, 2) - 2 * z * c * s,
                     2 * x * z * np.power(s, 2) + 2 * y * c * s],
                    [2 * x * y * np.power(s, 2) + 2 * z * c * s, 2 * (np.power(y, 2) - 1) * np.power(s, 2) + 1,
                     2 * z * y * np.power(s, 2) - 2 * x * c * s],
                    [2 * x * z * np.power(s, 2) - 2 * y * c * s, 2 * z * y * np.power(s, 2) + 2 * x * c * s,
                     2 * (np.power(z, 2) - 1) * np.power(s, 2) + 1]])
    end = 0
    # print(len(positions)-len(positions_old))
    if len(positions) - len(positions_old) == 30:
        end = len(positions_old) + 24
    elif len(positions) - len(positions_old) == 32:
        end = len(positions_old) + 26
    elif len(positions) - len(positions_old) == 33:
        end = len(positions_old) + 27

    for j in range(len(positions_old) + 11, end - 1):
        pos[j] += shift_forward

    for j in range(len(positions_old) + 11, end - 1):
        # pos[j] += drift
        roted = np.dot(np.array(pos[j].value_in_unit(unit.angstrom)), rot)
        pos[j] = mm.Vec3(roted[0], roted[1], roted[2]) * unit.angstrom
        pos[j] -= shift_forward

    positions_new = pos
    return positions_new


def initial_sample(topology, coordinates, Nsteps, index, box=50, rang=(0, 60)):
    print("Index is: ", index)
    aptamer_top = topology
    aptamer_crd = coordinates
    en = []
    xyz = []
    free_E_old = 1e50
    cnt = 0
    centre = np.math.ceil((rang[1] - rang[0]) / 2)
    system = aptamer_top.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, implicitSolvent=app.OBC1)
    integrator = mm.LangevinIntegrator(300. * unit.kelvin, 1. / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = app.Simulation(aptamer_top.topology, system, integrator)

    for i in range(Nsteps):
        pos = aptamer_crd.positions
        pos0 = aptamer_crd.positions[int(centre)]
        shift = mm.Vec3(np.random.uniform(-box, box), np.random.uniform(-box, box), np.random.uniform(-box, box)) * unit.angstrom
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        x, y, z = np.array([x, y, z]) * 1 / (np.linalg.norm(np.array([x, y, z])))

        phi_2 = np.random.uniform(-np.math.pi, np.math.pi)
        x, y, z = np.array([x, y, z]) * 1 / (np.linalg.norm(np.array([x, y, z])))
        xyz.append([shift[0].value_in_unit(unit.angstroms), shift[1].value_in_unit(unit.angstroms), shift[2].value_in_unit(unit.angstroms), x, y, z, phi_2])

        s = np.math.sin(phi_2)
        c = np.math.cos(phi_2)
        rot = np.array([[2 * (np.power(x, 2) - 1) * np.power(s, 2) + 1, 2 * x * y * np.power(s, 2) - 2 * z * c * s,
                         2 * x * z * np.power(s, 2) + 2 * y * c * s],
                        [2 * x * y * np.power(s, 2) + 2 * z * c * s, 2 * (np.power(y, 2) - 1) * np.power(s, 2) + 1,
                         2 * z * y * np.power(s, 2) - 2 * x * c * s],
                        [2 * x * z * np.power(s, 2) - 2 * y * c * s, 2 * z * y * np.power(s, 2) + 2 * x * c * s,
                         2 * (np.power(z, 2) - 1) * np.power(s, 2) + 1]])

        drift = get_aptamer(get_ligand_range(aptamer_top.topology), aptamer_crd.positions)[10]
        for j in range(get_ligand_range(aptamer_top.topology)[1], len(pos)):
            pos[j] -= drift
        for j in range(0, get_ligand_range(aptamer_top.topology)[1]):
            pos[j] -= pos0

        for j in range(get_ligand_range(aptamer_top.topology)[1], len(pos)):
            roted = np.dot(np.array(pos[j].value_in_unit(unit.angstrom)), rot)
            pos[j] = mm.Vec3(roted[0], roted[1], roted[2]) * unit.angstrom
            pos[j] += shift
        simulation.context.setPositions(pos)
        state = simulation.context.getState(getPositions=True, getEnergy=True, groups=1)
        free_E = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        en.append(free_E)
        if free_E < free_E_old:
            free_E_old = free_E
            cnt += 1
            fil = open("montetest%s.pdb" % i, "w")
            app.PDBFile.writeModel(aptamer_top.topology, pos, file=fil, modelIndex=i)
            fil.close()

    return en, pos, xyz, free_E, index


def mcmc_sample(topology, coordinates, old_coordinates, n_steps=5000):
    aptamer_top = topology
    aptamer_crd = coordinates
    pos = old_coordinates
    system = aptamer_top.createSystem(nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.2 * unit.nanometers, constraints=app.HBonds, implicitSolvent=app.OBC1)
    # print(index,index,index,index)
    integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)
    simulation = app.Simulation(aptamer_top.topology, system, integrator)
    en = []
    positions = []
    free_E_old = 1e20
    simulation.context.setPositions(get_C5(pos, get_C5O5(pos, get_base(pos, get_PO5(pos, get_PO3(pos, position_aptamer(pos, aptamer_crd.positions)))))))

    for i in range(n_steps):
        simulation.context.setPositions(get_base(pos, get_C5(pos, get_C5O5(pos, get_PO5(pos, get_PO3(pos, position_aptamer(pos,aptamer_crd.positions)))))))
        state = simulation.context.getState(getPositions=True, getEnergy=True, groups=1)
        free_E = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        fil = open("montestep%s.pdb" % i, "w")
        app.PDBFile.writeModel(aptamer_top.topology, state.getPositions(), file=fil, modelIndex=i)
        fil.close()

        if free_E < free_E_old:
            positions = state.getPositions()
        en.append(free_E)

    return en, positions, free_E


def initial(ntide):
    beta = _BETA
    print("Constructing Ligand/Aptamer complex ...")

    internal = Aptamer("leaprc.ff12SB", _INFILE)
    internal.sequence(ntide, ntide)
    internal.unify(ntide)
    internal.command("saveamberparm union %s.prmtop %s.inpcrd" % (ntide, ntide))

    print("Aptamer/Ligand complex constructed.")
    print("Loading Aptamer/Ligand complex ...")

    sleep(5)  # for some reason a pause is needed here

    aptamer_top = app.AmberPrmtopFile("%s.prmtop" % ntide)
    aptamer_crd = app.AmberInpcrdFile("%s.inpcrd" % ntide)

    ligand_range = get_ligand_range(aptamer_top.topology)
    sample_box = 5
    volume = (2 * sample_box) ** 3 * (2 * math.pi) ** 3
    print("Sampling parameter space ...")
    print(ntide)
    en_pos_xyz = [initial_sample(aptamer_top, aptamer_crd, _NINIT, i, box=sample_box, rang=ligand_range) for i in range(100)] # RESET TO AFTER TESTING!!!100)]

    print("done.")
    print("Harvesting results ...")
    en = []
    xyz = []
    positions_s = []
    for elem in en_pos_xyz:
        en += elem[0]
        positions_s.append([elem[3], elem[1]])
        xyz += elem[2]
    positions = min(positions_s)[1]

    Z = volume * sum([math.exp(-beta * elem) for elem in en]) / len(en)
    P = [math.exp(-beta * elem) / Z for elem in en]
    S = volume * sum([-elem * math.log(elem * volume) for elem in P]) / len(P)
    print("Ntide: %s entropy: %s" % (ntide, S))

    return positions, ntide, S


def evaluate(positions, Ntides, entropies, threshold=0.1):
    res_Ntide_positions = []
    print("Chosen ntides with entropies:")
    for pos, alem, blem in zip(positions, Ntides, entropies):
        if blem <= min(entropies) + threshold:
            res_Ntide_positions.append([pos, alem])
            print("%s : %s" % (alem, blem))
    return res_Ntide_positions


def step(array):
    beta = _BETA
    old_positions, Ntides, is_3prime = array
    internal = Aptamer("leaprc.ff12SB", _INFILE)
    identifier = Ntides.replace(" ", "")
    internal.sequence(identifier, Ntides.strip())
    internal.unify(identifier)
    internal.command("saveamberparm union %s.prmtop %s.inpcrd" % (identifier, identifier))

    sleep(5)  # for some reason a pause is needed here

    print("Identifier: " + Ntides)

    volume = (2 * math.pi) ** 5
    aptamer_top = app.AmberPrmtopFile("%s.prmtop" % identifier)
    aptamer_crd = app.AmberInpcrdFile("%s.inpcrd" % identifier)
    en_pos = [mcmc_sample(aptamer_top, aptamer_crd, old_positions, n_steps=_NSTEP) for index in range(10)]

    en = []
    positions_s = []
    for elem in en_pos:
        en += elem[0]
        positions_s.append([elem[2], elem[1]])

    positions = min(positions_s)[1]

    fil = open("best_structure%s.pdb" % Ntides, "w")
    app.PDBFile.writeModel(aptamer_top.topology, positions, file=fil)
    fil.close()

    Z = volume * math.fsum([math.exp(-beta * elem) for elem in en]) / len(en)
    P = [math.exp(-beta * elem) / Z for elem in en]
    S = volume * math.fsum([-elem * math.log(elem * volume) for elem in P]) / len(P)

    print("%s : %s" % (Ntides, S))

    return positions, Ntides, S


def loop():
    alphabet = ["DGN", "DAN", "DTN", "DCN"]

    print(alphabet)
    print("Choosing from candidates ...")

    pos_Nt_S = []
    for al in alphabet:
        pos_Nt_S.append(initial(al))

    positions = [elem[0] for elem in pos_Nt_S]
    lntides = [elem[1] for elem in pos_Nt_S]
    entropies = [elem[2] for elem in pos_Nt_S]

    pos_nt = evaluate(positions, lntides, entropies, threshold=0.5)

    positions = []
    lntides = []
    for elem in pos_nt:
        positions.append(elem[0])
        lntides.append(elem[1])

    print([len(elem) for elem in positions])

    print("Chosen nucleotides: ")
    print(lntides)

    for i in range(_NMER):
        if i == 0:
            print("Initializing 2nd step ...")
        elif i == 1:
            print("Initializing 3rd step ...")
        else:
            print("Initializing %sth step ..." % (i + 2))

        pos_Nt_S = []
        for ak in [[alem, blem.replace("3", "").replace("N", "5").strip() + ntide, 1] for alem, blem in zip(positions, lntides) for ntide in [" DG3", " DA3", " DT3", " DC3"]]:
            pos_Nt_S.append(step(ak))

        positions = [elem[0] for elem in pos_Nt_S]
        lntides = [elem[1] for elem in pos_Nt_S]
        entropies = [elem[2] for elem in pos_Nt_S]

        pos_nt = evaluate(positions, lntides, entropies, threshold=0.005)
        lntides = []
        positions = []
        for elem in pos_nt:
            lntides.append(elem[1])
        for elem in pos_nt:
            positions.append(elem[0])
        print("Chosen sequences are: ")
        for elem in pos_nt:
            print(elem[1])

    return pos_nt


def result(pos_Nt):
    """Write a lot of files"""
    count = 0
    for elem in pos_Nt:
        internal = Aptamer("leaprc.ff12SB", _INFILE)
        identifier = elem[1].replace(" ", "")
        internal.sequence(identifier, elem[1])
        internal.unify(identifier)
        internal.command("saveamberparm union Aptamer%s.prmtop Aptamer%s.inpcrd" % (count, count))

        sleep(5)

    print("Run successful! Have fun with your Aptamers")
    return 1


parser = argparse.ArgumentParser(description='MAWS - Make Aptamers Without SELEX',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('path', metavar='PATH', help='Path to the calculation directory.')
parser.add_argument('infile', metavar='INFILE', help='3D structure of the ligand.')
parser.add_argument('-b', '--beta', type=float, default=0.01, help='lagrange multiplier beta.')
parser.add_argument('-i', '--ninit', type=int, default=200, help='number of initial steps as multiple of 100.')
parser.add_argument('-s', '--nstep', type=int, default=200, help='number of samples in every step after the first.')
parser.add_argument('-l', '--nmer', type=int, default=15, help='The final length of the aptamer.')
parser.add_argument('-f', '--format', type=str, choices=['pdb', 'mol2'], default="pdb", help='input file format.')
parser.add_argument('-y', '--hybrid', type=str, default="", help='parameter modifying file for hybrid calculations')

args = parser.parse_args()

_INFILE = args.infile
_BETA = args.beta
_NINIT = args.ninit
_NSTEP = args.nstep
_NMER = args.nmer
_HYBRID = args.hybrid
_FORMAT = args.format

positions_and_Ntides = loop()
result(positions_and_Ntides)

print("Run successful!")
print("Please come again!")
