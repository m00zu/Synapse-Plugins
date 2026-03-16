"""
Ported Meeko PDBQT-writing classes (MoleculePreparation, PDBQTWriterLegacy).

Originally from MolDocker's meeko_functions.py — a self-contained reimplementation
of the Meeko ligand preparation pipeline.  Only the classes needed for
RDKit Mol → PDBQT string conversion are included here.
"""

import os
import sys
import json
import math
import warnings
import numpy as np

from io import StringIO
from copy import deepcopy
from inspect import signature
from operator import itemgetter
from collections import OrderedDict, defaultdict

from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdPartialCharges
from collections import namedtuple

# DeprecationWarning is not displayed by default
warnings.filterwarnings("default", category=DeprecationWarning)

### RDKitMoleculeSetup ###
class HJKRingDetection:
    """Implementation of the Hanser-Jauffret-Kaufmann exhaustive ring detection
    algorithm:
        ref:
        Th. Hanser, Ph. Jauffret, and G. Kaufmann
        J. Chem. Inf. Comput. Sci. 1996, 36, 1146-1152
    """

    def __init__(self, mgraph, max_iterations=8000000):
        self.mgraph = {key: [x for x in values] for (key, values) in mgraph.items()}
        self.rings = []
        self._iterations = 0
        self._max_iterations = max_iterations
        self._is_failed = False

    def scan(self, keep_chorded_rings=False, keep_equivalent_rings=False):
        """run the full protocol for exhaustive ring detection
        by default, only chordless rings are kept, and equivalent rings removed.
        (equivalent rings are rings that have the same size and share the same
        neighbors)
        """
        self.prune()
        self.build_pgraph()
        self.vertices = self._get_sorted_vertices()
        while self.vertices:
            self._remove_vertex(self.vertices[0])
        if not keep_chorded_rings:
            self.find_chordless_rings(keep_equivalent_rings)
        output_rings = []
        for ring in self.rings:
            output_rings.append(tuple(ring[:-1]))
        return output_rings

    def _get_sorted_vertices(self):
        """function to return the vertices to be removed, sorted by increasing
        connectivity order (see paper)"""
        vertices = ((k, len(v)) for k, v in self.mgraph.items())
        return [x[0] for x in sorted(vertices, key=itemgetter(1))]

    def prune(self):
        """iteratively prune graph until there are no leafs left (nodes with only
        one connection)"""
        while True:
            prune = []
            for node, neighbors in self.mgraph.items():
                if len(neighbors) == 1:
                    prune.append((node, neighbors))
            if len(prune) == 0:
                break
            for node, neighbors in prune:
                self.mgraph.pop(node)
                for n in neighbors:
                    self.mgraph[n].remove(node)

    def build_pgraph(self, prune=True):
        """convert the M-graph (molecular graph) into the P-graph (path/bond graph)"""
        self.pgraph = []
        for node, neigh in self.mgraph.items():
            for n in neigh:
                # use sets for unique id
                edge = set((node, n))
                if not edge in self.pgraph:
                    self.pgraph.append(edge)
        # re-convert the edges to lists because order matters in cycle detection
        self.pgraph = [list(x) for x in self.pgraph]

    def _remove_vertex(self, vertex):
        """remove a vertex and join all edges connected by that vertex (this is
        the REMOVE function from the paper)
        """
        visited = {}
        remove = []
        pool = []
        for path in self.pgraph:
            if self._has_vertex(vertex, path):
                pool.append(path)
        for i, path1 in enumerate(pool):
            for j, path2 in enumerate(pool):
                if i == j:
                    continue
                self._iterations += 1
                if self._iterations > self._max_iterations:
                    self._is_failed = True
                    break
                pair_id = tuple(set((i, j)))
                if pair_id in visited:
                    continue
                visited[pair_id] = None
                common = list(set(path1) & set(path2))
                common_count = len(common)
                # check if two paths have only this vertex in common or (or
                # two, if they're a cycle)
                if not 1 <= common_count <= 2:
                    continue
                # generate the joint path
                joint_path = self._concatenate_path(path1, path2, vertex)
                is_ring = joint_path[0] == joint_path[-1]
                # if paths share more than two vertices but they're not a ring, then skip
                if (common_count == 2) and not is_ring:
                    continue
                # store the ring...
                if is_ring:
                    self._add_ring(joint_path)
                # ...or the common path
                elif not joint_path in self.pgraph:
                    self.pgraph.append(joint_path)
        # remove used paths
        for p in pool:
            self.pgraph.remove(p)
        # remove the used vertex
        self.vertices.remove(vertex)

    def _add_ring(self, ring):
        """add newly found rings to the list (if not already there)"""
        r = set(ring)
        for candidate in self.rings:
            if r == set(candidate):
                return
        self.rings.append(ring)

    def _has_vertex(self, vertex, edge):
        """check if the vertex is part of this edge, and if true, return the
        sorted edge so that the vertex is the first in the list"""
        if edge[0] == vertex:
            return edge
        if edge[-1] == vertex:
            return edge[::-1]
        return None

    def _concatenate_path(self, path1, path2, v):
        """concatenate two paths sharing a common vertex
        a-b, c-b => a-b-c : idx1=1, idx2=1
        b-a, c-b => a-b-c : idx1=0, idx2=1
        a-b, b-c => a-b-c : idx1=1, idx2=0
        b-a, b-c => a-b-c : idx1=0, idx2=0
        """
        if not path1[-1] == v:
            path1.reverse()
        if not path2[0] == v:
            path2.reverse()
        return path1 + path2[1:]

    def _edge_in_pgraph(self, edge):
        """check if edge is already in pgraph"""
        e = set(edge)
        for p in self.pgraph:
            if e == set(p) and len(p) == len(edge):
                return True
        return False

    def find_chordless_rings(self, keep_equivalent_rings):
        """find chordless rings: cycles in which two vertices are not connected
        by an edge that does not itself belong to the cycle (Source:
        https://en.wikipedia.org/wiki/Cycle_%28graph_theory%29#Chordless_cycle)

        - iterate through rings starting from the smallest ones: A,B,C,D...
        - for each ring (A), find a candidate (e.g.: B) that is smaller and shares at least an edge
        - for this pair, calculate the two differences (A-B and B-A) in the list of edges of each
        - if  ( (A-B) + (B-A) ) a smaller ring (e.g.: C), then the current ring has a chord
        """
        # sort rings by the smallest to largest
        self.rings.sort(key=len, reverse=False)
        chordless_rings = []
        ring_edges = []
        rings_set = [set(x) for x in self.rings]
        for r in self.rings:
            edges = []
            for i in range(len(r) - 1):
                edges.append(
                    tuple(
                        set((r[i], r[(i + 1) % len(r)])),
                    )
                )
            edges = sorted(edges, key=itemgetter(0))
            ring_edges.append(edges)
        ring_contacts = {}
        for i, r1 in enumerate(self.rings):
            chordless = True
            r1_edges = ring_edges[i]
            ring_contacts[i] = []
            for j, r2 in enumerate(self.rings):
                if i == j:
                    continue
                if len(r2) >= len(r1):
                    # the candidate ring is larger than or the same size of the candidate
                    continue
                # avoid rings that don't share at least an edge
                # shared = set(r1) & set(r2)
                r2_edges = ring_edges[j]
                shared = set(r1_edges) & set(r2_edges)
                if len(shared) < 1:
                    continue
                ring_contacts[i].append(j)
                # get edges difference (r2_edges - r1_edges)
                core_edges = [x for x in r2_edges if not x in r1_edges]
                chord = [x for x in r1_edges if not x in r2_edges]
                # combined = chord + core_edges
                ring_new = []
                for edge in chord + core_edges:
                    ring_new.append(edge[0])
                    ring_new.append(edge[1])
                ring_new = set(ring_new)
                if (ring_new in rings_set) and (len(ring_new) < len(r1) - 1):
                    chordless = False
                    break
            if chordless:
                chordless_rings.append(i)
                ring_contacts[i] = set(ring_contacts[i])
        if not keep_equivalent_rings:
            chordless_rings = self._remove_equivalent_rings(
                chordless_rings, ring_contacts
            )
        self.rings = [self.rings[x] for x in chordless_rings]
        return

    def _remove_equivalent_rings(self, chordless_rings, ring_contacts):
        """remove equivalent rings by clustering by size, then by ring neighbors.
        Two rings A and B are equivalent if satisfy the following conditions:
            - same size
            - same neighbor ring(s) [C,D, ...]
            - (A - C) == (B -C)
        """
        size_clusters = {}
        # cluster rings by their size
        for ring_id in chordless_rings:
            if len(ring_contacts[ring_id]) == 0:
                continue
            size = len(self.rings[ring_id]) - 1
            if not size in size_clusters:
                size_clusters[size] = []
            size_clusters[size].append(ring_id)
        remove = []
        # process rings of the same size
        for size, ring_pool in size_clusters.items():
            for ri in ring_pool:
                if ri in remove:
                    continue
                for rj in ring_pool:
                    if ri == rj:
                        continue
                    common_neigh = ring_contacts[ri] & ring_contacts[rj]
                    for c in common_neigh:
                        d1 = set(self.rings[ri]) - set(self.rings[c])
                        d2 = set(self.rings[rj]) - set(self.rings[c])
                        if d1 == d2:
                            remove.append(rj)
        chordless_rings = [i for i in chordless_rings if not i in set(remove)]
        # for r in set(remove):
        #    chordless_rings.remove(r)
        return chordless_rings

mini_periodic_table = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
}

PDBAtomInfo = namedtuple('PDBAtomInfo', "name resName resNum chain")

def getPdbInfoNoNull(atom):
    """extract information for populating an ATOM/HETATM line
    in the PDB"""
    # res = atom.GetResidue()
    minfo = atom.GetMonomerInfo()
    if minfo is None:
        atomic_number = atom.GetAtomicNum()
        if atomic_number == 0:
            name = '%-2s' % '*'
        else:
            name = '%-2s' % mini_periodic_table[atomic_number]
        chain = ' '
        resNum = 1
        resName = 'UNL'
    else:
        name = minfo.GetName()
        chain = minfo.GetChainId()
        resNum = minfo.GetResidueNumber()
        resName = minfo.GetResidueName()
    return PDBAtomInfo(name=name, resName=resName, resNum=resNum, chain=chain)

class MoleculeSetup:
    """ mol: molecule structurally prepared with explicit hydrogens

        the setup provides:
            - data storage
            - SMARTS matcher for all atom typers
    """

    def __init__(self):
        self.atom_pseudo = []
        self.coord = OrderedDict()  # FIXME all OrderedDict shuold be converted to lists?
        self.charge = OrderedDict()
        self.pdbinfo = OrderedDict()
        self.atom_type = OrderedDict()
        self.atom_ignore = OrderedDict()
        self.chiral = OrderedDict()
        self.atom_true_count = 0
        self.graph = OrderedDict()
        self.bond = OrderedDict()
        self.element = OrderedDict()
        self.interaction_vector = OrderedDict()
        self.flexibility_model = {}
        self.ring_closure_info = {
            "bonds_removed": [],
            "pseudos_by_atom": {},
        }
        # ring information
        self.rings = {}
        self.rings_aromatic = []
        self.atom_to_ring_id = defaultdict(list)
        self.ring_corners = {}  # used to store corner flexibility
        self.name = None

    def copy(self):
        newsetup = MoleculeSetup()
        newsetup.__dict__ = deepcopy(self.__dict__)
        return newsetup

    def add_atom(self, idx=None, coord=np.array([0.0, 0.0, 0.0], dtype='float'),
            element=None, charge=0.0, atom_type=None, pdbinfo=None, neighbors=None,
            ignore=False, chiral=False, overwrite=False):
        """ function to add all atom information at once;
            every property is optional
        """
        if idx is None:
            idx = len(self.coord)
        if idx in self.coord and not overwrite:
            print("ADD_ATOM> Error: the idx [%d] is already occupied (use 'overwrite' to force)")
            return False
        self.set_coord(idx, coord)
        self.set_charge(idx, charge)
        self.set_element(idx, element)
        self.set_atom_type(idx, atom_type)
        self.set_pdbinfo(idx, pdbinfo)
        if neighbors is None:
            neighbors = []
        self.set_neigh(idx, neighbors)
        self.set_chiral(idx, chiral)
        self.set_ignore(idx, ignore)
        return idx

    def del_atom(self, idx):
        """ remove an atom and update all data associate with it """
        pass
        # coords
        # charge
        # element
        # type
        # neighbor graph
        # chiral
        # ignore
        # update bonds bonds (using the neighbor graph)
        # If pseudo-atom, update other information, too


    # pseudo-atoms
    def add_pseudo(self, coord=np.array([0.0,0.0,0.0], dtype='float'), charge=0.0,
            anchor_list=None, atom_type=None, bond_type=None, rotatable=False,
            pdbinfo=None, directional_vectors=None, ignore=False, chira0=False, overwrite=False):
        """ add a new pseudoatom
            multiple bonds can be specified in "anchor_list" to support the centroids of aromatic rings

            if rotatable, makes the anchor atom rotatable to allow the pseudoatom movement
        """
        idx = self.atom_true_count + len(self.atom_pseudo)
        if idx in self.coord and not overwrite:
            print("ADD_PSEUDO> Error: the idx [%d] is already occupied (use 'overwrite' to force)")
            return False
        self.atom_pseudo.append(idx)
        # add the pseudoatom information to the atoms
        self.add_atom(idx=idx, coord=coord,
                element=0,
                charge=charge,
                atom_type=atom_type,
                pdbinfo=pdbinfo,
                neighbors=[],
                ignore=ignore,
                overwrite=overwrite)
        # anchor atoms
        if not anchor_list is None:
            for anchor in anchor_list:
                self.add_bond(idx, anchor, 0, rotatable, bond_type=bond_type)
        # directional vectors
        if not directional_vectors is None:
            self.add_interaction_vector(idx, directional_vectors)
        return idx

    # Bonds
    def add_bond(self, idx1, idx2, order=0, rotatable=False, in_rings=None, bond_type=None):
        """ bond_type default: 0 (non rotatable) """
        # NOTE: in_ring is used during bond typing to keep endo-cyclic rotatable bonds (e.g., sp3)
        #       as non-rotatable. Possibly, this might be handled by the bond typer itself?
        #       the type would allow it
        # TODO check if in_rings should be checked by this function?
        if in_rings is None:
            in_rings = []
        if not idx2 in self.graph[idx1]:
            self.graph[idx1].append(idx2)
        if not idx1 in self.graph[idx2]:
            self.graph[idx2].append(idx1)
        self.set_bond(idx1, idx2, order, rotatable, in_rings, bond_type)

    # atom types
    def set_atom_type(self, idx, atom_type):
        """ set the atom type for atom index
        atom_type : int or str?
        return: void
        """
        self.atom_type[idx] = atom_type

    def get_atom_type(self, idx):
        """ return the atom type for atom index in the lookup table
        idx : int
        return: str
        """
        return self.atom_type[idx]

    # ignore flag
    def set_ignore(self, idx, state):
        """ set the ignore flag (bool) for the atom
        (used formerged hydrogen)
        idx: int
        state: bool
        """
        self.atom_ignore[idx] = state

    # charge
    def get_charge(self, idx):
        """ return partial charge for atom index
        idx: int

        """
        return self.charge[idx]

    def set_charge(self, idx, charge):
        """ set partial charge"""
        self.charge[idx] = charge

    def get_coord(self, idx):
        """ return coordinates of atom index"""
        return self.coord[idx]

    def set_coord(self, idx, coord):
        """ define coordinates of atom index"""
        self.coord[idx] = coord

    def get_neigh(self, idx):
        """ return atoms connected to atom index"""
        return self.graph[idx]

    def set_neigh(self, idx, neigh_list):
        """ update the molecular graph with the neighbor indices provided """
        if not idx in self.graph:
            self.graph[idx] = []
        for n in neigh_list:
            if not n in self.graph[idx]:
                self.graph[idx].append(n)
            if not n in self.graph:
                self.graph[n] = []
            if not idx in self.graph[n]:
                self.graph[n].append(idx)

    def set_chiral(self, idx, chiral):
        """ set chiral flag for atom """
        self.chiral[idx] = chiral

    def get_chiral(self, idx):
        """ get chiral flag for atom """
        return self.chiral[idx]

    def get_ignore(self, idx):
        """ return if the atom is ignored"""
        return bool(self.atom_ignore[idx])

    def is_aromatic(self, idx):
        """ check if atom is aromatic """
        for r in self.rings_aromatic:
            if idx in r:
                return True
        return False

    def set_element(self, idx, elem_num):
        """ set the atomic number of the atom idx"""
        self.element[idx] = elem_num

    def get_element(self, idx):
        """ return the atomic number of the atom idx"""
        return self.element[idx]

    # def get_atom_ring_count(self, idx):
    #     """ return the number of rings to which this atom belongs"""
    #     # FIXME this should be replaced by self.get_atom_rings()
    #     return len(self.atom_to_ring_id[idx])

    def get_atom_rings(self, idx):
        # FIXME this should replace self.get_atom_ring_count()
        """ return the list of rings to which the atom idx belongs"""
        if idx in self.atom_to_ring_id:
            return self.atom_to_ring_id[idx]
        return []

    def get_atom_indices(self, true_atoms_only=False):
        """ return the indices of the atoms registered in the setup
            if 'true_atoms_only' are requested, then pseudoatoms are ignored
        """
        indices = list(self.coord.keys())
        if true_atoms_only:
            return [ x for x in indices if not x in self.atom_pseudo ]
        return indices

    # interaction vectors
    def add_interaction_vector(self, idx, vector_list):
        """ add vector list to list of directional interaction vectors for atom idx"""
        if not idx in self.interaction_vector:
            self.interaction_vector[idx] = []
        for vec in vector_list:
            self.interaction_vector[idx].append(vec)

    # TODO evaluate if useful
    def _get_attrib(self, idx, attrib, default=None):
        """ generic function to provide a default for retrieving properties and returning standard values """
        return getattr(self, attrib).get(idx, default)

    def get_interaction_vector(self, idx):
        """ get list of directional interaction vectors for atom idx"""
        return self.interaction_vector[idx]

    def del_interaction_vector(self, idx):
        """ delete list of directional interaction vectors for atom idx"""
        del self.interaction_vector[idx]

    def set_pdbinfo(self, idx, data):
        """ add PDB data (resname/num, atom name, etc.) to the atom """
        self.pdbinfo[idx] = data

    def get_pdbinfo(self, idx):
        """ retrieve PDB data (resname/num, atom name, etc.) to the atom """
        return self.pdbinfo[idx]

    def set_bond(self, idx1, idx2, order=None, rotatable=None, in_rings=None, bond_type=None):
        """ populate bond lookup table with properties
            bonds are identified by any tuple of atom indices
            the function generates the canonical bond id

            order      : int
            rotatable  : bool
            in_rings   : list (rings to which the bond belongs)
            bond_type  : int
        """
        bond_id = self.get_bond_id(idx1, idx2)
        if order is None:
            order = 0
        if rotatable is None:
            rotatable = False
        if in_rings is None:
            in_rings = []
        self.bond[bond_id] = {'bond_order': order,
                              'type': bond_type,
                              'rotatable': rotatable,
                              'in_rings': in_rings}

    def del_bond(self, idx1, idx2):
        """ remove a bond from the lookup table """
        bond_id = self.get_bond_id(idx1, idx2)
        del self.bond[bond_id]
        self.graph[idx1].remove(idx2)
        # TODO check if we want to delete nodes that have no connections (we might want to keep them)
        if not self.graph[idx1]:
            del self.graph[idx1]
        self.graph[idx2].remove(idx1)
        if not self.graph[idx2]:
            del self.graph[idx2]

    def get_bond(self, idx1, idx2):
        """ return properties of a bond in the lookup table
            if the bond does not exist, None is returned

            idx1, idx2 : int

            return: dict or voidko
        """
        bond_idx = self.get_bond_id(idx1, idx2)
        try:
            return self.bond[bond_idx]
        except IndexError:
            return None

    @staticmethod
    def get_bond_id(idx1, idx2):
        """ used to generate canonical bond id from a pair of nodes in the graph"""
        idx_min = min(idx1, idx2)
        idx_max = max(idx1, idx2)
        return (idx_min, idx_max)

    # replaced by
    # def ring_atom_to_ring(self, arg):
    #     return self.atom_to_ring_id[arg]

    def get_bonds_in_ring(self, ring):
        """ input: 'ring' (list of atom indices)
            returns list of bonds in ring, each bond is a pair of atom indices
        """
        n = len(ring)
        bonds = []
        for i in range(n):
            bond = (ring[i], ring[(i+1) % n])
            bond = (min(bond), max(bond))
            bonds.append(bond)
        return bonds

    def walk_recursive(self, idx, collected=None, exclude=None):
        """ walk molecular graph and return subgraphs that are bond-connected"""
        if collected is None:
            collected = []
        if exclude is None:
            exclude = []
        for neigh in self.get_neigh(idx):
            if neigh in exclude:
                continue
            collected.append(neigh)
            exclude.append(neigh)
            self.walk_recursive(neigh, collected, exclude)
        return collected

    def copy_attributes_from(self, template):
        """ copy attributes to duplicate the template setup
        NOTE: the molecule will always keep the original setup (i.e., template)"""
        # TODO enable some kind of plugin system here too, to allow other objects to
        # add attributes?
        # TODO although, this would make the setup more fragile-> better have attributes
        # explicitely added here, and that's it
        for attr in self.attributes_to_copy:
            attr_copy = deepcopy(getattr(template, attr))
            setattr(self, attr, attr_copy)
        # TODO possible BUG? the molecule is shared by the different setups
        #      if one of them alters the molecule, properties will not be the same
        self.mol = template.mol

    def merge_terminal_atoms(self, indices):
        """for merging hydrogens, but will merge any atom or pseudo atom
            that is bonded to only one other atom"""

        for index in indices:
            if len(self.graph[index]) != 1:
                msg = "Atempted to merge atom %d with %d neighbors. "
                msg += "Only atoms with one neighbor can be merged."
                msg = msg % (index + 1, len(self.graph[index]))
                raise RuntimeError(msg)
            neighbor_index = self.graph[index][0]
            self.charge[neighbor_index] += self.get_charge(index)
            self.charge[index] = 0.0
            self.set_ignore(index, True)
            
    def has_implicit_hydrogens(self):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def init_atom(self):
        """ iterate through molecule atoms and build the atoms table """
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def perceive_rings(self, keep_chorded_rings, keep_equivalent_rings):
        """ populate the rings and aromatic rings tableshe atoms table:
        self.rings_aromatics : list
            contains the list of ring_id items that are aromatic
        self.rings: dict
            store information about rings, like if they have corners that can
            be flipped and the graph of atoms that belong to them:

                self.rings[ring_id] = {
                                'corner_flip': False
                                'graph': {}
                                }

            The atom is built using the `walk_recursive` method
        self.ring_atom_to_ring_id: dict
            mapping of each atom belonginig to the ring: atom_idx -> ring_id
        """

        def isRingAromatic(ring_atom_indices):
            for atom_idx1, atom_idx2 in self.get_bonds_in_ring(ring_atom_indices):
                bond = self.mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
                if not bond.GetIsAromatic():
                    return False
            return True

        hjk_ring_detection = HJKRingDetection(self.graph) 
        rings = hjk_ring_detection.scan(keep_chorded_rings, keep_equivalent_rings) # list of tuples of atom indices
        for ring_atom_idxs in rings:
            if isRingAromatic(ring_atom_idxs):
                self.rings_aromatic.append(ring_atom_idxs)
            self.rings[ring_atom_idxs] = {'corner_flip':False}
            graph = {}
            for atom_idx in ring_atom_idxs:
                self.atom_to_ring_id[atom_idx].append(ring_atom_idxs)
                # graph of atoms affected by potential ring movements
                graph[atom_idx] = self.walk_recursive(atom_idx, collected=[], exclude=list(ring_atom_idxs))
            self.rings[ring_atom_idxs]['graph'] = graph

    def init_bond(self):
        """ iterate through molecule bonds and build the bond table (id, table)
            CALCULATE
                bond_order: int
                    0       : pseudo-bond (for pseudoatoms)
                    1,2,3   : single-triple bond
                    5       : aromatic
                    999     : rigid

                if bond is in ring (both start and end atom indices in the bond are in the same ring)

            SETUP OPERATION
                Setup.add_bond(idx1, idx2, order, in_rings=[])
        """
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def get_mol_name(self):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def find_pattern(self, smarts):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def get_smiles_and_order(self):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def show(self):
        tot_charge = 0

        print("Molecule setup\n")
        print("==============[ ATOMS ]===================================================")
        print("idx  |          coords            | charge |ign| atype    | connections")
        print("-----+----------------------------+--------+---+----------+--------------- . . . ")
        for k, v in list(self.coord.items()):
            print("% 4d | % 8.3f % 8.3f % 8.3f | % 1.3f | %d" % (k, v[0], v[1], v[2],
                  self.charge[k], self.atom_ignore[k]),
                  "| % -8s |" % self.atom_type[k],
                  self.graph[k])
            tot_charge += self.charge[k]
        print("-----+----------------------------+--------+---+----------+--------------- . . . ")
        print("  TOT CHARGE: %3.3f" % tot_charge)

        print("\n======[ DIRECTIONAL VECTORS ]==========")
        for k, v in list(self.coord.items()):
            if k in self.interaction_vector:
                print("% 4d " % k, self.atom_type[k], end=' ')

        print("\n==============[ BONDS ]================")
        # For sanity users, we won't show those keys for now
        keys_to_not_show = ['bond_order', 'type']
        for k, v in list(self.bond.items()):
            t = ', '.join('%s: %s' % (i, j) for i, j in v.items() if not i in keys_to_not_show)
            print("% 8s - " % str(k), t)

        # _macrocycle_typer.show_macrocycle_scores(self)

        print('')

class RDKitMoleculeSetup(MoleculeSetup):

    @classmethod
    def from_mol(cls, mol, keep_chorded_rings=False, keep_equivalent_rings=False,
                 assign_charges=True, conformer_id=-1):
        if mol.GetNumConformers() == 0: 
            raise ValueError("RDKit molecule does not have a conformer. Need 3D coordinates.")
        rdkit_conformer = mol.GetConformer(conformer_id) 
        if not rdkit_conformer.Is3D():
            warnings.warn("RDKit molecule not labeled as 3D. This warning won't show again.")
            RDKitMoleculeSetup.warned_not3D = True
        if mol.GetNumConformers() > 1 and conformer_id == -1:
            msg = "RDKit molecule has multiple conformers. Considering only the first one." 
            print(msg, file=sys.stderr)
        molsetup = cls()
        molsetup.mol = mol
        molsetup.atom_true_count = molsetup.get_num_mol_atoms()
        molsetup.name = molsetup.get_mol_name()
        coords = rdkit_conformer.GetPositions()
        molsetup.init_atom(assign_charges, coords)
        molsetup.perceive_rings(keep_chorded_rings, keep_equivalent_rings)
        molsetup.init_bond()
        return molsetup


    def get_smiles_and_order(self):
        """
            return the SMILES after Chem.RemoveHs()
            and the mapping between atom indices in smiles and self.mol
        """

        # 3D SDF files written by other toolkits (OEChem, ChemAxon)
        # seem to not include the chiral flag in the bonds block, only in
        # the atoms block. RDKit ignores the atoms chiral flag as per the
        # spec. When reading SDF (e.g. from PubChem/PDB),
        # we may need to have RDKit assign stereo from coordinates, see:
        # https://sourceforge.net/p/rdkit/mailman/message/34399371/
        mol_noH = Chem.RemoveHs(self.mol) # imines (=NH) may become chiral
        # stereo imines [H]/N=C keep [H] after RemoveHs()
        # H isotopes also kept after RemoveHs()
        atomic_num_mol_noH = [atom.GetAtomicNum() for atom in mol_noH.GetAtoms()]
        noH_to_H = []
        parents_of_hs = {}
        for (index, atom) in enumerate(self.mol.GetAtoms()):
            if atom.GetAtomicNum() == 1: continue
            for i in range(len(noH_to_H), len(atomic_num_mol_noH)):
                if atomic_num_mol_noH[i] > 1:
                    break
                h_atom = mol_noH.GetAtomWithIdx(len(noH_to_H))
                assert(h_atom.GetAtomicNum() == 1)
                neighbors = h_atom.GetNeighbors()
                assert(len(neighbors) == 1)
                parents_of_hs[len(noH_to_H)] = neighbors[0].GetIdx()
                noH_to_H.append('H')
            noH_to_H.append(index)
        extra_hydrogens = len(atomic_num_mol_noH) - len(noH_to_H)
        if extra_hydrogens > 0:
            assert(set(atomic_num_mol_noH[len(noH_to_H):]) == {1})
        for i in range(extra_hydrogens):
            h_atom = mol_noH.GetAtomWithIdx(len(noH_to_H))
            assert(h_atom.GetAtomicNum() == 1)
            neighbors = h_atom.GetNeighbors()
            assert(len(neighbors) == 1)
            parents_of_hs[len(noH_to_H)] = neighbors[0].GetIdx()
            noH_to_H.append('H')

        # noH_to_H has the same length as the number of atoms in mol_noH
        # and each value is:
        #    - the index of the corresponding atom in mol, if value is integer
        #    - an hydrogen, if value is "H"
        # now, we need to replace those "H" with integers
        # "H" occur with stereo imine (e.g. [H]/N=C) and heavy Hs (e.g. [2H])
        hs_by_parent = {}
        for hidx, pidx in parents_of_hs.items():
            hs_by_parent.setdefault(pidx, [])
            hs_by_parent[pidx].append(hidx)
        for pidx, hidxs in hs_by_parent.items():
            siblings_of_h = [atom for atom in self.mol.GetAtomWithIdx(noH_to_H[pidx]).GetNeighbors() if atom.GetAtomicNum() == 1]
            sortidx = [i for i, j in sorted(list(enumerate(siblings_of_h)), key=lambda x: x[1].GetIdx())]
            if len(hidxs) == len(siblings_of_h):  
                # This is the easy case, just map H to each other in the order they appear
                for i, hidx in enumerate(hidxs):
                    noH_to_H[hidx] = siblings_of_h[sortidx[i]].GetIdx()
            elif len(hidxs) < len(siblings_of_h):
                # check hydrogen isotopes
                sibling_isotopes = [siblings_of_h[sortidx[i]].GetIsotope() for i in range(len(siblings_of_h))]
                molnoH_isotopes = [mol_noH.GetAtomWithIdx(hidx) for hidx in hidxs]
                matches = []
                for i, sibling_isotope in enumerate(sibling_isotopes):
                    for hidx in hidxs[len(matches):]:
                        if mol_noH.GetAtomWithIdx(hidx).GetIsotope() == sibling_isotope:
                            matches.append(i)
                            break
                if len(matches) != len(hidxs):
                    raise RuntimeError("Number of matched isotopes %d differs from query Hs: %d" % (len(matches), len(hidxs)))
                for hidx, i in zip(hidxs, matches):
                    noH_to_H[hidx] = siblings_of_h[sortidx[i]].GetIdx()
            else:
                raise RuntimeError("nr of Hs in mol_noH bonded to an atom exceeds nr of Hs in self.mol")

        smiles = Chem.MolToSmiles(mol_noH)
        order_string = mol_noH.GetProp("_smilesAtomOutputOrder")
        order_string = order_string.replace(',]', ']') # remove trailing comma
        order = json.loads(order_string) # mol_noH to smiles
        order = list(np.argsort(order))
        order = {noH_to_H[i]: order[i]+1 for i in range(len(order))} # 1-index
        return smiles, order

    def find_pattern(self, smarts):
        p = Chem.MolFromSmarts(smarts)
        return self.mol.GetSubstructMatches(p)

    def get_mol_name(self):
        if self.mol.HasProp("_Name"):
            return self.mol.GetProp("_Name")
        else:
            return None

    def get_num_mol_atoms(self):
        return self.mol.GetNumAtoms()

    def get_equivalent_atoms(self):
       return list(Chem.CanonicalRankAtoms(self.mol, breakTies=False))

    def init_atom(self, assign_charges, coords):
        """ initialize the atom table information """
        # extract/generate charges
        if assign_charges:
            copy_mol = Chem.Mol(self.mol)
            for atom in copy_mol.GetAtoms():
                if atom.GetAtomicNum() == 34:
                    atom.SetAtomicNum(16)
            rdPartialCharges.ComputeGasteigerCharges(copy_mol)
            charges = [a.GetDoubleProp('_GasteigerCharge') for a in copy_mol.GetAtoms()]
        else:
            charges = [0.0] * self.mol.GetNumAtoms()
        # perceive chirality
        # TODO check consistency for chiral model between OB and RDKit
        chiral_info = {}
        for data in Chem.FindMolChiralCenters(self.mol, includeUnassigned=True):
            chiral_info[data[0]] = data[1]
        # register atom
        for a in self.mol.GetAtoms():
            idx = a.GetIdx()
            chiral = False
            if idx in chiral_info:
                chiral = chiral_info[idx]
            self.add_atom(idx,
                    coord=coords[idx],
                    element=a.GetAtomicNum(),
                    charge=charges[idx],
                    atom_type=None,
                    pdbinfo = getPdbInfoNoNull(a),
                    neighbors = [n.GetIdx() for n in a.GetNeighbors()],
                    chiral=False,
                    ignore=False)

    def init_bond(self):
        """ initialize bond information """
        for b in self.mol.GetBonds():
            idx1 = b.GetBeginAtomIdx()
            idx2 = b.GetEndAtomIdx()
            bond_order = int(b.GetBondType())
            # fix the RDKit aromatic type (FIXME)
            if bond_order == 12: # aromatic
                bond_order = 5
            if bond_order == 1:
                rotatable = True
            else:
                rotatable = False
            idx1_rings = set(self.get_atom_rings(idx1))
            idx2_rings = set(self.get_atom_rings(idx2))
            in_rings = list(set.intersection(idx1_rings, idx2_rings))
            self.add_bond(idx1, idx2, order=bond_order, rotatable=rotatable, in_rings=in_rings)

    def copy(self):
        """ return a copy of the current setup"""
        newsetup = RDKitMoleculeSetup()
        for key, value in self.__dict__.items():
            if key != "mol":
                newsetup.__dict__[key] = deepcopy(value)
        newsetup.mol = Chem.Mol(self.mol) # not sure how deep of a copy this is
        return newsetup

    def has_implicit_hydrogens(self):
        # based on needsHs from RDKit's AddHs.cpp
        for atom in self.mol.GetAtoms():
            nr_H_neighbors = 0
            for neighbor in atom.GetNeighbors():
                nr_H_neighbors += int(neighbor.GetAtomicNum() == 1)
            if atom.GetTotalNumHs(includeNeighbors=False) > nr_H_neighbors:
                return True
        return False

### AtomTyper ###
class AtomicGeometry():
    """generate reference frames and add extra sites"""

    PLANAR_TOL = 0.1 # angstroms, length of neighbour vecs for Z axis

    def __init__(self, parent, neigh, xneigh=[], x90=False):
        """arguments are indices of atoms"""

        # real atom hosting extra sites
        if type(parent) != int:
            raise RuntimeError('parent must be int')
        self.parent = parent

        # list of bonded atoms (used to define z-axis)
        self.neigh = []
        for i in neigh:
            if type(i) != int:
                raise RuntimeError('neigh indices must be int')
            self.neigh.append(i)

        # list of atoms that
        self.xneigh = []
        for i in xneigh:
            if type(i) != int:
                raise RuntimeError('xneigh indices must be int')
            self.xneigh.append(i)

        self.calc_x = len(self.xneigh) > 0
        self.x90 = x90 # y axis becomes x axis (useful to rotate in-plane by 90 deg)

    def calc_point(self, distance, theta, phi, coords):
        """return coordinates of point specified in spherical coordinates"""

        z = self._calc_z(coords)

        # return pt aligned with z-axis
        if phi == 0.:
            return z * distance + np.array(coords[self.parent])

        # need x-vec if phi != 0
        elif self.calc_x == False:
            raise RuntimeError('phi must be zero if X undefined')

        else:
            x = self._calc_x(coords)
            if self.x90:
                x = np.cross(self.z, x)
            y = np.cross(z, x)
            pt = z * distance
            pt = self._rot3D(pt, y, phi)
            pt = self._rot3D(pt, z, theta)
            pt += np.array(coords[self.parent])
            return pt

    def _calc_z(self, coords):
        """ maximize distance from neigh """
        z = np.zeros(3)
        cumsum = np.zeros(3)
        for i in self.neigh:
            v = np.array(coords[self.parent]) - np.array(coords[i])
            cumsum += v
            z += self.normalized(v)
        z = self.normalized(z)
        if np.sum(cumsum**2) < self.PLANAR_TOL**2:
            raise RuntimeError('Refusing to place Z axis on planar atom')
        return z

    def _calc_x(self, coords):
        x = np.zeros(3)
        for i in self.xneigh:
            v = np.array(coords[self.parent]) - np.array(coords[i])
            x += self.normalized(v)
        x = self.normalized(x)
        return x

    def _rot3D(self, pt, ax, rad):
        """
            Rotate point:
            pt = (x,y,z) coordinates to be rotated
            ax = vector around wich rotation is performed
            rad = rotate by "rad" radians
        """
        # If axis has len=0, rotate by 0.0 rad on any axis
        # Make sure ax has unitary length
        len_ax = (ax[0]**2 + ax[1]**2 + ax[2]**2)**0.5
        if len_ax == 0.:
            u, v, w = (1, 0, 0)
            rad = 0.0
        else:
            u, v, w = [i/len_ax for i in ax]
        x, y, z = pt
        ux, uy, uz = u*x, u*y, u*z
        vx, vy, vz = v*x, v*y, v*z
        wx, wy, wz = w*x, w*y, w*z
        sa=np.sin(rad)
        ca=np.cos(rad)
        p0 =(u*(ux+vy+wz)+(x*(v*v+w*w)-u*(vy+wz))*ca+(-wy+vz)*sa)
        p1=(v*(ux+vy+wz)+(y*(u*u+w*w)-v*(ux+wz))*ca+(wx-uz)*sa)
        p2=(w*(ux+vy+wz)+(z*(u*u+v*v)-w*(ux+vy))*ca+(-vx+uy)*sa)
        return (p0, p1, p2)


    def normalized(self, vec):
        l = sum([x**2 for x in vec])**0.5
        if type(vec) == list:
            return [x/l for x in vec]
        else:
            # should be np.array
            return vec / l

class AtomTyper:

    defaults_json = """{
        "ATOM_PARAMS": {
            "alkyl glue": [
                {"smarts": "[#1]",                  "atype": "H", "comment": "invisible"},
                {"smarts": "[#1][#7,#8,#9,#15,#16]","atype": "HD"},
                {"smarts": "[#5]",              "atype": "B"},
                {"smarts": "[C]",               "atype": "C"},
                {"smarts": "[c]",               "atype": "A"},
                {"smarts": "[#7]",              "atype": "NA"},
                {"smarts": "[#8]",              "atype": "OA"},
                {"smarts": "[#9]",              "atype": "F"},
                {"smarts": "[#12]",             "atype": "Mg"},
                {"smarts": "[#14]",             "atype": "Si"},
                {"smarts": "[#15]",             "atype": "P"},
                {"smarts": "[#16]",             "atype": "S"},
                {"smarts": "[#17]",             "atype": "Cl"},
                {"smarts": "[#20]",             "atype": "Ca"},
                {"smarts": "[#25]",             "atype": "Mn"},
                {"smarts": "[#26]",             "atype": "Fe"},
                {"smarts": "[#30]",             "atype": "Zn"},
                {"smarts": "[#35]",             "atype": "Br"},
                {"smarts": "[#53]",             "atype": "I"},
                {"smarts": "[#7X3v3][a]",       "atype": "N",  "comment": "pyrrole, aniline"},
                {"smarts": "[#7X3v3][#6X3v4]",  "atype": "N",  "comment": "amide"},
                {"smarts": "[#7+1]",            "atype": "N",  "comment": "ammonium, pyridinium"},
                {"smarts": "[SX2]",             "atype": "SA", "comment": "sulfur acceptor"}
            ]
        }
    }
    """
    def __init__(self, parameters={}, add_parameters=[]):
        self.parameters = json.loads(self.defaults_json)
        for key in parameters:
            self.parameters[key] = json.loads(json.dumps(parameters[key])) # a safe copy
        # add additional parameters
        if len(add_parameters) > 0:
            keys = list(self.parameters["ATOM_PARAMS"].keys())
            if len(keys) != 1:
                msg = "add_parameters is usable only when there is one group of parameters"
                msg += ", but there are %d groups: %s" % (len(keys), str(keys))
                raise RuntimeError(msg)
            key = keys[0]
            self.parameters['ATOM_PARAMS'][key].extend(add_parameters)

    def __call__(self, setup):
        self._type_atoms(setup)
        if 'OFFATOMS' in self.parameters:
            cached_offatoms = self._cache_offatoms(setup)
            coords = [x for x in setup.coord.values()]
            self._set_offatoms(setup, cached_offatoms, coords)
        return

    def _type_atoms(self, setup):
        parsmar = self.parameters['ATOM_PARAMS']
        # ensure every "atompar" is defined in a single "smartsgroup"
        ensure = {}
        # go over all "smartsgroup"s
        for smartsgroup in parsmar:
            if smartsgroup == 'comment': continue
            for line in parsmar[smartsgroup]: # line is a dict, e.g. {"smarts": "[#1][#7,#8,#9,#15,#16]","atype": "HD"}
                smarts = str(line['smarts'])
                if 'atype' not in line: continue
                # get indices of the atoms in the smarts to which the parameters will be assigned
                idxs = [0] # by default, the first atom in the smarts gets parameterized
                if 'IDX' in line:
                    idxs = [i - 1 for i in line['IDX']] # convert from 1- to 0-indexing
                # match SMARTS
                hits = setup.find_pattern(smarts)
                atompar = 'atype' # we care only about 'atype', for now, but may want to extend
                atom_type = line[atompar]
                # keep track of every "smartsgroup" that modified "atompar"
                ensure.setdefault(atompar, [])
                ensure[atompar].append(smartsgroup)
                # Each "hit" is a tuple of atom indeces that matched the smarts
                # The length of each "hit" is the number of atoms in the smarts
                for hit in hits:
                    # Multiple atoms may be targeted by a single smarts:
                    # For example: both oxygens in NO2 are parameterized by a single smarts pattern.
                    # "idxs" are 1-indeces of atoms in the smarts to which parameters are to be assigned.
                    for idx in idxs:
                        setup.set_atom_type(hit[idx], atom_type) # overrides previous calls
        # guarantee that each atompar is exclusive of a single group
        for atompar in ensure:
            if len(set(ensure[atompar])) > 1:
                msg = 'WARNING: %s is modified in multiple smartsgroups: %s' % (atompar, set(ensure[atompar]))
                print(msg)
        return


    def _cache_offatoms(self, setup):
        """ precalculate off-site atoms """
        parsmar = self.parameters['OFFATOMS']
        cached_offatoms = {}
        n_offatoms = 0
        # each parent atom can only be matched once in each smartsgroup
        for smartsgroup in parsmar:
            if smartsgroup == "comment": continue
            tmp = {}
            for line in parsmar[smartsgroup]:
                # SMARTS
                smarts = str(line['smarts'])
                hits = setup.find_pattern(smarts)
                # atom indexes in smarts string
                smarts_idxs = [0]
                if 'IDX' in line:
                    smarts_idxs = [i - 1 for i in line['IDX']]
                for smarts_idx in smarts_idxs:
                    for hit in hits:
                        parent_idx = hit[smarts_idx]
                        tmp.setdefault(parent_idx, []) # TODO tmp[parent_idx] = [], yeah?
                        for offatom in line['OFFATOMS']:
                            # set defaults
                            tmp[parent_idx].append(
                                {'offatom': {'distance': 1.0,
                                             'x90': False,
                                             'phi': 0.0,
                                             'theta': 0.0,
                                             'z': [],
                                             'x': []},
                                 'atom_params': {}
                                })
                            for key in offatom:
                                if key in ['distance', 'x90']:
                                    tmp[parent_idx][-1]['offatom'][key] = offatom[key]
                                # replace SMARTS indexes by the atomic index
                                elif key in ['z', 'x']:
                                    for i in offatom[key]:
                                        idx = hit[i - 1]
                                        tmp[parent_idx][-1]['offatom'][key].append(idx)
                                # convert degrees to radians
                                elif key in ['theta', 'phi']:
                                    tmp[parent_idx][-1]['offatom'][key] = np.radians(offatom[key])
                                # ignore comments
                                elif key in ['comment']:
                                    pass
                                elif key == 'atype':
                                    tmp[parent_idx][-1]['atom_params'][key] = offatom[key]
                                else:
                                    pass
            for parent_idx in tmp:
                for offatom_dict in tmp[parent_idx]:
                    #print '1-> ', self.atom_params['q'], len(self.coords)
                    atom_params = offatom_dict['atom_params']
                    offatom = offatom_dict['offatom']
                    atomgeom = AtomicGeometry(parent_idx,
                                              neigh=offatom['z'],
                                              xneigh=offatom['x'],
                                              x90=offatom['x90'])
                    args = (atom_params['atype'],
                            offatom['distance'],
                            offatom['theta'],
                            offatom['phi'])
                    # number of coordinates (before adding new offatom)
                    cached_offatoms[n_offatoms] = (atomgeom, args)
                    n_offatoms += 1
        return cached_offatoms

    def _set_offatoms(self, setup, cached_offatoms, coords):
        """add cached offatoms"""
        for k, (atomgeom, args) in cached_offatoms.items():
            (atom_type, dist, theta, phi) = args
            offatom_coords = atomgeom.calc_point(dist, theta, phi, coords)
            tmp = setup.get_pdbinfo(atomgeom.parent+1)
            pdbinfo = PDBAtomInfo('G', tmp.resName, tmp.resNum, tmp.chain)
            pseudo_atom = {
                    'coord': offatom_coords,
                    'anchor_list': [atomgeom.parent],
                    'charge': 0.0,
                    'pdbinfo': pdbinfo,
                    'atom_type': atom_type,
                    'bond_type': 0,
                    'rotatable': False
                    }
            setup.add_pseudo(**pseudo_atom)
        return

### BondTyperLegacy ###
class BondTyperLegacy:

    def __call__(self, setup, flexible_amides, rigidify_bonds_smarts, rigidify_bonds_indices, not_terminal_atoms=[]):
        """Typing atom bonds in the legacy way

        Args:
            setup: MoleculeSetup object

            rigidify_bond_smarts (list): patterns to freeze bonds, e.g. conjugated carbons
        """
        def _is_terminal(idx):
            """ check if the atom has more than one connection with non-ignored atoms"""
            if setup.get_element(idx) == 1:
                return True
            return len([x for x in setup.get_neigh(idx) if not setup.get_ignore(x)]) == 1
        amide_bonds = [(x[0], x[1]) for x in setup.find_pattern('[NX3]-[CX3]=[O,N]')] # includes amidines

        # tertiary amides with non-identical substituents will be allowed to rotate
        tertiary_amides = [x for x in setup.find_pattern('[NX3]([!#1])([!#1])-[CX3]=[O,N]')]
        equivalent_atoms = setup.get_equivalent_atoms()
        num_amides_removed = 0
        num_amides_originally = len(amide_bonds)
        for x in tertiary_amides:
            r1, r2 = x[1], x[2]
            if equivalent_atoms[r1] != equivalent_atoms[r2]:
                amide_bonds.remove((x[0], x[3]))
                num_amides_removed += 1
        assert(num_amides_originally == num_amides_removed + len(amide_bonds))

        to_rigidify = set()
        n_smarts = len(rigidify_bonds_smarts)
        assert(n_smarts == len(rigidify_bonds_indices))
        for i in range(n_smarts):
            a, b = rigidify_bonds_indices[i]
            smarts = rigidify_bonds_smarts[i]
            indices_list = setup.find_pattern(smarts)
            for indices in indices_list:
                atom_a = indices[a]
                atom_b = indices[b]
                to_rigidify.add((atom_a, atom_b))
                to_rigidify.add((atom_b, atom_a))

        for bond_id, bond_info in setup.bond.items():
            rotatable = True
            bond_order = setup.bond[bond_id]['bond_order']
            # bond requested to be rigid
            if bond_id in to_rigidify:
                bond_order = 1.1 # macrocycle class breaks bonds if bond_order == 1
                rotatable = False
            # non-rotatable bond
            if bond_info['bond_order'] > 1:
                rotatable = False
            # in-ring bond
            if len(bond_info['in_rings']):
                rotatable = False
            # it's a terminal atom (methyl, halogen, hydrogen...)
            is_terminal_1 = _is_terminal(bond_id[0]) and (bond_id[0] not in not_terminal_atoms)
            is_terminal_2 = _is_terminal(bond_id[1]) and (bond_id[1] not in not_terminal_atoms)
            if is_terminal_1 or is_terminal_2:
                rotatable = False
            # check if bond is amide
            # NOTE this should have been done during the setup, right?
            if (bond_id in amide_bonds or (bond_id[1], bond_id[0]) in amide_bonds) and not flexible_amides:
                rotatable = False
                bond_order = 99
            setup.bond[bond_id]['rotatable'] = rotatable
            setup.bond[bond_id]['bond_order'] = bond_order

### FlexMacrocycle ###
class FlexMacrocycle:
    def __init__(self, min_ring_size=7, max_ring_size=33, double_bond_penalty=50, max_breaks=4):
        """Initialize macrocycle typer.

        Args:
            min_ring_size (int): minimum size of the ring (default: 7)
            max_ring_size (int): maximum size of the ring (default: 33)
            double_bond_penalty (float)

        """
        self._min_ring_size = min_ring_size
        self._max_ring_size = max_ring_size
        # accept also double bonds (if nothing better is found)
        self._double_bond_penalty = double_bond_penalty
        self.max_breaks = max_breaks

        self.setup = None
        self.breakable_rings = None
        self._conj_bond_list = None

    def collect_rings(self, setup):
        """ get non-aromatic rings of desired size and
            list bonds that are part of unbreakable rings

            Bonds belonging to rigid cycles can't be deleted or
            made rotatable even if they are part of a breakable ring
        """
        breakable_rings = []
        rigid_rings = []
        for ring_id in list(setup.rings.keys()): # ring_id are the atom indices in each ring
            size = len(ring_id)
            if ring_id in setup.rings_aromatic:
                rigid_rings.append(ring_id)
            elif size < self._min_ring_size:
                rigid_rings.append(ring_id)
                # do not add rings > _max_ring_size to rigid_rings
                # because bonds in rigid rings will not be breakable 
                # and these bonds may also belong to breakable rings
            elif size <= self._max_ring_size:
                breakable_rings.append(ring_id)

        bonds_in_rigid_cycles = set()
        for ring_atom_indices in rigid_rings:
            for bond in setup.get_bonds_in_ring(ring_atom_indices):
                bonds_in_rigid_cycles.add(bond)

        return breakable_rings, bonds_in_rigid_cycles 

    def _detect_conj_bonds(self):
        """ detect bonds in conjugated systems
        """
        # TODO this should be removed once atom typing will be done
        conj_bond_list = []
        # pattern = "[R0]=[R0]-[R0]=[R0]" # Does not match conjugated bonds inside  the macrocycle?
        pattern = '*=*[*]=,#,:[*]' # from SMARTS_InteLigand.txt
        found = self.setup.find_pattern(pattern)
        for f in found:
            bond = (f[1], f[2])
            bond = (min(bond), max(bond))
            conj_bond_list.append(bond)
        return conj_bond_list

    def _score_bond(self, bond):
        """ provide a score for the likeness of the bond to be broken"""
        bond = self.setup.get_bond_id(bond[0], bond[1])
        atom_idx1, atom_idx2 = bond
        score = 100

        bond_order = self.setup.bond[bond]['bond_order']
        if bond_order not in [1, 2, 3]: # aromatic, double, made rigid explicitly (order=1.1 from --rigidify)
            return -1
        if not self.setup.get_element(atom_idx1) == 6 or self.setup.is_aromatic(atom_idx1):
            return -1
        if not self.setup.get_element(atom_idx2) == 6 or self.setup.is_aromatic(atom_idx2):
            return -1
        # triple bond tolerated but not preferred (TODO true?)
        if bond_order == 3:
            score -= 30
        elif (bond_order == 2):
            score -= self._double_bond_penalty
        if bond in self._conj_bond_list:
            score -= 30
        # discourage chiral atoms
        if self.setup.get_chiral(atom_idx1) or self.setup.get_chiral(atom_idx2):
            score -= 20
        return score

    def get_breakable_bonds(self, bonds_in_rigid_rings):
        """ find breaking points for rings
            following guidelines defined in [1]
            The optimal bond has the following properties:
            - does not involve a chiral atom
            - is not double/triple (?)
            - is between two carbons
            (preferably? we can now generate pseudoAtoms on the fly!)
            - is a bond present only in one ring

             [1] Forli, Botta, J. Chem. Inf. Model., 2007, 47 (4)
              DOI: 10.1021/ci700036j
        """
        breakable = {}
        for ring_atom_indices in self.breakable_rings:
            for bond in self.setup.get_bonds_in_ring(ring_atom_indices):
                score = self._score_bond(bond)
                if score > 0 and bond not in bonds_in_rigid_rings:
                    breakable[bond] = {'score': score}
        return breakable

    def search_macrocycle(self, setup, delete_these_bonds=[]):
        """Search for macrocycle in the molecule

        Args:
            setup : MoleculeSetup object

        """
        self.setup = setup

        self.breakable_rings, bonds_in_rigid_rings = self.collect_rings(setup)
        self._conj_bond_list = self._detect_conj_bonds()
        if len(delete_these_bonds) == 0:
            breakable_bonds = self.get_breakable_bonds(bonds_in_rigid_rings)
        else:
            breakable_bonds = {}
            for bond in delete_these_bonds:
                bond = self.setup.get_bond_id(bond[0], bond[1])
                breakable_bonds[bond] = {"score": self._score_bond(bond)}
        break_combo_data = self.combinatorial_break_search(breakable_bonds)
        return break_combo_data, bonds_in_rigid_rings

    def combinatorial_break_search(self, breakable_bonds):
        """ enumerate all combinations of broken bonds
            once a bond is broken, it will break one or more rings
            subsequent bonds will be pulled from intact (unbroken) rings
            
            the number of broken bonds may be variable
            returns only combinations of broken bonds that break the max number of broken bonds
        """

        max_breaks = self.max_breaks
        break_combos = self._recursive_break(self.breakable_rings, max_breaks, breakable_bonds, set(), [])
        break_combos = list(break_combos) # convert from set
        max_broken_bonds = 0
        output_break_combos = [] # found new max, discard prior data
        output_bond_scores = []
        output_broken_rings = []
        for broken_bonds in break_combos:
            n_broken_bonds = len(broken_bonds)
            bond_score = sum([breakable_bonds[bond]['score'] for bond in broken_bonds])
            broken_rings = self.get_broken_rings(self.breakable_rings, broken_bonds)
            if n_broken_bonds > max_broken_bonds:
                max_broken_bonds = n_broken_bonds
                output_break_combos = [] # found new max, discard prior data
                output_bond_scores = []
                output_broken_rings = []
            if n_broken_bonds == max_broken_bonds:
                output_break_combos.append(broken_bonds)
                output_bond_scores.append(bond_score)
                output_broken_rings.append(broken_rings)
        break_combo_data = {"bond_break_combos": output_break_combos,
                            "bond_break_scores": output_bond_scores,
                            "broken_rings": output_broken_rings}
        return break_combo_data


    def _recursive_break(self, rings, max_breaks, breakable_bonds, output=set(), broken_bonds=[]):
        if max_breaks == 0:
            return output
        unbroken_rings = self.get_unbroken_rings(rings, broken_bonds)
        atoms_in_broken_bonds = atoms_in_broken_bonds = [atom_idx for bond in broken_bonds for atom_idx in bond]
        for bond in breakable_bonds:
            if bond[0] in atoms_in_broken_bonds or bond[1] in atoms_in_broken_bonds:
                continue # each atom can be in only one broken bond
            is_bond_in_ring = False
            for ring in unbroken_rings:
                if bond in self.setup.get_bonds_in_ring(ring):
                    is_bond_in_ring = True
                    break
            if is_bond_in_ring:
                current_broken_bonds = [(a, b) for (a, b) in broken_bonds + [bond]]
                num_unbroken_rings = len(self.get_unbroken_rings(rings, current_broken_bonds))
                data_row = tuple(sorted([(a, b) for (a, b) in current_broken_bonds]))
                output.add(data_row)
                if num_unbroken_rings > 0:
                    output = self._recursive_break(rings, max_breaks-1, breakable_bonds,
                                                   output, current_broken_bonds)
        return output

    
    def get_unbroken_rings(self, rings, broken_bonds):
        unbroken = []
        for ring in rings:
            is_unbroken = True
            for bond in broken_bonds:
                if bond in self.setup.get_bonds_in_ring(ring): # consider precalculating bonds
                    is_unbroken = False
                    break # pun intended
            if is_unbroken:
                unbroken.append(ring)        
        return unbroken

    def get_broken_rings(self, rings, broken_bonds):
        broken_rings = []
        for ring in rings:
            is_broken = False
            for bond in broken_bonds:
                if bond in self.setup.get_bonds_in_ring(ring): # consider precalculating bonds
                    is_broken = True
                    break # pun intended
            if is_broken:
                broken_rings.append(ring)        
        return broken_rings


    def show_macrocycle_scores(self, setup):
        print("Warning: not showing macrocycle scores, check implementation.")
        return
        if setup is not None:
            print("\n==============[ MACROCYCLE SCORES ]================")
            bond_by_ring = defaultdict(list)

            for bond_id, data in list(setup.ring_bond_breakable.items()):
                ring_id = data['ring_id']
                bond_by_ring[ring_id].append(bond_id)

            for ring_id, bonds in list(bond_by_ring.items()):
                data = []
                print("-----------[ ring id: %s | size: %2d ]-----------" % (",".join([str(x) for x in ring_id]), len(ring_id)))

                for b in bonds:
                    score = setup.ring_bond_breakable[b]['score']
                    data.append((b, score))

                data = sorted(data, key=itemgetter(1), reverse=True)

                for b_count, b in enumerate(data):
                    begin = b[0][0]
                    end = b[0][1]
                    # bond = self.mol.setup.get_bond(b[0][0], b[0][1])
                    # begin = bond.GetBeginAtomIdx()
                    # end = bond.GetEndAtomIdx()
                    info = (b_count, begin, end, b[1], "#" * int(b[1] / 5), "-" * int(20 - b[1] / 5))
                    print("[ %2d] Bond [%3d --%3d] s:%3d [%s%s]" % info)

### FlexibilityBuilder ###
class FlexibilityBuilder:

    def __call__(self, setup, freeze_bonds=None, root_atom_index=None, break_combo_data=None, bonds_in_rigid_rings=None, glue_pseudo_atoms=None):
        """ """
        self.setup = setup
        self.flexibility_models = {}
        self._frozen_bonds = []
        if not freeze_bonds is None:
            self._frozen_bonds = freeze_bonds[:]
        # build graph for standard molecule (no open macrocycle rings)
        model = self.build_rigid_body_connectivity()
        model = self.set_graph_root(model, root_atom_index) # finds root if root_atom_index==None
        self.add_flex_model(model, score=False)

        # evaluate possible graphs for various ring openings
        if break_combo_data is not None:
            bond_break_combos = break_combo_data['bond_break_combos']
            bond_break_scores = break_combo_data['bond_break_scores']
            broken_rings_list = break_combo_data['broken_rings']
            for index in range(len(bond_break_combos)):
                bond_break_combo = bond_break_combos[index]
                bond_break_score = bond_break_scores[index]
                broken_rings =     broken_rings_list[index]
                model = self.build_rigid_body_connectivity(bond_break_combo, broken_rings, bonds_in_rigid_rings, glue_pseudo_atoms)
                self.set_graph_root(model, root_atom_index) # finds root if root_atom_index==None
                self.add_flex_model(model, score=True, initial_score=bond_break_score)

        self.select_best_model()

        # clean up
        del self._frozen_bonds
        del self.flexibility_models
        return self.setup


    def select_best_model(self):
        """
        select flexibility model with best complexity score
        """
        if len(self.flexibility_models) == 1: # no macrocyle open rings
            best_model = list(self.flexibility_models.values())[0]
        else:
            score_sorted_models = []
            for m_id, model in list(self.flexibility_models.items()):
                score_sorted_models.append((m_id, model['score']))
            #print("SORTED", score_sorted_models)
            score_sorted = sorted(score_sorted_models, key=itemgetter(1), reverse=True)
            #for model_id, score in score_sorted:
            #    print("ModelId[% 3d] score: %2.2f" % (model_id, score))
            # the 0-model is the rigid model used as reference
            best_model_id, best_model_score = score_sorted[1]
            best_model = self.flexibility_models[best_model_id]
        
        setup = best_model['setup']
        del best_model['setup']
        self.setup = setup
        best_model['torsions_org'] = self.flexibility_models[0]['torsions']

        self.setup.flexibility_model = best_model

    def add_flex_model(self, model, score=False, initial_score=0):
        """ add a flexible model to the list of configurations,
            and optionally score it, basing on the connectivity properties
        """

        model_id = len(self.flexibility_models)
        if score == False:
            model['score'] = float('inf')
        else:
            penalty = self.score_flex_model(model)
            model['score'] = initial_score + penalty
        self.flexibility_models[model_id] = model

    def build_rigid_body_connectivity(self, bonds_to_break=None, broken_rings=None, bonds_in_rigid_rings=None, glue_pseudo_atoms=None):
        """
        rigid_body_graph is the graph of rigid bodies
        ( rigid_body_id->[rigid_body_id,...] )

        rigid_body_members contains the atom indices in each rigid_body_id,
        ( rigid_body_id->[atom,...] )

        rigid_body_connectivity contains connectivity information between
        rigid bodies, mapping a two rigid bodies to the two atoms that connect
        them
        ( (rigid_body1, rigid_body2) -> (atom1,atom2)
        """
        # make a copy of the current mol graph, updated with the broken bond
        if bonds_to_break is None:
            self._current_setup = self.setup
        else:
            self._current_setup = self.copy_setup(bonds_to_break, broken_rings, bonds_in_rigid_rings)
            self.update_closure_atoms(bonds_to_break, glue_pseudo_atoms)

        # walk the mol graph to build the rigid body maps
        self._visited = defaultdict(lambda:False)
        self._rigid_body_members = {}
        self._rigid_body_connectivity = {}
        self._rigid_body_graph = defaultdict(list)
        self._rigid_index_by_atom = {}
        # START VALUE HERE SHOULD BE MADE MODIFIABLE FOR FLEX CHAIN
        self._rigid_body_count = 0
        self.walk_rigid_body_graph(start=0)
        # if only a rigid body is found
        if len(self._rigid_body_members) == 1:
            self._rigid_body_connectivity[0] = [0]
            self._rigid_body_graph[0] = [0]
        model = {'rigid_body_graph' : deepcopy(self._rigid_body_graph),
                'rigid_body_connectivity' : deepcopy(self._rigid_body_connectivity),
                'rigid_body_members' : deepcopy(self._rigid_body_members),
                'setup' : self._current_setup}
        return model

    def copy_setup(self, bond_list, broken_rings, bonds_in_rigid_rings):
        """ copy connectivity information (graph and bonds) from the setup,
            optionally delete bond_id listed in bonds_to_break,
            updating connectivty information
        """
        setup = self.setup.copy()
        for bond in bond_list:
            setup.del_bond(*bond)

        for ring in broken_rings:
            for bond in setup.get_bonds_in_ring(ring):
                if bond not in bonds_in_rigid_rings: # e.g. bonds in small rings do not rotata
                    if bond in bond_list:
                        continue # this bond has been deleted
                    bond_item = setup.get_bond(*bond)
                    if bond_item['bond_order'] == 1:
                        setup.bond[bond]['rotatable'] = True
        return setup


    def calc_max_depth(self, graph, seed_node, visited=[], depth=0):
        maxdepth = depth 
        visited.append(seed_node)
        for node in graph[seed_node]:
            if node not in visited:
                visited.append(node)
                newdepth = self.calc_max_depth(graph, node, visited, depth + 1)
                maxdepth = max(maxdepth, newdepth)
        return maxdepth


    def set_graph_root(self, model, root_atom_index=None):
        """ TODO this has to be made aware of the weight of the groups left
         (see 1jff:TA1)
        """

        if root_atom_index is None: # find rigid group that minimizes max_depth
            graph = deepcopy(model['rigid_body_graph'])
            while len(graph) > 2: # remove leafs until 1 or 2 rigid groups remain
                leaves = []
                for vertex, edges in list(graph.items()):
                    if len(edges) == 1:
                        leaves.append(vertex)
                for l in leaves:
                    for vertex, edges in list(graph.items()):
                        if l in edges:
                            edges.remove(l)
                            graph[vertex] = edges
                    del graph[l]

            if len(graph) == 1:
                root_body_index = list(graph.keys())[0]
            else:
                r1, r2 = list(graph.keys())
                r1_size = len(model['rigid_body_members'][r1])
                r2_size = len(model['rigid_body_members'][r2])
                if r1_size >= r2_size:
                    root_body_index = r1
                else:
                    root_body_index = r2

        else: # find index of rigid group
            for body_index in model['rigid_body_members']:
                if root_atom_index in model['rigid_body_members'][body_index]: # 1-index atoms
                    root_body_index = body_index

        model['root'] = root_body_index
        model['torsions'] = len(model['rigid_body_members']) - 1
        model['graph_depth'] = self.calc_max_depth(model['rigid_body_graph'], root_body_index, visited=[], depth=0)

        return model


    def score_flex_model(self, model):
        """ score a flexibility model basing on the graph properties"""
        base = self.flexibility_models[0]['graph_depth']
        score = 10 * (base-model['graph_depth'])
        return score


    def _generate_closure_pseudo(self, setup, bond_id, coords_dict={}):
        """ calculate position and parameters of the pseudoatoms for the closure"""
        closure_pseudo = []
        for idx in (0,1):
            target = bond_id[1 - idx]
            anchor = bond_id[0 - idx]
            if coords_dict is None or len(coords_dict) == 0:
                coord = setup.get_coord(target)
            else:
                coord = coords_dict[anchor]
            anchor_info = setup.pdbinfo[anchor]
            pdbinfo = PDBAtomInfo('G', anchor_info.resName, anchor_info.resNum, anchor_info.chain)
            closure_pseudo.append({
                'coord': coord,
                'anchor_list': [anchor],
                'charge': 0.0,
                'pdbinfo': pdbinfo,
                'atom_type': 'G',
                'bond_type': 0,
                'rotatable': False})
        return closure_pseudo


    def update_closure_atoms(self, bonds_to_break, coords_dict):
        """ create pseudoatoms required by the flexible model with broken bonds"""

        setup = self._current_setup
        for i, bond in enumerate(bonds_to_break):
            setup.ring_closure_info["bonds_removed"].append(bond) # bond is pair of atom indices
            pseudos = self._generate_closure_pseudo(setup, bond, coords_dict)
            for pseudo in pseudos:
                pseudo['atom_type'] = "%s%d" % (pseudo['atom_type'], i)
                pseudo_index = setup.add_pseudo(**pseudo)
                atom_index = pseudo['anchor_list'][0]
                if atom_index in setup.ring_closure_info:
                    raise RuntimeError("did not expect more than one G per atom")
                setup.ring_closure_info["pseudos_by_atom"][atom_index] = pseudo_index
            setup.set_atom_type(bond[0], "CG%d" % i)
            setup.set_atom_type(bond[1], "CG%d" % i)


    def walk_rigid_body_graph(self, start):
        """ recursive walk to build the graph of rigid bodies"""
        idx = 0
        rigid = [start]
        self._visited[start] = True
        current_rigid_body_count = self._rigid_body_count
        self._rigid_index_by_atom[start] = current_rigid_body_count
        sprouts_buffer = []
        while idx < len(rigid):
            current = rigid[idx]
            for neigh in self._current_setup.get_neigh(current):
                bond_id = self._current_setup.get_bond_id(current, neigh)
                bond_info = self._current_setup.get_bond(current, neigh)
                if self._visited[neigh]:
                    is_rigid_bond = (bond_info['rotatable'] == False) or (bond_id in self._frozen_bonds)
                    neigh_in_other_rigid_body = current_rigid_body_count != self._rigid_index_by_atom[neigh]
                    if is_rigid_bond and neigh_in_other_rigid_body:
                        raise RuntimeError('Flexible bonds within rigid group. We have a problem.')
                    continue
                if bond_info['rotatable'] and (bond_id not in self._frozen_bonds):
                    sprouts_buffer.append((current, neigh))
                else:
                    rigid.append(neigh)
                    self._rigid_index_by_atom[neigh] = current_rigid_body_count
                    self._visited[neigh] = True
            idx += 1
        self._rigid_body_members[current_rigid_body_count] = rigid
        for current, neigh in sprouts_buffer:
            if self._visited[neigh]: continue
            self._rigid_body_count+=1
            self._rigid_body_connectivity[current_rigid_body_count, self._rigid_body_count] = current, neigh
            self._rigid_body_connectivity[self._rigid_body_count, current_rigid_body_count] = neigh, current
            self._rigid_body_graph[current_rigid_body_count].append(self._rigid_body_count)
            self._rigid_body_graph[self._rigid_body_count].append(current_rigid_body_count)
            self.walk_rigid_body_graph(neigh)

### PDBQTWriterLegacy ###
PDBResInfo  = namedtuple('PDBResInfo',       "resName resNum chain")
class PDBQTWriterLegacy():

    @staticmethod
    def _get_pdbinfo_fitting_pdb_chars(pdbinfo):
        """ return strings and integers that are guaranteed
            to fit within the designated chars of the PDB format """

        atom_name = pdbinfo.name
        res_name = pdbinfo.resName
        res_num = pdbinfo.resNum
        chain = pdbinfo.chain
        if len(atom_name) > 4: atom_name = atom_name[0:4]
        if len(res_name) > 3: res_name = res_name[0:3]
        if res_num > 9999: res_num = res_num % 10000
        if len(chain) > 1: chain = chain[0:1]
        return atom_name, res_name, res_num, chain

    @classmethod
    def _make_pdbqt_line(cls, setup, atom_idx, resinfo_set, count):
        """ """
        record_type = "ATOM"
        alt_id = " "
        pdbinfo = setup.pdbinfo[atom_idx]
        if pdbinfo is None:
            pdbinfo = PDBAtomInfo('', '', 0, '')
        resinfo = PDBResInfo(pdbinfo.resName, pdbinfo.resNum, pdbinfo.chain)
        resinfo_set.add(resinfo)
        atom_name, res_name, res_num, chain = cls._get_pdbinfo_fitting_pdb_chars(pdbinfo)
        in_code = ""
        occupancy = 1.0
        temp_factor = 0.0
        coord = setup.coord[atom_idx]
        atom_type = setup.get_atom_type(atom_idx)
        charge = setup.charge[atom_idx]
        atom = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}"

        pdbqt_line = atom.format(record_type, count, pdbinfo.name, alt_id, res_name, chain,
                           res_num, in_code, float(coord[0]), float(coord[1]), float(coord[2]),
                           occupancy, temp_factor, charge, atom_type)
        return pdbqt_line, resinfo_set

    @classmethod
    def _walk_graph_recursive(cls, setup, node, data, edge_start=0, first=False):
        """ recursive walk of rigid bodies"""
        
        if first:
            data["pdbqt_buffer"].append('ROOT')
            member_pool = sorted(setup.flexibility_model['rigid_body_members'][node])
        else:
            member_pool = setup.flexibility_model['rigid_body_members'][node][:]
            member_pool.remove(edge_start)
            member_pool = [edge_start] + member_pool

        for member in member_pool:
            if setup.atom_ignore[member] == 1:
                continue
            pdbqt_line, resinfo_set = cls._make_pdbqt_line(setup, member, data["resinfo_set"], data["count"])
            data["resinfo_set"] = resinfo_set # written as if _make_pdbqt_line() doesn't modify its args (for readability)
            data["pdbqt_buffer"].append(pdbqt_line)
            data["numbering"][member] = data["count"] # count starts at 1
            data["count"] += 1

        if first:
            data["pdbqt_buffer"].append('ENDROOT')

        data["visited"].append(node)

        for neigh in setup.flexibility_model['rigid_body_graph'][node]:
            if neigh in data["visited"]:
                continue

            # Write the branch
            begin, next_index = setup.flexibility_model['rigid_body_connectivity'][node, neigh]

            # do not write branch (or anything downstream) if any of the two atoms
            # defining the rotatable bond are ignored
            if setup.atom_ignore[begin] or setup.atom_ignore[next_index]:
                continue

            begin = data["numbering"][begin]
            end = data["count"]

            data["pdbqt_buffer"].append("BRANCH %3d %3d" % (begin, end))
            data = cls._walk_graph_recursive(setup, neigh, data, edge_start=next_index)
            data["pdbqt_buffer"].append("ENDBRANCH %3d %3d" % (begin, end))
        
        return data

    @classmethod
    def write_string(cls, setup, add_index_map=False, remove_smiles=False, bad_charge_ok=False):
        """Output a PDBQT file as a string.

        Args:
            setup: MoleculeSetup

        Returns:
            str:  PDBQT string of the molecule
            bool: success
            str:  error message
        """

        success = True
        error_msg = ""
        
        if setup.has_implicit_hydrogens():
            error_msg += "molecule has implicit hydrogens (name=%s)\n" % setup.get_mol_name()
            success = False

        for idx, atom_type in setup.atom_type.items():
            if setup.atom_ignore[idx]:
                continue
            if atom_type is None:
                error_msg += 'atom number %d has None type, mol name: %s\n' % (idx, setup.get_mol_name())
                success = False
            c = setup.charge[idx]
            if not bad_charge_ok and (type(c) != float and type(c) != int or math.isnan(c) or math.isinf(c)):
                error_msg += 'atom number %d has non finite charge, mol name: %s, charge: %s\n' % (idx, setup.get_mol_name(), str(c))
                success = False

        if not success:
            pdbqt_string = ""
            return pdbqt_string, success, error_msg

        data = {
            "visited": [],
            "numbering": {},
            "pdbqt_buffer": [],
            "count": 1,
            "resinfo_set": set(),
        }
        atom_counter = {}

        torsdof = len(setup.flexibility_model['rigid_body_graph']) - 1

        if 'torsions_org' in setup.flexibility_model:
            torsdof_org = setup.flexibility_model['torsions_org']
            data["pdbqt_buffer"].append('REMARK Flexibility Score: %2.2f' % setup.flexibility_model['score'] )
            active_tors = torsdof_org
        else:
            active_tors = torsdof

        data = cls._walk_graph_recursive(setup, setup.flexibility_model["root"], data, first=True)

        if add_index_map:
            for i, remark_line in enumerate(cls.remark_index_map(setup, data["numbering"])):
                # Need to use 'insert' because data["numbering"]
                # is populated in self._walk_graph_recursive.
                data["pdbqt_buffer"].insert(i, remark_line)

        if not remove_smiles:
            smiles, order = setup.get_smiles_and_order()
            missing_h = [] # hydrogens which are not in the smiles
            strings_h_parent = []
            for key in data["numbering"]:
                if key in setup.atom_pseudo: continue
                if key not in order:
                    if setup.get_element(key) != 1:
                        error_msg += "non-Hydrogen atom unexpectedely missing from smiles!?"
                        error_msg += " (mol name: %s)\n" % setup.get_mol_name()
                        pdbqt_string = ""
                        success = False
                        return pdbqt_string, success, error_msg
                    missing_h.append(key)
                    parents = setup.get_neigh(key)
                    parents = [i for i in parents if i < setup.atom_true_count] # exclude pseudos
                    if len(parents) != 1:
                        error_msg += "expected hydrogen to be bonded to exactly one atom"
                        error_msg += " (mol name: %s)\n" % setup.get_mol_name()
                        pdbqt_string = ""
                        success = False
                        return pdbqt_string, success, error_msg
                    parent_idx = order[parents[0]] # already 1-indexed
                    string = ' %d %d' % (parent_idx, data["numbering"][key]) # key 0-indexed; _numbering[key] 1-indexed
                    strings_h_parent.append(string)
            remarks_h_parent = cls.break_long_remark_lines(strings_h_parent, "REMARK H PARENT")
            remark_prefix = "REMARK SMILES IDX"
            remark_idxmap = cls.remark_index_map(setup, data["numbering"], order, remark_prefix, missing_h)
            remarks = []
            remarks.append("REMARK SMILES %s" % smiles) # break line at 79 chars?
            remarks.extend(remark_idxmap)
            remarks.extend(remarks_h_parent)

            for i, remark_line in enumerate(remarks):
                # Need to use 'insert' because data["numbering"]
                # is populated in self._walk_graph_recursive.
                data["pdbqt_buffer"].insert(i, remark_line)

        if False: #self.setup.is_protein_sidechain:
            if len(data["resinfo_set"]) > 1:
                print("Warning: more than a single resName, resNum, chain in flexres", file=sys.stderr)
                print(data["resinfo_set"], file=sys.stderr)
            resinfo = list(data["resinfo_set"])[0]
            pdbinfo = pdbutils.PDBAtomInfo('', resinfo.resName, resinfo.resNum, resinfo.chain)
            _, res_name, res_num, chain = cls._get_pdbinfo_fitting_pdb_chars(pdbinfo)
            resinfo_string = "{:3s} {:1s}{:4d}".format(res_name, chain, res_num)
            data["pdbqt_buffer"].insert(0, 'BEGIN_RES %s' % resinfo_string)
            data["pdbqt_buffer"].append('END_RES %s' % resinfo_string)
        else: # no TORSDOF in flexres
            # torsdof is always going to be the one of the rigid, non-macrocyclic one
            data["pdbqt_buffer"].append('TORSDOF %d' % active_tors)

        pdbqt_string =  '\n'.join(data["pdbqt_buffer"]) + '\n'
        return pdbqt_string, success, error_msg

    @classmethod
    def remark_index_map(cls, setup, numbering, order=None, prefix="REMARK INDEX MAP", missing_h=()):
        """ write mapping of atom indices from input molecule to output PDBQT
            order[ob_index(i.e. 'key')] = smiles_index
        """

        if order is None: order = {key: key+1 for key in numbering} # key+1 breaks OB
        #max_line_length = 79
        #remark_lines = []
        #line = prefix
        strings = []
        for key in numbering:
            if key in setup.atom_pseudo: continue
            if key in missing_h: continue
            string = " %d %d" % (order[key], numbering[key])
            strings.append(string)
        return cls.break_long_remark_lines(strings, prefix)
        #    candidate_text = " %d %d" % (order[key], self._numbering[key])
        #    if (len(line) + len(candidate_text)) < max_line_length:
        #        line += candidate_text
        #    else:
        #        remark_lines.append(line)
        #        line = 'REMARK INDEX MAP' + candidate_text
        #remark_lines.append(line)
        #return remark_lines

    @staticmethod
    def break_long_remark_lines(strings, prefix, max_line_length=79):
        remarks = [prefix]
        for string in strings:
            if (len(remarks[-1]) + len(string)) < max_line_length:
                remarks[-1] += string
            else:
                remarks.append(prefix + string)
        return remarks

    @staticmethod
    def adapt_pdbqt_for_autodock4_flexres(pdbqt_string, res, chain, num):
        """ adapt pdbqt_string to be compatible with AutoDock4 requirements:
             - first and second atoms named CA and CB
             - write BEGIN_RES / END_RES
             - remove TORSDOF
            this is for covalent docking (tethered)
        """
        new_string = "BEGIN_RES %s %s %s\n" % (res, chain, num)
        atom_number = 0
        for line in pdbqt_string.split("\n"):
            if line == "":
                continue
            if line.startswith("TORSDOF"):
                continue
            if line.startswith("ATOM"):
                atom_number+=1
                if atom_number == 1:
                    line = line[:13] + 'CA' + line[15:]
                elif atom_number == 2:
                    line = line[:13] + 'CB' + line[15:]
                new_string += line + '\n'
                continue
            new_string += line + '\n'
        new_string += "END_RES %s %s %s\n" % (res, chain, num)
        return new_string

### ReactiveAtomTyper ###
class ReactiveAtomTyper:

    def __init__(self):

        self.ff = {
            "HD": {"rii": 2.00, "epsii": 0.020},
            "C":  {"rii": 4.00, "epsii": 0.150},
            "A":  {"rii": 4.00, "epsii": 0.150},
            "N":  {"rii": 3.50, "epsii": 0.160},
            "NA": {"rii": 3.50, "epsii": 0.160},
            "OA": {"rii": 3.20, "epsii": 0.200},
            "OS": {"rii": 3.20, "epsii": 0.200},
            "F":  {"rii": 3.09, "epsii": 0.080},
            "P":  {"rii": 4.20, "epsii": 0.200},
            "SA": {"rii": 4.00, "epsii": 0.200},
            "S":  {"rii": 4.00, "epsii": 0.200},
            "Cl": {"rii": 4.09, "epsii": 0.276},
            "CL": {"rii": 4.09, "epsii": 0.276},
            "Br": {"rii": 4.33, "epsii": 0.389},
            "BR": {"rii": 4.33, "epsii": 0.389},
            "I":  {"rii": 4.72, "epsii": 0.550},
            "Si": {"rii": 4.10, "epsii": 0.200},
            "B":  {"rii": 3.84, "epsii": 0.155},
            "W":  {"rii": 0.00, "epsii": 0.000},
        }
        std_atypes = list(self.ff.keys())
        rt, r2s, r2o = self.enumerate_reactive_types(std_atypes)
        self.reactive_type = rt
        self.reactive_to_std_atype_mapping = r2s
        self.reactive_to_order = r2o

    @staticmethod
    def enumerate_reactive_types(atypes):
        reactive_type = {1:{}, 2:{}, 3:{}}
        reactive_to_std_atype_mapping = {}
        reactive_to_order = {}
        for order in (1,2,3):
            for atype in atypes:
                if len(atype) == 1:
                    new_atype = "%s%d" % (atype, order)
                else:
                    new_atype = "%s%d" % (atype[0], order+3)
                    if new_atype in reactive_to_std_atype_mapping:
                        new_atype = "%s%d" % (atype[0], order+6)
                        if new_atype in reactive_to_std_atype_mapping:
                            raise RuntimeError("ran out of numbers for reactive types :(")
                reactive_to_std_atype_mapping[new_atype] = atype
                reactive_to_order[new_atype] = order
                reactive_type[order][atype] = new_atype
                ### # avoid atom type clashes with multiple reactive residues by
                ### # prefixing with the index of the residue, e.g. C3 -> 1C3.
                ### for i in range(8): # hopefully 8 reactive residues is sufficient
                ###     prefixed_new_atype = '%d%s' % ((i+1), new_atype)
                ###     reactive_to_std_atype_mapping[prefixed_new_atype] = atype
        return reactive_type, reactive_to_std_atype_mapping, reactive_to_order


    def get_scaled_parm(self, atype1, atype2):
        """ generate scaled parameters for a pairwise interaction between two atoms involved in a
            reactive interaction

            Rij and epsij are calculated according to the AD force field:
                # - To obtain the Rij value for non H-bonding atoms, calculate the
                #        arithmetic mean of the Rii values for the two atom types.
                #        Rij = (Rii + Rjj) / 2
                #
                # - To obtain the epsij value for non H-bonding atoms, calculate the
                #        geometric mean of the epsii values for the two atom types.
                #        epsij = sqrt( epsii * epsjj )
        """

        atype1_org, _ = self.get_basetype_and_order(atype1)
        atype2_org, _ = self.get_basetype_and_order(atype2)
        atype1_rii = self.ff[atype1_org]['rii']
        atype1_epsii = self.ff[atype1_org]['epsii']
        atype2_rii = self.ff[atype2_org]['rii']
        atype2_epsii = self.ff[atype2_org]['epsii']
        atype1_rii = atype1_rii
        atype2_rii = atype2_rii
        rij = (atype1_rii + atype2_rii) / 2
        epsij = math.sqrt(atype1_epsii * atype2_epsii)
        return rij, epsij


    def get_reactive_atype(self, atype, reactive_order):
        """ create or retrive new reactive atom type label name"""
        macrocycle_glue_types = ["CG%d" % i for i in range(9)]
        macrocycle_glue_types += ["G%d" % i for i in range(9)]
        if atype in macrocycle_glue_types:
            return None
        return self.reactive_type[reactive_order][atype]


    def get_basetype_and_order(self, atype):
        if len(atype) > 1:
            if atype[0].isdecimal():
                atype = atype[1:] # reactive residues are prefixed with a digit
        if atype not in self.reactive_to_std_atype_mapping:
            return None, None
        basetype = self.reactive_to_std_atype_mapping[atype]
        order = self.reactive_to_order[atype]
        return basetype, order

reactive_typer = ReactiveAtomTyper()
get_reactive_atype = reactive_typer.get_reactive_atype

def assign_reactive_types(molsetup, smarts, smarts_idx, get_reactive_atype=get_reactive_atype):

    atype_dicts = []
    for atom_indices in molsetup.find_pattern(smarts):
        atypes = molsetup.atom_type.copy()
        reactive_atom_index = atom_indices[smarts_idx]

        # type reactive atom
        original_type = molsetup.atom_type[reactive_atom_index]
        reactive_type = get_reactive_atype(original_type, reactive_order=1) 
        atypes[reactive_atom_index] = reactive_type

        # type atoms 1 bond away from reactive atom
        for index1 in molsetup.graph[reactive_atom_index]:
            if not molsetup.atom_ignore[index1]:
                original_type = molsetup.atom_type[index1]
                reactive_type = get_reactive_atype(original_type, reactive_order=2)
                atypes[index1] = reactive_type

            # type atoms 2 bonds away from reactive
            for index2 in molsetup.graph[index1]:
                if index2 == reactive_atom_index:
                    continue
                if not molsetup.atom_ignore[index2]:
                    original_type = molsetup.atom_type[index2]
                    reactive_type = get_reactive_atype(original_type, reactive_order=3)
                    atypes[index2] = reactive_type

        atype_dicts.append(atypes)

    return atype_dicts

### MoleculePreparation ###
class MoleculePreparation:
    def __init__(self,
            merge_these_atom_types=("H",),
            hydrate=False,
            flexible_amides=False,
            rigid_macrocycles=False,
            min_ring_size=7,
            max_ring_size=33,
            keep_chorded_rings=False,
            keep_equivalent_rings=False,
            double_bond_penalty=50,
            rigidify_bonds_smarts=[],
            rigidify_bonds_indices=[],
            atom_type_smarts={},
            add_atom_types=[],
            reactive_smarts=None,
            reactive_smarts_idx=None,
            add_index_map=False,
            remove_smiles=False,
        ):

        self.deprecated_setup_access = None
        self.merge_these_atom_types = merge_these_atom_types
        self.hydrate = hydrate
        self.flexible_amides = flexible_amides
        self.rigid_macrocycles = rigid_macrocycles
        self.min_ring_size = min_ring_size
        self.max_ring_size = max_ring_size
        self.keep_chorded_rings = keep_chorded_rings
        self.keep_equivalent_rings = keep_equivalent_rings
        self.double_bond_penalty = double_bond_penalty
        self.rigidify_bonds_smarts = rigidify_bonds_smarts
        self.rigidify_bonds_indices = rigidify_bonds_indices
        self.atom_type_smarts = atom_type_smarts
        self.add_atom_types = add_atom_types
        self.reactive_smarts = reactive_smarts
        self.reactive_smarts_idx = reactive_smarts_idx
        self.add_index_map = add_index_map
        self.remove_smiles = remove_smiles

        self._atom_typer = AtomTyper(self.atom_type_smarts, self.add_atom_types)
        self._bond_typer = BondTyperLegacy()
        self._macrocycle_typer = FlexMacrocycle(
                self.min_ring_size, self.max_ring_size, self.double_bond_penalty)
        self._flex_builder = FlexibilityBuilder()
        self._classes_setup = {Chem.rdchem.Mol: RDKitMoleculeSetup}
        if keep_chorded_rings and keep_equivalent_rings==False:
            warnings.warn("keep_equivalent_rings=False ignored because keep_chorded_rings=True", RuntimeWarning)
        if (reactive_smarts is None) != (reactive_smarts_idx is None):
            raise ValueError("reactive_smarts and reactive_smarts_idx require each other")

    @property
    def setup(self):
        msg = "MoleculePreparation.setup is deprecated in Meeko v0.5."
        msg += " MoleculePreparation.prepare() returns a list of MoleculeSetup instances."
        warnings.warn(msg, DeprecationWarning)
        return self.deprecated_setup_access

    @classmethod
    def get_defaults_dict(cls):
        defaults = {}
        sig = signature(cls)
        for key in sig.parameters:
            defaults[key] = sig.parameters[key].default 
        return defaults

    @ classmethod
    def from_config(cls, config):
        expected_keys = cls.get_defaults_dict().keys()
        bad_keys = [k for k in config if k not in expected_keys]
        for key in bad_keys:
            print("ERROR: unexpected key \"%s\" in MoleculePreparation.from_config()" % key, file=sys.stderr)
        if len(bad_keys) > 0:
            raise ValueError
        p = cls(**config)
        return p

    def prepare(self,
            mol,
            root_atom_index=None,
            not_terminal_atoms=[],
            delete_ring_bonds=[],
            glue_pseudo_atoms={},
            conformer_id=-1,
        ):
        """ 
        Create molecule setup from RDKit molecule

        Args:
            mol (rdkit.Chem.rdchem.Mol): with explicit hydrogens and 3D coordinates
            root_atom_index (int): to set ROOT of torsion tree instead of searching
            not_terminal_atoms (list): make bonds with terminal atoms rotatable
                                       (e.g. C-Alpha carbon in flexres)
            delete_ring_bonds (list): bonds deleted for macrocycle flexibility
                                      each bond is a tuple of two ints (atom 0-indices)
            glue_pseudo_atoms (dict): keys are parent atom indices, values are (x, y, z)
        """
        mol_type = type(mol)
        if not mol_type in self._classes_setup:
            raise TypeError("Molecule is not an instance of supported types: %s" % type(mol))
        setup_class = self._classes_setup[mol_type]
        setup = setup_class.from_mol(mol,
            keep_chorded_rings=self.keep_chorded_rings,
            keep_equivalent_rings=self.keep_equivalent_rings,
            conformer_id=conformer_id,
            )

        self.check_external_ring_break(setup, delete_ring_bonds, glue_pseudo_atoms)

        # 1.  assign atom types (including HB types, vectors and stuff)
        # DISABLED TODO self.atom_typer.set_parm(mol)
        self._atom_typer(setup)
        # 2a. add pi-model + merge_h_pi (THIS CHANGE SOME ATOM TYPES)
        # disabled

        # merge hydrogens (or any terminal atoms)
        indices = set()
        for atype_to_merge in self.merge_these_atom_types:
            for index, atype in setup.atom_type.items():
                if atype == atype_to_merge:
                    indices.add(index)
        setup.merge_terminal_atoms(indices)

        # 3.  assign bond types by using SMARTS...
        #     - bonds should be typed even in rings (but set as non-rotatable)
        #     - if macrocycle is selected, they will be enabled (so they must be typed already!)
        self._bond_typer(setup, self.flexible_amides, self.rigidify_bonds_smarts, self.rigidify_bonds_indices, not_terminal_atoms)
        # 5.  break macrocycles into open/linear form
        if self.rigid_macrocycles:
            break_combo_data = None
            bonds_in_rigid_rings = None # not true, but this is only needed when breaking macrocycles
        else:
            break_combo_data, bonds_in_rigid_rings = self._macrocycle_typer.search_macrocycle(setup, delete_ring_bonds)

        # 6.  build flexibility...
        # 6.1 if macrocycles typed:
        #     - walk the setup graph by skipping proposed closures
        #       and score resulting flex_trees basing on the lenght
        #       of the branches generated
        #     - actually break the best closure bond (THIS CHANGES SOME ATOM TYPES)
        # 6.2  - walk the graph and build the flextree
        # 7.  but disable all bonds that are in rings and not
        #     in flexible macrocycles
        # TODO restore legacy AD types for PDBQT
        #self._atom_typer.set_param_legacy(mol)

        setup = self._flex_builder(setup,
                                   root_atom_index=root_atom_index,
                                   break_combo_data=break_combo_data,
                                   bonds_in_rigid_rings=bonds_in_rigid_rings,
                                   glue_pseudo_atoms=glue_pseudo_atoms,
        )

        if self.reactive_smarts is None:
            setups = [setup]
        else:
            reactive_types_dicts = assign_reactive_types(
                    setup,
                    self.reactive_smarts,
                    self.reactive_smarts_idx,
            )
            setups = []
            for r in reactive_types_dicts:
                new_setup = setup.copy()
                new_setup.atom_type = r
                setups.append(new_setup)

        self.deprecated_setup_access = setups[0] # for a gentle introduction of the new API
        return setups


    @staticmethod
    def check_external_ring_break(molsetup, break_ring_bonds, glue_pseudo_atoms):
        for (index1, index2) in break_ring_bonds:
            has_bond = molsetup.get_bond_id(index1, index2) in molsetup.bond
            if not has_bond:
                raise ValueError("bond (%d, %d) not in molsetup" % (index1, index2))
            for index in (index1, index2):
                if index not in glue_pseudo_atoms:
                    raise ValueError("missing glue pseudo for atom %d" % index) 
                xyz = glue_pseudo_atoms[index]
                if len(xyz) != 3:
                    raise ValueError("expected 3 coordinates (got %d) for glue pseudo of atom %d" % (len(xyz), index)) 


    def write_pdbqt_string(self, add_index_map=None, remove_smiles=None):
        msg = "MoleculePreparation.write_pdbqt_string() is deprecated in Meeko v0.5."
        msg += " Pass the MoleculeSetup instance to PDBQTWriterLegacy.write_string()."
        msg += " MoleculePreparation.prepare() returns a list of MoleculeSetup instances."
        warnings.warn(msg, DeprecationWarning)
        pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(self.setup)
        if not is_ok:
            msg = 'Cannot generate PDBQT, error from PDBQTWriterLegacy:' + os.linesep
            msg += err_msg
            raise RuntimeError(msg)
        return pdbqt_string


    def write_pdbqt_file(self, pdbqt_filename, add_index_map=None, remove_smiles=None):
        warnings.warn("MoleculePreparation.write_pdbqt_file() is deprecated since Meeko v0.5", DeprecationWarning)
        with open(pdbqt_filename,'w') as w:
            w.write(self.write_pdbqt_string(add_index_map, remove_smiles))
