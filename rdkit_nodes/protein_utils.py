"""
protein_utils — Protein preparation and PDBQT conversion utilities.

Ported from moldocker/utilities/utilis.py for use as node-graph plugin.
PDBFixer/OpenMM imports are lazy — functions that need them will raise a
clear ImportError if the packages are missing.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Geometry import Point3D

# ── Data files ───────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent / 'data'

with open(_DATA_DIR / 'residue_params.json') as _f:
    residue_params = json.load(_f)

with open(_DATA_DIR / 'flexres_templates.json') as _f:
    flexres_templates = json.load(_f)

# ── Constants ────────────────────────────────────────────────────────────────
_retreive_mdl_compiled = re.compile(r'MODEL\s+[0-9]+\s+((\n|.)*?)ENDMDL')

exclude_list = ['HOH', 'SO4', 'GOL', 'ACT']

amino_acids = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]

protein_format_regexes = {
    'pdb': {
        'line':   re.compile(r'(^|\n)(ATOM..|TER...|HETATM).{11}(?!\s+(DT|DA|DC|DG|DI|A|U|C|G|I)).*'),
        'hetatm': re.compile(rf'HETATM.{{11}}(\w{{3}}(?<!{"|".join(exclude_list)})).(\w)(.{{4}}).{{4}}(.{{8}})(.{{8}})(.{{8}})'),
        'aa_het': re.compile(rf'HETATM.{{11}}({"|".join(amino_acids)}).*'),
    },
    'cif': {
        'line':   re.compile(r'ATOM\s+\d+\s+\w+\s[\w|\"|\']+\s+\.\s(DT|DA|DC|DG|DI|A\s|T\s|C\s|G\s|I\s).*'),
        'hetatm': re.compile(rf'HETATM\s+\d+\s+\w+\s+\w+\s+\.\s+(\w+(?<!{"|".join(exclude_list)}))\s+(\w+)\s(\d+)\s\.\s+.\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+'),
        'aa_het': re.compile(rf'HETATM\s+\d+\s+\w+\s+\w+\s+\.\s+({"|".join(amino_acids)}).*'),
    },
}

atom_type_map = {
    'HD': 'H', 'HS': 'H',
    'NA': 'N', 'NS': 'N',
    'A': 'C', 'G': 'C', 'CG0': 'C', 'CG1': 'C', 'CG2': 'C', 'CG3': 'C',
    'G0': 'C', 'G1': 'C', 'G2': 'C', 'G3': 'C',
    'OA': 'O', 'OS': 'O',
    'SA': 'S',
}

atom_property_definitions = {
    'H': 'vdw', 'C': 'vdw', 'A': 'vdw', 'N': 'vdw', 'P': 'vdw', 'S': 'vdw',
    'Br': 'vdw', 'I': 'vdw', 'F': 'vdw', 'Cl': 'vdw',
    'NA': 'hb_acc', 'OA': 'hb_acc', 'SA': 'hb_acc', 'OS': 'hb_acc', 'NS': 'hb_acc',
    'HD': 'hb_don', 'HS': 'hb_don',
    'Mg': 'metal', 'Ca': 'metal', 'Fe': 'metal', 'Zn': 'metal', 'Mn': 'metal',
    'MG': 'metal', 'CA': 'metal', 'FE': 'metal', 'ZN': 'metal', 'MN': 'metal',
    'W': 'water',
    'G0': 'glue', 'G1': 'glue', 'G2': 'glue', 'G3': 'glue',
    'CG0': 'glue', 'CG1': 'glue', 'CG2': 'glue', 'CG3': 'glue',
}

# ── Lazy PDBFixer / OpenMM imports ───────────────────────────────────────────
_pdbfixer_cls = None
_PDBFile_cls = None
_PDBxFile_cls = None


def _ensure_pdbfixer():
    """Lazy-import PDBFixer and openmm.  Raises ImportError with install hint."""
    global _pdbfixer_cls, _PDBFile_cls, _PDBxFile_cls
    if _pdbfixer_cls is not None:
        return
    try:
        from .pdbfixer.pdbfixer import PDBFixer
        from openmm.app import PDBFile, PDBxFile
        _pdbfixer_cls = PDBFixer
        _PDBFile_cls = PDBFile
        _PDBxFile_cls = PDBxFile
    except ImportError as e:
        raise ImportError(
            "Protein preparation requires and OpenMM.\n"
            "Install OpenMM:  pip install openmm\n"
            f"Original error: {e}"
        ) from e


# ══════════════════════════════════════════════════════════════════════════════
#  Low-level PDB / PDBQT helpers
# ══════════════════════════════════════════════════════════════════════════════

def parse_line(line: str) -> dict:
    return {
        'record_name': line[:6].strip(),
        'atom_pos':    int(line[6:11].strip()),
        'atom_name':   line[12:16].strip(),
        'alt_id':      line[16:17],
        'aa_name':     line[17:20].strip(),
        'chain':       line[21].strip(),
        'aa_pos':      int(line[22:26].strip()),
        'insert':      line[26],
        'x':           float(line[30:38].strip()),
        'y':           float(line[38:46].strip()),
        'z':           float(line[46:54].strip()),
        'occupency':   float(line[54:60].strip()),
        'b_factor':    float(line[60:66].strip()),
        'atom_type':   line[76:78].strip(),
    }


def parse_ter(line: str) -> dict:
    return {
        'record_name': line[:6].strip(),
        'atom_pos':    int(line[6:11].strip()),
        'atom_name':   line[12:16].strip(),
        'alt_id':      line[16:17],
        'aa_name':     line[17:20].strip(),
        'chain':       line[21].strip(),
        'aa_pos':      int(line[22:26].strip()),
    }


def _write_pdbqt_line(atomidx, x, y, z, charge, atom_name, res_name, res_num,
                       atom_type, chain, alt_id=" ", in_code="",
                       occupancy=1.0, temp_factor=0.0, record_type="ATOM"):
    if len(atom_name) > 4:
        raise ValueError("max length of atom_name is 4 but atom name is %s" % atom_name)
    atom_name = "%-3s" % atom_name
    line = ("{:6s}{:5d} {:4s}{:1s}{:3s} {:1s}{:4d}{:1s}   "
            "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}\n")
    return line.format(record_type, atomidx, atom_name, alt_id, res_name,
                       chain, res_num, in_code, x, y, z,
                       occupancy, temp_factor, charge, atom_type)


def _read_receptor_pdbqt_string(pdbqt_string, skip_typing=False):
    atoms = []
    atoms_dtype = [
        ('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
        ('resname', 'U3'), ('chain', 'U1'), ('xyz', 'f4', (3)),
        ('partial_charges', 'f4'), ('atom_type', 'U2'),
        ('alt_id', 'U1'), ('in_code', 'U1'),
        ('occupancy', 'f4'), ('temp_factor', 'f4'), ('record_type', 'U6'),
    ]
    atom_annotations = {'hb_acc': [], 'hb_don': [], 'all': [], 'vdw': [], 'metal': []}
    pseudo_atom_types = ['TZ']

    idx = 0
    for line in pdbqt_string.split('\n'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            serial = int(line[6:11].strip())
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chainid = line[21].strip()
            resid = int(line[22:26].strip())
            xyz = np.array([line[30:38].strip(), line[38:46].strip(),
                            line[46:54].strip()], dtype=np.float32)
            try:
                partial_charges = float(line[71:77].strip())
            except Exception:
                partial_charges = 0.0
            atom_type = line[77:79].strip()
            alt_id = line[16:17].strip()
            in_code = line[26:27].strip()
            try:
                occupancy = float(line[54:60])
            except Exception:
                occupancy = 1.0
            try:
                temp_factor = float(line[60:68])
            except Exception:
                temp_factor = 0.0
            record_type = line[0:6].strip()

            if skip_typing:
                atoms.append((idx, serial, name, resid, resname, chainid, xyz,
                              partial_charges, atom_type, alt_id, in_code,
                              occupancy, temp_factor, record_type))
            elif atom_type not in pseudo_atom_types:
                atom_annotations['all'].append(idx)
                atom_annotations[atom_property_definitions[atom_type]].append(idx)
                atoms.append((idx, serial, name, resid, resname, chainid, xyz,
                              partial_charges, atom_type, alt_id, in_code,
                              occupancy, temp_factor, record_type))
            idx += 1

    atoms = np.array(atoms, dtype=atoms_dtype)
    return atoms, atom_annotations


# ══════════════════════════════════════════════════════════════════════════════
#  PDBQTReceptor — atom typing + PDBQT generation
# ══════════════════════════════════════════════════════════════════════════════

class PDBQTReceptor:
    _flexres_templates = flexres_templates
    skip_types = ('H',)

    def __init__(self, pdbqt_string: str, skip_typing=False):
        self._atoms, self._atom_annotations = _read_receptor_pdbqt_string(
            pdbqt_string, skip_typing)
        self.atom_idxs_by_res = self._get_atom_indices_by_residue(self._atoms)

    @staticmethod
    def _get_atom_indices_by_residue(atoms):
        atom_idx_by_res = {}
        for atom_index, atom in enumerate(atoms):
            res_id = (atom['chain'], atom['resname'], atom['resid'])
            atom_idx_by_res.setdefault(res_id, [])
            atom_idx_by_res[res_id].append(atom_index)
        return atom_idx_by_res

    @staticmethod
    def get_params_for_residue(resname, atom_names, params=None):
        if params is None:
            params = residue_params
        excluded_params = ('atom_names', 'bond_cut_atoms', 'bonds')
        atom_params = {}
        atom_counter = 0
        err = ''
        ok = True
        # PDBFixer uses H/H2/H3 for N-terminal amines while AMBER templates
        # use H1/H2/H3.  Build a rename map so we can match either convention.
        _nterm_h_rename = {}  # original_name → template_name
        is_matched = False
        for terminus in ['', 'N', 'C']:
            r_id = '%s%s' % (terminus, resname)
            if r_id not in params:
                err = 'residue %s not in residue_params\n' % r_id
                ok = False
                return atom_params, ok, err
            ref_names = set(params[r_id]['atom_names'])
            query_names = set(atom_names)
            if ref_names == query_names:
                is_matched = True
                break
            # Try H→H1 rename for N-terminal templates
            if terminus == 'N' and 'H1' in ref_names and 'H' in query_names:
                renamed = {('H1' if n == 'H' else n) for n in query_names}
                if ref_names == renamed:
                    _nterm_h_rename = {'H': 'H1'}
                    is_matched = True
                    break

        # Fallback: residues at chain breaks can be both N- and C-terminal
        # (H/H2/H3 from N-term + OXT from C-term).  Match against the
        # N-terminal template and assign OXT standard C-terminal params.
        _oxt_fallback = False
        if not is_matched:
            nterm_id = 'N%s' % resname
            if nterm_id in params:
                query_no_oxt = {n for n in atom_names if n != 'OXT'}
                ref_nterm = set(params[nterm_id]['atom_names'])
                if ref_nterm == query_no_oxt:
                    r_id = nterm_id
                    _oxt_fallback = True
                    is_matched = True
                # Also try H→H1 rename + OXT strip
                elif 'H1' in ref_nterm and 'H' in query_no_oxt:
                    renamed = {('H1' if n == 'H' else n) for n in query_no_oxt}
                    if ref_nterm == renamed:
                        r_id = nterm_id
                        _nterm_h_rename = {'H': 'H1'}
                        _oxt_fallback = True
                        is_matched = True

        if not is_matched:
            ok = False
            return atom_params, ok, err

        for atom_name in atom_names:
            if atom_name == 'OXT' and _oxt_fallback:
                # Assign standard C-terminal OXT params
                for param in params[r_id]:
                    if param in excluded_params:
                        continue
                    if param not in atom_params:
                        atom_params[param] = [None] * atom_counter
                    if param == 'atom_types':
                        atom_params[param].append('OA')
                    elif param == 'gasteiger':
                        atom_params[param].append(-0.532)
                    else:
                        atom_params[param].append(None)
                atom_counter += 1
                continue
            lookup_name = _nterm_h_rename.get(atom_name, atom_name)
            name_index = params[r_id]['atom_names'].index(lookup_name)
            for param in params[r_id]:
                if param in excluded_params:
                    continue
                if param not in atom_params:
                    atom_params[param] = [None] * atom_counter
                atom_params[param].append(params[r_id][param][name_index])
            atom_counter += 1
        return atom_params, ok, err

    def assign_types_charges(self):
        wanted_params = ('atom_types', 'gasteiger')
        atom_params = {key: [] for key in wanted_params}
        ok = True
        err = ''
        for r_id, atom_indices in self.atom_idxs_by_res.items():
            atom_names = tuple(self.atoms(atom_indices)['name'])
            resname = r_id[1]
            params_this_res, ok_, err_ = self.get_params_for_residue(resname, atom_names)
            ok &= ok_
            err += err_
            if not ok_:
                print('did not match %s with template' % str(r_id), file=sys.stderr)
                continue
            for key in wanted_params:
                atom_params[key].extend(params_this_res[key])
        if ok:
            self._atoms['partial_charges'] = atom_params['gasteiger']
            self._atoms['atom_type'] = atom_params['atom_types']
        return ok, err

    def write_flexres_from_template(self, res_id, atom_index=0):
        success = True
        error_msg = ''
        branch_offset = atom_index
        output = {'pdbqt': '', 'flex_indices': [], 'atom_index': atom_index}
        resname = res_id[1]
        if resname not in self._flexres_templates:
            return output, False, 'no flexible residue template for resname %s' % resname
        if res_id not in self.atom_idxs_by_res:
            chains = set(self._atoms['chain'])
            error_msg = "could not find residue with chain='%s', resname=%s, resnum=%d\n" % res_id
            error_msg += 'chains in this receptor: %s\n' % ', '.join("'%s'" % c for c in chains)
            return output, False, error_msg

        atoms_by_name = {}
        for i in self.atom_idxs_by_res[res_id]:
            name = self._atoms[i]['name']
            if name in ['C', 'N', 'O', 'H', 'H1', 'H2', 'H3', 'OXT']:
                continue
            atype = self._atoms[i]['atom_type']
            if atype in self.skip_types:
                continue
            output['flex_indices'].append(i)
            atoms_by_name[name] = self.atoms(i)

        template = self._flexres_templates[resname]
        got_atoms = set(atoms_by_name)
        ref_atoms = set()
        for i in range(len(template['is_atom'])):
            if template['is_atom'][i]:
                ref_atoms.add(template['atom_name'][i])
        if got_atoms != ref_atoms:
            error_msg = 'mismatch in atom names for residue %s\n' % str(res_id)
            error_msg += 'names found but not in template: %s\n' % str(got_atoms.difference(ref_atoms))
            error_msg += 'missing names: %s\n' % str(ref_atoms.difference(got_atoms))
            return output, False, error_msg

        n_lines = len(template['is_atom'])
        for i in range(n_lines):
            if template['is_atom'][i]:
                atom_index += 1
                name = template['atom_name'][i]
                atom = atoms_by_name[name]
                if atom['atom_type'] not in self.skip_types:
                    atom['serial'] = atom_index
                    output['pdbqt'] += self._write_pdbqt_line_from_atom(atom)
            else:
                line = template['original_line'][i]
                if branch_offset > 0 and (line.startswith('BRANCH') or line.startswith('ENDBRANCH')):
                    keyword, ii, jj = line.split()
                    ii = int(ii) + branch_offset
                    jj = int(jj) + branch_offset
                    line = '%s %3d %3d' % (keyword, ii, jj)
                output['pdbqt'] += line + '\n'
        output['atom_index'] = atom_index
        return output, success, error_msg

    @staticmethod
    def _write_pdbqt_line_from_atom(atom):
        return _write_pdbqt_line(
            atom['serial'], atom['xyz'][0], atom['xyz'][1], atom['xyz'][2],
            atom['partial_charges'], atom['name'], atom['resname'],
            atom['resid'], atom['atom_type'], atom['chain'],
            atom['alt_id'], atom['in_code'], atom['occupancy'],
            atom['temp_factor'], atom['record_type'])

    def write_pdbqt_string(self, flexres=()):
        ok = True
        err = ''
        pdbqt = {'rigid': '', 'flex': {}, 'flex_indices': []}
        atom_index = 0
        for res_id in set(flexres):
            output, ok_, err_ = self.write_flexres_from_template(res_id, atom_index)
            atom_index = output['atom_index']
            ok &= ok_
            err += err_
            pdbqt['flex_indices'].extend(output['flex_indices'])
            pdbqt['flex'][res_id] = ''
            pdbqt['flex'][res_id] += 'BEGIN_RES %3s %1s%4d\n' % res_id
            pdbqt['flex'][res_id] += output['pdbqt']
            pdbqt['flex'][res_id] += 'END_RES %3s %1s%4d\n' % res_id

        all_flex_pdbqt = ''
        for res_id, flexres_pdbqt in pdbqt['flex'].items():
            all_flex_pdbqt += flexres_pdbqt
        pdbqt['flex'] = all_flex_pdbqt

        for i, atom in enumerate(self._atoms):
            if i not in pdbqt['flex_indices'] and atom['atom_type'] not in self.skip_types:
                pdbqt['rigid'] += self._write_pdbqt_line_from_atom(atom)

        return pdbqt, ok, err

    def atoms(self, atom_idx=None):
        if atom_idx is not None and self._atoms.size > 1:
            if not isinstance(atom_idx, (list, tuple, np.ndarray)):
                atom_idx = np.array(atom_idx, dtype=int)
            return self._atoms[atom_idx].copy()
        return self._atoms.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  PDB cleaning / parsing
# ══════════════════════════════════════════════════════════════════════════════

def clean_pdb(protein_str: str, return_hetatm: bool = False,
              fill_gap: bool = False, format: str = None):
    """Strip non-protein atoms from a PDB/CIF string.

    Returns the cleaned PDB string.  If *return_hetatm* is True, also
    returns a dict of HETATM ligands with their bounding-box info.
    """
    if format is None:
        format = 'cif' if protein_str.startswith('data_') else 'pdb'

    retrieve_line = protein_format_regexes[format]['line']
    exclude_aa_het = protein_format_regexes[format]['aa_het']

    s = re.search(_retreive_mdl_compiled, protein_str)
    if s is not None:
        header_strs = []
        for l in protein_str.splitlines():
            if l.startswith('MODEL        1'):
                break
            header_strs.append(l)
        protein_str = '\n'.join(header_strs) + '\n' + s.group(0)

    if not fill_gap:
        if format == 'pdb':
            final = ''.join(
                [m.group(0) for m in re.finditer(retrieve_line, protein_str)]
            ).strip()
        elif format == 'cif':
            final = '\n'.join(
                l for l in protein_str.splitlines()
                if not retrieve_line.match(l)
            )
    else:
        final = protein_str

    if not return_hetatm:
        final = '\n'.join(
            l for l in final.splitlines() if not exclude_aa_het.match(l))
        return final
    else:
        retrieve_hetatm = protein_format_regexes[format]['hetatm']
        hetatm_dict = {}
        for matched in re.finditer(retrieve_hetatm, protein_str):
            name, chain, pos, x, y, z = matched.group(1, 2, 3, 4, 5, 6)
            ligand = f'[{name}]{pos.strip()}:{chain}'
            hetatm_dict.setdefault(ligand, [])
            hetatm_dict[ligand].append([float(x), float(y), float(z)])

        final_hetatm_dict = {}
        for ligand, xyz in hetatm_dict.items():
            xyz = np.array(xyz)
            max_xyz = xyz.max(0) + 1.5
            min_xyz = xyz.min(0) - 1.5
            box = np.round(max_xyz - min_xyz, 3)
            center = np.round((max_xyz + min_xyz) / 2, 3)
            volume = np.prod(box).round(3)
            final_hetatm_dict[ligand] = {
                'Center': list(center),
                'Box': list(box),
                'Volume': float(volume),
            }
        final = '\n'.join(
            l for l in final.splitlines() if not exclude_aa_het.match(l))
        return final, final_hetatm_dict


def read_pdb_string(pdb_str: str):
    """Parse a cleaned PDB string into a DataFrame + CONECT records."""
    result = []
    conect_record = []
    for l in pdb_str.splitlines():
        if l.startswith('ATOM'):
            result.append(parse_line(l))
        elif l.startswith('TER'):
            result.append(parse_ter(l))
        elif l.startswith('CONECT'):
            conect_record.append(l)
    return pd.DataFrame(result), conect_record


def read_pdb_file(pdb_pth: str):
    with open(pdb_pth) as f:
        return read_pdb_string(f.read())


def get_pdb_string(aa_df: pd.DataFrame) -> str:
    """Convert an atom DataFrame back to PDB-format string."""
    pdb_fmt = '{:6s}{:5d} {:4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}           {:<2s}'
    ter_fmt = '{:6s}{:5d} {:4s}{:1s}{:3s} {:1s}{:4d}'
    lines = []
    for _, row in aa_df.iterrows():
        if row['record_name'] == 'TER':
            lines.append(ter_fmt.format(*row))
        else:
            lines.append(pdb_fmt.format(*row))
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  PDBFixer wrapper (requires OpenMM)
# ══════════════════════════════════════════════════════════════════════════════

def fix_pdb_missing_atoms(pdb_pth: str, out_pth: str | None = None,
                          replace_nonstandard: bool = True,
                          fill_gap: bool = False, ph: float = 7.0) -> str:
    """Use PDBFixer to repair a PDB/CIF structure (add missing atoms/H).

    Accepts both PDB and CIF (mmCIF) files — PDBFixer auto-detects format.
    Always outputs PDB format (needed for downstream PDBQT conversion).
    """
    _ensure_pdbfixer()

    # Rename non-standard amino acid codes in the raw PDB text before
    # PDBFixer parses them (e.g. DUD-E uses CYT for free-thiol CYS,
    # GLV for γ-glutamate).  PDBFixer misinterprets CYT as cytosine.
    _RESIDUE_ALIASES = {'CYT': 'CYS', 'GLV': 'GLU'}
    import tempfile as _tmpmod
    with open(pdb_pth) as _f:
        pdb_text = _f.read()
    needs_rewrite = False
    for old, new in _RESIDUE_ALIASES.items():
        if old in pdb_text:
            pdb_text = pdb_text.replace(old, new)
            needs_rewrite = True
    if needs_rewrite:
        _tmp = _tmpmod.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
        _tmp.write(pdb_text)
        _tmp.close()
        pdb_pth = _tmp.name

    fixer = _pdbfixer_cls(pdb_pth)

    if needs_rewrite:
        os.unlink(pdb_pth)

    fixer.findNonstandardResidues()
    nonstandards = fixer.nonstandardResidues.copy()

    for chain_resname in nonstandards:
        tmp = chain_resname[0]
        chain = tmp.chain
        if tmp.index > len(list(chain.residues())):
            fixer.nonstandardResidues.remove(chain_resname)

    if fill_gap:
        fixer.findMissingResidues()
        chains = list(fixer.topology.chains())
        keys = list(fixer.missingResidues.keys())
        for key in keys:
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                del fixer.missingResidues[key]
    else:
        fixer.missingResidues = {}

    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Strip existing H before re-adding — some PDBs (e.g. DUD-E) have
    # non-standard hydrogen names (HN, HNE, HN11 …) that confuse the
    # residue template matching.  Remove them all so addMissingHydrogens()
    # can add correctly-named ones from scratch.
    from openmm.app import Modeller
    from openmm.app.element import hydrogen as _H_elem
    modeller = Modeller(fixer.topology, fixer.positions)
    h_atoms = [a for a in modeller.topology.atoms() if a.element == _H_elem]
    if h_atoms:
        modeller.delete(h_atoms)
    fixer.topology = modeller.topology
    fixer.positions = modeller.positions

    fixer.addMissingHydrogens(ph)

    # Always write as PDB (PDBQT pipeline needs PDB-format ATOM records)
    if out_pth is not None:
        _PDBFile_cls.writeFile(fixer.topology, fixer.positions, out_pth, keepIds=True)
    else:
        sio = io.StringIO()
        _PDBFile_cls.writeFile(fixer.topology, fixer.positions, sio, keepIds=True)
        return sio.getvalue()


def fix_pdb_missing_atoms_from_string(pdb_str: str, ph: float = 7.0,
                                      fill_gap: bool = False,
                                      format: str = 'pdb') -> str:
    """Like fix_pdb_missing_atoms but accepts a PDB/CIF string.

    Args:
        format: 'pdb' or 'cif' — determines temp file suffix for PDBFixer.
    """
    _ensure_pdbfixer()
    import tempfile
    suffix = '.cif' if format == 'cif' else '.pdb'
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as tmp:
        tmp.write(pdb_str)
        tmp_path = tmp.name
    try:
        return fix_pdb_missing_atoms(tmp_path, fill_gap=fill_gap, ph=ph)
    finally:
        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
#  Amino acid protonation / correction checks
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_disulfide_bonds(conect_record: list) -> set:
    bonds = []
    for l in conect_record:
        pos1, pos2 = int(l[6:11].strip()), int(l[11:16].strip())
        bonds.extend([pos1, pos2])
    return set(bonds)


def _check_histidine(aa_df: pd.DataFrame, *args):
    chains = list(dict.fromkeys(aa_df['chain']))
    result = []
    for c in chains:
        chain_df = aa_df[aa_df['chain'] == c]
        for aa_pos in set(chain_df[chain_df['aa_name'] == 'HIS']['aa_pos']):
            curr_pos = chain_df['aa_pos'] == aa_pos
            names = set(chain_df[curr_pos]['atom_name'])
            if {'HD1', 'HE2'}.issubset(names):
                chain_df.loc[curr_pos, 'aa_name'] = 'HIP'
            elif {'HD1'}.issubset(names):
                chain_df.loc[curr_pos, 'aa_name'] = 'HID'
            elif {'HE2'}.issubset(names):
                chain_df.loc[curr_pos, 'aa_name'] = 'HIE'
        result.append(chain_df)
    return pd.concat(result, axis=0, ignore_index=True)


def _check_glutamate(aa_df: pd.DataFrame, *args):
    chains = list(dict.fromkeys(aa_df['chain']))
    result = []
    for c in chains:
        chain_df = aa_df[aa_df['chain'] == c]
        for aa_pos in set(chain_df[chain_df['aa_name'] == 'GLU']['aa_pos']):
            curr_pos = chain_df['aa_pos'] == aa_pos
            if {'HE2'}.issubset(set(chain_df[curr_pos]['atom_name'])):
                chain_df.loc[curr_pos, 'aa_name'] = 'GLH'
        result.append(chain_df)
    return pd.concat(result, axis=0, ignore_index=True)


def _check_aspartate(aa_df: pd.DataFrame, *args):
    chains = list(dict.fromkeys(aa_df['chain']))
    result = []
    for c in chains:
        chain_df = aa_df[aa_df['chain'] == c]
        for aa_pos in set(chain_df[chain_df['aa_name'] == 'ASP']['aa_pos']):
            curr_pos = chain_df['aa_pos'] == aa_pos
            if {'HD2'}.issubset(set(chain_df[curr_pos]['atom_name'])):
                chain_df.loc[curr_pos, 'aa_name'] = 'ASH'
        result.append(chain_df)
    return pd.concat(result, axis=0, ignore_index=True)


def _check_lysine(aa_df: pd.DataFrame, *args):
    chains = list(dict.fromkeys(aa_df['chain']))
    result = []
    for c in chains:
        chain_df = aa_df[aa_df['chain'] == c]
        for aa_pos in set(chain_df[chain_df['aa_name'] == 'LYS']['aa_pos']):
            curr_pos = chain_df['aa_pos'] == aa_pos
            if not {'HZ1'}.issubset(set(chain_df[curr_pos]['atom_name'])):
                chain_df.loc[curr_pos, 'aa_name'] = 'LYN'
        result.append(chain_df)
    return pd.concat(result, axis=0, ignore_index=True)


def _check_cysteine(aa_df: pd.DataFrame, conect_record: list):
    disulfide_bonds = None
    chains = list(dict.fromkeys(aa_df['chain']))
    result = []
    for c in chains:
        chain_df = aa_df[aa_df['chain'] == c]
        all_cys = set(chain_df[chain_df['aa_name'] == 'CYS']['aa_pos'])
        for aa_pos in all_cys:
            curr_pos = chain_df['aa_pos'] == aa_pos
            names = set(chain_df[curr_pos]['atom_name'])
            if not {'HG'}.issubset(names):
                if len(all_cys) == 1:
                    chain_df.loc[curr_pos, 'aa_name'] = 'CYM'
                else:
                    if disulfide_bonds is None:
                        disulfide_bonds = retrieve_disulfide_bonds(conect_record)
                    if disulfide_bonds:
                        atom_pos = set(chain_df[curr_pos]['atom_pos'].to_list())
                        if disulfide_bonds.intersection(atom_pos):
                            chain_df.loc[curr_pos, 'aa_name'] = 'CYX'
                        else:
                            chain_df.loc[curr_pos, 'aa_name'] = 'CYM'
        result.append(chain_df)
    return pd.concat(result, axis=0, ignore_index=True)


def _check_for_N_terminal_H(aa_df: pd.DataFrame, *args):
    chains = list(dict.fromkeys(aa_df['chain']))
    result = []
    for c in chains:
        chain_df = aa_df[aa_df['chain'] == c]
        first_aa_pos = chain_df.iloc[0]['aa_pos']
        curr_pos = chain_df['aa_pos'] == first_aa_pos
        names = set(chain_df[curr_pos]['atom_name'])
        if {'H2', 'H3'}.issubset(names):
            chain_df.loc[curr_pos & (chain_df['atom_name'] == 'H'), 'atom_name'] = 'H1'
        elif {'H3'}.issubset(names):
            chain_df.loc[curr_pos & (chain_df['atom_name'] == 'H'), 'atom_name'] = 'H2'
        elif {'H2'}.issubset(names):
            chain_df.loc[curr_pos & (chain_df['atom_name'] == 'H'), 'atom_name'] = 'H3'
        result.append(chain_df)
    aa_df = pd.concat(result, axis=0, ignore_index=True)
    aa_df['atom_pos'] = list(range(1, len(aa_df) + 1))
    return aa_df


def _check_for_inserts(aa_df: pd.DataFrame, *args):
    finalidx_chain_dict = {}
    last_aa_pos = aa_df.loc[0, 'aa_pos']
    last_insert = aa_df.loc[0, 'insert']
    for idx, row in aa_df.iterrows():
        if idx == 0:
            continue
        curr_aa_pos, curr_insert = row['aa_pos'], row['insert']
        if (curr_aa_pos == last_aa_pos) and (curr_insert != last_insert):
            finalidx_chain_dict[idx] = row['chain']
        last_aa_pos, last_insert = curr_aa_pos, curr_insert
    for idx, chain in finalidx_chain_dict.items():
        mask = (aa_df['chain'] == chain) & (aa_df.index >= idx)
        aa_df.loc[mask, 'aa_pos'] += 1
    aa_df['insert'] = ' '
    return aa_df


def _check_for_aa_count(aa_df: pd.DataFrame):
    chain_unique_cnt = aa_df.groupby('chain')['aa_pos'].nunique()
    single_aa = chain_unique_cnt[chain_unique_cnt == 1]
    if not single_aa.empty:
        for chain in single_aa.index:
            aa_df = aa_df[aa_df['chain'] != chain]
    return aa_df


def check_amino_acids(aa_df: pd.DataFrame, conect_record: list) -> pd.DataFrame:
    """Apply amino-acid–specific corrections (protonation, inserts, etc.)."""
    aa_func_map = {
        'INS':   _check_for_inserts,
        'HIS':   _check_histidine,
        'GLU':   _check_glutamate,
        'ASP':   _check_aspartate,
        'LYS':   _check_lysine,
        'CYS':   _check_cysteine,
        'N_ter': _check_for_N_terminal_H,
    }
    for fn in aa_func_map.values():
        aa_df = fn(aa_df, conect_record)
    return aa_df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  High-level protein preparation
# ══════════════════════════════════════════════════════════════════════════════

def write_to_pdbqt(aa_df: pd.DataFrame, output_pth: str | None = None):
    """Convert an atom DataFrame → PDBQT string (or write to file)."""
    pdb_string = get_pdb_string(aa_df)
    receptor = PDBQTReceptor(pdb_string, skip_typing=True)
    ok, err = receptor.assign_types_charges()
    if ok:
        pdbqt_string, ok, err = receptor.write_pdbqt_string()
        if ok:
            if output_pth is not None:
                with open(output_pth, 'w') as f:
                    f.write(pdbqt_string['rigid'])
            else:
                return pdbqt_string['rigid']
    return (err,)


def fix_and_convert(pdb_pth: str, output_pth: str = None,
                    fill_gap: bool = False, ph: float = 7.0):
    """Master pipeline: PDBFixer → parse → protonation checks → PDBQT.

    Returns the PDBQT string on success, or a 1-tuple ``(error_msg,)`` on failure.
    """
    try:
        pdb_str = fix_pdb_missing_atoms(pdb_pth, fill_gap=fill_gap, ph=ph)
    except Exception:
        return ('PDB file lacks protein!',)
    df, conect_record = read_pdb_string(pdb_str)
    df = _check_for_aa_count(df)
    if df.empty:
        return ('PDB file lacks protein!',)
    df = check_amino_acids(df, conect_record)
    return write_to_pdbqt(df, output_pth)


def fix_and_convert_from_string(pdb_str: str, ph: float = 7.0,
                                fill_gap: bool = False,
                                format: str = 'pdb'):
    """Like fix_and_convert but accepts a PDB/CIF string.

    Args:
        format: 'pdb' or 'cif' — determines temp file suffix for PDBFixer.
    """
    import tempfile
    suffix = '.cif' if format == 'cif' else '.pdb'
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as tmp:
        tmp.write(pdb_str)
        tmp_path = tmp.name
    try:
        return fix_and_convert(tmp_path, fill_gap=fill_gap, ph=ph)
    finally:
        os.unlink(tmp_path)


def process_rigid_flex(pdbqt_str: str, flex_res: set):
    """Split a PDBQT string into rigid + flexible parts.

    Returns ``(rigid_pdbqt, flex_pdbqt, ok, err)``.
    """
    receptor = PDBQTReceptor(pdbqt_str, skip_typing=True)
    pdbqt, ok, err = receptor.write_pdbqt_string(flex_res)
    return pdbqt['rigid'], pdbqt['flex'], ok, err


def pdbqt_to_pdb(pdbqt_str: str):
    """Convert a receptor PDBQT string back to PDB format."""
    meet_end_of_chain = 0
    sub_re_map = {
        'HIS': r'HID|HIP|HIE',
        'GLU': r'GLH',
        'ASP': r'ASH',
        'LYS': r'LYN',
        'CYS': r'CYM|CYX',
    }
    last_chain = last_atomidx = last_resname = last_respos = None

    def map_pdbqt_line_to_pdb(line, idx):
        nonlocal meet_end_of_chain, last_chain, last_resname, last_respos, last_atomidx
        atom_type = line[77:].strip()
        if atom_type in atom_type_map:
            atom_type = atom_type_map[atom_type]
        chain = line[21]
        res_name = line[17:20].strip()
        res_pos = int(line[22:26].strip())
        final = []
        if chain != last_chain and last_chain is not None:
            final.append({
                'atom_idx': last_atomidx + 1,
                'res_name': last_resname,
                'chain': last_chain,
                'res_pos': last_respos,
            })
            meet_end_of_chain += 1
        atom_idx = idx + meet_end_of_chain
        final.append({
            'atom_idx': atom_idx,
            'atom_name': line[12:16].strip(),
            'alt_id': line[16],
            'res_name': res_name,
            'chain': chain,
            'res_pos': res_pos,
            'others': line[26:66],
            'atom_type': atom_type,
        })
        last_chain, last_resname, last_respos, last_atomidx = chain, res_name, res_pos, atom_idx
        return final

    def convert_to_pdb_str(atom_data):
        lines = []
        fmt = 'ATOM  {:5d} {:4s}{:1s}{:3s} {:1s}{:4d}{:40s}            {}\n'
        ter = 'TER   {:5d}      {:3s} {:1s}{:4d}\n'
        for entry in atom_data:
            if len(entry) == 8:
                lines.append(fmt.format(
                    entry['atom_idx'], entry['atom_name'], entry['alt_id'],
                    entry['res_name'], entry['chain'], entry['res_pos'],
                    entry['others'], entry['atom_type']))
            else:
                lines.append(ter.format(
                    entry['atom_idx'], entry['res_name'],
                    entry['chain'], entry['res_pos']))
        return ''.join(lines)

    for aa, re_comp in sub_re_map.items():
        pdbqt_str = re.sub(re_comp, aa, pdbqt_str)

    protein_data = [
        item
        for idx, line in enumerate(pdbqt_str.strip().splitlines())
        if line.startswith('ATOM')
        for item in map_pdbqt_line_to_pdb(line, idx)
    ]
    final_ter = {
        'atom_idx': last_atomidx + 1,
        'res_name': last_resname,
        'chain': last_chain,
        'res_pos': last_respos,
    }
    protein_data.append(final_ter)
    return protein_data, convert_to_pdb_str(protein_data)


# ══════════════════════════════════════════════════════════════════════════════
#  PDBQTMolecule — parse docking output (multi-pose PDBQT / DLG)
# ══════════════════════════════════════════════════════════════════════════════

def _read_ligand_pdbqt_file(pdbqt_string, poses_to_read=-1, energy_range=-1,
                            is_dlg=False, skip_typing=False):
    i = 0
    n_poses = 0
    previous_serial = 0
    tmp_positions = []
    tmp_atoms = []
    tmp_actives = []
    tmp_pdbqt_string = ''
    water_indices = set()
    location = 'ligand'
    energy_best_pose = None
    is_first_pose = True
    is_model = False
    mol_index = -1

    atoms_dtype = [
        ('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
        ('resname', 'U3'), ('chain', 'U1'), ('xyz', 'f4', (3)),
        ('partial_charges', 'f4'), ('atom_type', 'U3'),
    ]

    atoms = None
    positions = []

    atom_annotations = {
        'ligand': [], 'flexible_residue': [], 'water': [],
        'hb_acc': [], 'hb_don': [], 'all': [], 'vdw': [],
        'glue': [], 'reactive': [], 'metal': [],
        'mol_index': {},
    }
    pose_data = {
        'n_poses': None,
        'active_atoms': [],
        'free_energies': [],
        'intermolecular_energies': [],
        'internal_energies': [],
        'index_map': {},
        'pdbqt_string': [],
        'smiles': {},
        'smiles_index_map': {},
        'smiles_h_parent': {},
        'cluster_id': [],
        'rank_in_cluster': [],
        'cluster_leads_sorted': [],
        'cluster_size': [],
    }
    tmp_cluster_data = {}
    buffer_index_map = {}
    buffer_smiles = None
    buffer_smiles_index_map = []
    buffer_smiles_h_parent = []

    lines = pdbqt_string.split('\n')
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    lines = [line + '\n' for line in lines]

    for line in lines:
        if is_dlg:
            if line.startswith('DOCKED'):
                line = line[8:]
            elif line.endswith('RANKING\n'):
                fields = line.split()
                cluster_id = int(fields[0])
                subrank = int(fields[1])
                run_id = int(fields[2])
                tmp_cluster_data[run_id] = (cluster_id, subrank)
            else:
                continue

        if not line.startswith(('MODEL', 'ENDMDL')):
            tmp_pdbqt_string += line

        if line.startswith('MODEL'):
            i = 0
            previous_serial = 0
            tmp_positions = []
            tmp_atoms = []
            tmp_actives = []
            tmp_pdbqt_string = ''
            is_model = True
            mol_index = -1
        elif line.startswith('ATOM') or line.startswith('HETATM'):
            serial = int(line[6:11].strip())
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chainid = line[21].strip()
            resid = int(line[22:26].strip())
            xyz = np.array([line[30:38].strip(), line[38:46].strip(),
                            line[46:54].strip()], dtype=float)
            try:
                partial_charges = float(line[71:77].strip())
            except Exception:
                partial_charges = 0.0
            atom_type = line[77:-1].strip()

            if (previous_serial + 1 != serial) and not (serial < previous_serial and serial == 1):
                diff = serial - previous_serial - 1
                for _ in range(diff):
                    xyz_nan = [999.999, 999.999, 999.999]
                    tmp_atoms.append((i, 9999, 'XXXX', 9999, 'XXX', 'X', xyz_nan, 999.999, 'XX'))
                    tmp_positions.append(xyz_nan)
                    i += 1

            tmp_atoms.append((i, serial, name, resid, resname, chainid, xyz, partial_charges, atom_type))
            tmp_positions.append(xyz)
            tmp_actives.append(i)

            if is_first_pose:
                atom_annotations['mol_index'].setdefault(mol_index, [])
                atom_annotations['mol_index'][mol_index].append(i)
                if atom_type != 'W':
                    atom_annotations[location].append(i)
                    atom_annotations['all'].append(i)
                    if not skip_typing:
                        atom_annotations[atom_property_definitions[atom_type]].append(i)

            if atom_type == 'W':
                water_indices.add(i)

            previous_serial = serial
            i += 1
        elif line.startswith('ROOT') and is_first_pose:
            mol_index += 1
            pose_data['index_map'][mol_index] = buffer_index_map
            pose_data['smiles'][mol_index] = buffer_smiles
            pose_data['smiles_index_map'][mol_index] = buffer_smiles_index_map
            pose_data['smiles_h_parent'][mol_index] = buffer_smiles_h_parent
            buffer_index_map = {}
            buffer_smiles = None
            buffer_smiles_index_map = []
            buffer_smiles_h_parent = []
        elif line.startswith('REMARK INDEX MAP') and is_first_pose:
            integers = [int(x) for x in line.split()[3:]]
            if len(integers) % 2 == 1:
                raise RuntimeError('Number of indices in INDEX MAP is odd')
            for j in range(len(integers) // 2):
                buffer_index_map[integers[j * 2]] = integers[j * 2 + 1]
        elif line.startswith('REMARK SMILES IDX') and is_first_pose:
            integers = [int(x) for x in line.split()[3:]]
            if len(integers) % 2 == 1:
                raise RuntimeError('Number of indices in SMILES IDX is odd')
            buffer_smiles_index_map.extend(integers)
        elif line.startswith('REMARK H PARENT') and is_first_pose:
            integers = [int(x) for x in line.split()[3:]]
            if len(integers) % 2 == 1:
                raise RuntimeError('Number of indices in H PARENT is odd')
            buffer_smiles_h_parent.extend(integers)
        elif line.startswith('REMARK SMILES') and is_first_pose:
            buffer_smiles = line.split()[2]
        elif line.startswith('REMARK VINA RESULT') or line.startswith('USER    Estimated Free Energy of Binding    ='):
            try:
                energy = float(line.split()[3])
            except Exception:
                energy = float(line[45:].split()[0])
            if energy_best_pose is None:
                energy_best_pose = energy
            diff_energy = energy - energy_best_pose
            if energy_range != -1 and energy_range <= diff_energy:
                break
            pose_data['free_energies'].append(energy)
        elif not is_dlg and line.startswith('REMARK INTER:'):
            pose_data['intermolecular_energies'].append(float(line.split()[2]))
        elif not is_dlg and line.startswith('REMARK INTRA:'):
            pose_data['internal_energies'].append(float(line.split()[2]))
        elif is_dlg and line.startswith('USER    (1) Final Intermolecular Energy     ='):
            pose_data['intermolecular_energies'].append(float(line[45:].split()[0]))
        elif is_dlg and line.startswith('USER    (2) Final Total Internal Energy     ='):
            pose_data['internal_energies'].append(float(line[45:].split()[0]))
        elif line.startswith('BEGIN_RES'):
            location = 'flexible_residue'
        elif line.startswith('END_RES'):
            location = 'ligand'
        elif line.startswith('ENDMDL'):
            n_poses += 1
            is_first_pose = False
            tmp_atoms = np.array(tmp_atoms, dtype=atoms_dtype)
            if atoms is None:
                atoms = tmp_atoms.copy()
            else:
                columns = ['idx', 'serial', 'name', 'resid', 'resname', 'chain', 'partial_charges', 'atom_type']
                topology1 = atoms[np.isin(atoms['atom_type'], ['W', 'XX'], invert=True)][columns]
                topology2 = tmp_atoms[np.isin(atoms['atom_type'], ['W', 'XX'], invert=True)][columns]
                if not np.array_equal(topology1, topology2):
                    raise RuntimeError('molecules have different topologies')
                tmp_water_idx = tmp_atoms[tmp_atoms['atom_type'] == 'W']['idx']
                water_idx = atoms[atoms['atom_type'] == 'XX']['idx']
                new_water_idx = list(set(tmp_water_idx).intersection(water_idx))
                atoms[new_water_idx] = tmp_atoms[new_water_idx]

            positions.append(tmp_positions)
            pose_data['active_atoms'].append(tmp_actives)
            pose_data['pdbqt_string'].append(tmp_pdbqt_string)
            if n_poses >= poses_to_read and poses_to_read != -1:
                break

    if not is_model:
        n_poses += 1
        atoms = np.array(tmp_atoms, dtype=atoms_dtype)
        positions.append(tmp_positions)
        pose_data['active_atoms'].append(tmp_actives)
        pose_data['pdbqt_string'].append(tmp_pdbqt_string)

    positions = np.array(positions).reshape((n_poses, atoms.shape[0], 3))
    pose_data['n_poses'] = n_poses

    if water_indices:
        atom_annotations['water'] = list(water_indices)

    if tmp_cluster_data:
        if len(tmp_cluster_data) != n_poses:
            raise RuntimeError(
                'Nr of poses in cluster data (%d) differs from nr of poses (%d)'
                % (len(tmp_cluster_data), n_poses))
        pose_data['cluster_id'] = [None] * n_poses
        pose_data['rank_in_cluster'] = [None] * n_poses
        pose_data['cluster_size'] = [None] * n_poses
        cluster_ids = [cid for _, (cid, _) in tmp_cluster_data.items()]
        n_clusters = max(cluster_ids)
        pose_data['cluster_leads_sorted'] = [None] * n_clusters
        for pose_index, (cid, rank) in tmp_cluster_data.items():
            pose_data['cluster_id'][pose_index - 1] = cid
            pose_data['rank_in_cluster'][pose_index - 1] = rank
            pose_data['cluster_size'][pose_index - 1] = cluster_ids.count(cid)
            if rank == 1:
                pose_data['cluster_leads_sorted'][cid - 1] = pose_index - 1

    return atoms, positions, atom_annotations, pose_data


class PDBQTMolecule:
    """Parse multi-pose PDBQT / DLG docking output."""

    def __init__(self, pdbqt_string, name=None, poses_to_read=None,
                 energy_range=None, is_dlg=False, skip_typing=False):
        self._current_pose = 0
        self._pdbqt_filename = None
        self._name = name

        ptr = poses_to_read if poses_to_read is not None else -1
        er = energy_range if energy_range is not None else -1
        results = _read_ligand_pdbqt_file(pdbqt_string, ptr, er, is_dlg, skip_typing)
        self._atoms, self._positions, self._atom_annotations, self._pose_data = results

        if self._atoms.shape[0] == 0:
            raise RuntimeError('read 0 atoms from PDBQT string')

    @classmethod
    def from_file(cls, pdbqt_filename, name=None, poses_to_read=None,
                  energy_range=None, is_dlg=False, skip_typing=False):
        if name is None:
            name = os.path.splitext(os.path.basename(pdbqt_filename))[0]
        with open(pdbqt_filename) as f:
            pdbqt_string = f.read()
        instance = cls(pdbqt_string, name, poses_to_read, energy_range, is_dlg, skip_typing)
        instance._pdbqt_filename = pdbqt_filename
        return instance

    def __getitem__(self, value):
        if isinstance(value, int):
            if value < 0 or value >= self._positions.shape[0]:
                raise IndexError('The index (%d) is out of range.' % value)
        elif isinstance(value, slice):
            raise TypeError('Slicing is not implemented.')
        else:
            raise TypeError('Invalid argument type.')
        self._current_pose = value
        return self

    def __iter__(self):
        self._current_pose = -1
        return self

    def __next__(self):
        if self._current_pose + 1 >= self._positions.shape[0]:
            raise StopIteration
        self._current_pose += 1
        return self

    def __repr__(self):
        return '<PDBQTMolecule %s: %d poses, %d atoms>' % (
            self._name, self._pose_data['n_poses'], self._atoms.shape[0])

    @property
    def name(self):
        return self._name

    @property
    def pose_id(self):
        return self._current_pose

    @property
    def score(self):
        return self._pose_data['free_energies'][self._current_pose]

    @property
    def n_poses(self):
        return self._pose_data['n_poses']

    @property
    def free_energies(self):
        return self._pose_data['free_energies']

    def atoms(self, atom_idx=None, only_active=True):
        if atom_idx is not None:
            if not isinstance(atom_idx, (list, tuple, np.ndarray)):
                atom_idx = np.array(atom_idx, dtype=int)
        else:
            atom_idx = np.arange(0, self._atoms.shape[0])
        if only_active:
            active = self._pose_data['active_atoms'][self._current_pose]
            atom_idx = sorted(list(set(atom_idx).intersection(active)))
        a = self._atoms[atom_idx].copy()
        a['xyz'] = self._positions[self._current_pose, atom_idx, :]
        return a

    def positions(self, atom_idx=None, only_active=True):
        return np.atleast_2d(self.atoms(atom_idx, only_active)['xyz'])

    def write_pdbqt_string(self, as_model=True):
        if as_model:
            s = 'MODEL    %5d\n' % (self._current_pose + 1)
            s += self._pose_data['pdbqt_string'][self._current_pose]
            s += 'ENDMDL\n'
            return s
        return self._pose_data['pdbqt_string'][self._current_pose]

    def has_flexible_residues(self):
        return bool(self._atom_annotations['flexible_residue'])


# ══════════════════════════════════════════════════════════════════════════════
#  RDKitMolCreate — convert PDBQT poses to RDKit Mol objects
# ══════════════════════════════════════════════════════════════════════════════

class RDKitMolCreate:

    ambiguous_flexres_choices = {
        'HIS': ['HIE', 'HID', 'HIP'],
        'ASP': ['ASP', 'ASH'],
        'GLU': ['GLU', 'GLH'],
        'CYS': ['CYS', 'CYM'],
        'LYS': ['LYS', 'LYN'],
        'ARG': ['ARG', 'ARG_mgltools'],
        'ASN': ['ASN', 'ASN_mgltools'],
        'GLN': ['GLN', 'GLN_mgltools'],
    }

    flexres = {
        'CYS':  {'smiles': 'CCS',  'atom_names_in_smiles_order': ['CA', 'CB', 'SG'], 'h_to_parent_index': {'HG': 2}},
        'CYM':  {'smiles': 'CC[S-]', 'atom_names_in_smiles_order': ['CA', 'CB', 'SG'], 'h_to_parent_index': {}},
        'ASP':  {'smiles': 'CCC(=O)[O-]', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'OD1', 'OD2'], 'h_to_parent_index': {}},
        'ASH':  {'smiles': 'CCC(=O)O', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'OD1', 'OD2'], 'h_to_parent_index': {'HD2': 4}},
        'GLU':  {'smiles': 'CCCC(=O)[O-]', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD', 'OE1', 'OE2'], 'h_to_parent_index': {}},
        'GLH':  {'smiles': 'CCCC(=O)O', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD', 'OE1', 'OE2'], 'h_to_parent_index': {'HE2': 5}},
        'PHE':  {'smiles': 'CCc1ccccc1', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'], 'h_to_parent_index': {}},
        'HIE':  {'smiles': 'CCc1c[nH]cn1', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD2', 'NE2', 'CE1', 'ND1'], 'h_to_parent_index': {'HE2': 4}},
        'HID':  {'smiles': 'CCc1cnc[nH]1', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD2', 'NE2', 'CE1', 'ND1'], 'h_to_parent_index': {'HD1': 6}},
        'HIP':  {'smiles': 'CCc1c[nH+]c[nH]1', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD2', 'NE2', 'CE1', 'ND1'], 'h_to_parent_index': {'HE2': 4, 'HD1': 6}},
        'ILE':  {'smiles': 'CC(C)CC', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG2', 'CG1', 'CD1'], 'h_to_parent_index': {}},
        'LYS':  {'smiles': 'CCCCC[NH3+]', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD', 'CE', 'NZ'], 'h_to_parent_index': {'HZ1': 5, 'HZ2': 5, 'HZ3': 5}},
        'LYN':  {'smiles': 'CCCCCN', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD', 'CE', 'NZ'], 'h_to_parent_index': {'HZ2': 5, 'HZ3': 5}},
        'LEU':  {'smiles': 'CCC(C)C', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD1', 'CD2'], 'h_to_parent_index': {}},
        'MET':  {'smiles': 'CCCSC', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'SD', 'CE'], 'h_to_parent_index': {}},
        'ASN':  {'smiles': 'CCC(=O)N', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'OD1', 'ND2'], 'h_to_parent_index': {'HD21': 4, 'HD22': 4}},
        'ASN_mgltools': {'smiles': 'CCC(=O)N', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'OD1', 'ND2'], 'h_to_parent_index': {'1HD2': 4, '2HD2': 4}},
        'GLN':  {'smiles': 'CCCC(=O)N', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD', 'OE1', 'NE2'], 'h_to_parent_index': {'HE21': 5, 'HE22': 5}},
        'GLN_mgltools': {'smiles': 'CCCC(=O)N', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD', 'OE1', 'NE2'], 'h_to_parent_index': {'1HE2': 5, '2HE2': 5}},
        'ARG':  {'smiles': 'CCCCNC(N)=[NH2+]', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'], 'h_to_parent_index': {'HE': 4, 'HH11': 6, 'HH12': 6, 'HH21': 7, 'HH22': 7}},
        'ARG_mgltools': {'smiles': 'CCCCNC(N)=[NH2+]', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'], 'h_to_parent_index': {'HE': 4, '1HH1': 6, '2HH1': 6, '1HH2': 7, '2HH2': 7}},
        'SER':  {'smiles': 'CCO', 'atom_names_in_smiles_order': ['CA', 'CB', 'OG'], 'h_to_parent_index': {'HG': 2}},
        'THR':  {'smiles': 'CC(C)O', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG2', 'OG1'], 'h_to_parent_index': {'HG1': 3}},
        'VAL':  {'smiles': 'CC(C)C', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG1', 'CG2'], 'h_to_parent_index': {}},
        'TRP':  {'smiles': 'CCc1c[nH]c2c1cccc2', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD1', 'NE1', 'CE2', 'CD2', 'CE3', 'CZ3', 'CH2', 'CZ2'], 'h_to_parent_index': {'HE1': 4}},
        'TYR':  {'smiles': 'CCc1ccc(cc1)O', 'atom_names_in_smiles_order': ['CA', 'CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2', 'OH'], 'h_to_parent_index': {'HH': 8}},
    }

    @classmethod
    def from_pdbqt_mol(cls, pdbqt_mol, only_cluster_leads=False, add_Hs=True):
        if only_cluster_leads and not pdbqt_mol._pose_data['cluster_leads_sorted']:
            raise RuntimeError('no cluster_leads but only_cluster_leads=True')
        mol_list = []
        for mol_index in pdbqt_mol._atom_annotations['mol_index']:
            smiles = pdbqt_mol._pose_data['smiles'][mol_index]
            index_map = pdbqt_mol._pose_data['smiles_index_map'][mol_index]
            h_parent = pdbqt_mol._pose_data['smiles_h_parent'][mol_index]
            atom_idx = pdbqt_mol._atom_annotations['mol_index'][mol_index]

            if smiles is None:
                residue_names = set()
                atom_names = []
                for atom in pdbqt_mol.atoms(atom_idx):
                    residue_names.add(atom[4])
                    atom_names.append(atom[2])
                if len(residue_names) == 1:
                    resname = residue_names.pop()
                    smiles, index_map, h_parent = cls.guess_flexres_smiles(resname, atom_names)
                    if smiles is None:
                        mol_list.append(None)
                        continue

            if only_cluster_leads:
                pose_ids = pdbqt_mol._pose_data['cluster_leads_sorted']
            else:
                pose_ids = range(pdbqt_mol._pose_data['n_poses'])

            mol = Chem.MolFromSmiles(smiles)
            coords_all = []
            for pose_i in pose_ids:
                pdbqt_mol._current_pose = pose_i
                coords = pdbqt_mol.positions(atom_idx)
                mol = cls.add_pose_to_mol(mol, coords, index_map)
                coords_all.append(coords)

            if add_Hs:
                mol = cls.add_hydrogens(mol, coords_all, h_parent)
            mol_list.append(mol)
        return mol_list

    @classmethod
    def guess_flexres_smiles(cls, resname, atom_names):
        if len(set(atom_names)) != len(atom_names):
            return None, None, None
        candidates = cls.ambiguous_flexres_choices.get(resname, [resname])
        is_match = False
        for rn in candidates:
            if rn not in cls.flexres:
                continue
            names_order = cls.flexres[rn]['atom_names_in_smiles_order']
            h_parent = cls.flexres[rn]['h_to_parent_index']
            expected = names_order + list(h_parent.keys())
            if len(atom_names) != len(expected):
                continue
            if sum(int(n in atom_names) for n in expected) == len(expected):
                is_match = True
                break
        if not is_match:
            return None, None, None
        smiles = cls.flexres[rn]['smiles']
        index_map = []
        for si, name in enumerate(names_order):
            index_map.append(si + 1)
            index_map.append(atom_names.index(name) + 1)
        h_out = []
        for name, si in h_parent.items():
            h_out.append(si + 1)
            h_out.append(atom_names.index(name) + 1)
        return smiles, index_map, h_out

    @classmethod
    def add_pose_to_mol(cls, mol, ligand_coordinates, index_map):
        n_atoms = mol.GetNumAtoms()
        n_mappings = len(index_map) // 2
        conf = Chem.Conformer(n_atoms)
        if n_atoms < n_mappings:
            raise RuntimeError(
                'Given %d atom coordinates but index_map has %d atoms.' % (n_atoms, n_mappings))
        coord_is_set = [False] * n_atoms
        for i in range(n_mappings):
            pdbqt_idx = int(index_map[i * 2 + 1]) - 1
            mol_idx = int(index_map[i * 2]) - 1
            x, y, z = [float(c) for c in ligand_coordinates[pdbqt_idx]]
            conf.SetAtomPosition(mol_idx, Point3D(x, y, z))
            coord_is_set[mol_idx] = True
        mol.AddConformer(conf, assignId=True)
        e_mol = Chem.RWMol(mol)
        for i, is_set in reversed(list(enumerate(coord_is_set))):
            if not is_set:
                e_mol.RemoveAtom(i)
        mol = e_mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol

    @staticmethod
    def add_hydrogens(mol, coordinates_list, h_parent):
        mol = Chem.AddHs(mol, addCoords=True)
        conformers = list(mol.GetConformers())
        n_h = len(h_parent) // 2
        for conf_idx, coords in enumerate(coordinates_list):
            conf = conformers[conf_idx]
            used_h = []
            for i in range(n_h):
                parent_idx = h_parent[2 * i] - 1
                h_pdbqt_idx = h_parent[2 * i + 1] - 1
                x, y, z = [float(c) for c in coords[h_pdbqt_idx]]
                parent_atom = mol.GetAtomWithIdx(parent_idx)
                candidates = [
                    a.GetIdx() for a in parent_atom.GetNeighbors()
                    if a.GetAtomicNum() == 1
                ]
                h_rdkit_idx = None
                for h_rdkit_idx in candidates:
                    if h_rdkit_idx not in used_h:
                        break
                used_h.append(h_rdkit_idx)
                if h_rdkit_idx is not None:
                    conf.SetAtomPosition(h_rdkit_idx, Point3D(x, y, z))
        return mol

    @staticmethod
    def combine_rdkit_mols(mol_list):
        combined = None
        for mol in mol_list:
            if mol is None:
                continue
            combined = mol if combined is None else Chem.CombineMols(combined, mol)
        return combined

    @classmethod
    def write_sd_string(cls, pdbqt_mol, only_cluster_leads=False):
        from io import StringIO
        sio = StringIO()
        f = Chem.SDWriter(sio)
        mol_list = cls.from_pdbqt_mol(pdbqt_mol, only_cluster_leads)
        failures = [i for i, mol in enumerate(mol_list) if mol is None]
        combined = cls.combine_rdkit_mols(mol_list)
        if combined is None:
            return '', failures

        property_names = {
            'free_energy': 'free_energies',
            'intermolecular_energy': 'intermolecular_energies',
            'internal_energy': 'internal_energies',
            'cluster_size': 'cluster_size',
            'cluster_id': 'cluster_id',
            'rank_in_cluster': 'rank_in_cluster',
        }
        if only_cluster_leads:
            nr_poses = len(pdbqt_mol._pose_data['cluster_leads_sorted'])
            pose_idxs = pdbqt_mol._pose_data['cluster_leads_sorted']
        else:
            nr_poses = pdbqt_mol._pose_data['n_poses']
            pose_idxs = list(range(nr_poses))

        nr_conformers = combined.GetNumConformers()
        props = {}
        for prop_sdf, prop_pdbqt in property_names.items():
            if nr_conformers == nr_poses:
                props[prop_sdf] = prop_pdbqt

        has_all = all(
            len(pdbqt_mol._pose_data[v]) == nr_conformers
            for _, v in props.items()
        )
        for conformer in combined.GetConformers():
            i = conformer.GetId()
            j = pose_idxs[i]
            if has_all and props:
                data = {k: pdbqt_mol._pose_data[v][j] for k, v in props.items()}
                combined.SetProp('meeko', json.dumps(data))
            f.write(combined, i)
        f.close()
        return sio.getvalue(), failures
