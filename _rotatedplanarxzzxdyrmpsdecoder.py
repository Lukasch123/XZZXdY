import functools
import itertools
import json
import logging
import operator

import numpy as np
from mpmath import mp

from qecsim import paulitools as pt, tensortools as tt
from qecsim.model import Decoder, cli_description
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarRMPSDecoder


logger = logging.getLogger(__name__)


@cli_description('Rotated MPS ([chi] INT >=0, [mode] CHAR, ...)')
class RotatedPlanarXZZXdYRMPSDecoder(RotatedPlanarRMPSDecoder):
    r"""
    Implements a rotated planar XZZXdY Rotated Matrix Product State (RMPS) decoder.

    Decoding algorithm:

    * A sample recovery operation :math:`f` is found by applying a path of X operators between each plaquette,
      identified by the syndrome, along a diagonal to an appropriate boundary.
    * The probability of the left coset :math:`fG` of the stabilizer group :math:`G` of the planar code with respect
      to :math:`f` is found by contracting an appropriately defined MPS-based tensor network (see
      https://arxiv.org/abs/1405.4883).
    * Since this is a rotated MPS decoder, the links of the network are rotated 45 degrees by splitting each stabilizer
      node into 4 delta nodes that are absorbed into the neighbouring qubit nodes.
    * The complexity of the algorithm can managed by defining a bond dimension :math:`\chi` to which the MPS bond
      dimension is truncated after each row/column of the tensor network is contracted into the MPS.
    * The probability of cosets :math:`f\bar{X}G`, :math:`f\bar{Y}G` and :math:`f\bar{Z}G` are calculated similarly.
    * The default contraction is column-by-column but can be set using the mode parameter to row-by-row or the average
      of both contractions.
    * A sample recovery operation from the most probable coset is returned.

    Notes:

    * Specifying chi=None gives an exact contract (up to rounding errors) but is exponentially slow in the size of
      the lattice.
    * Modes:

        * mode='c': contract by columns
        * mode='r': contract by rows
        * mode='a': contract by columns and by rows and, for each coset, take the average of the probabilities.

    * Contracting by columns (i.e. truncating vertical links) may give different coset probabilities to contracting by
      rows (i.e. truncating horizontal links). However, the effect is symmetric in that transposing the sample_pauli on
      the lattice and exchanging X and Z single Paulis reverses the difference between X and Z cosets probabilities.

    Tensor network example:

    3x3 rotated planar code with H or V indicating qubits and plaquettes indicating stabilizers:
    ::

           /---\
           |   |
           H---V---H--\
           |   |   |  |
           |   |   |  |
           |   |   |  |
        /--V---H---V--/
        |  |   |   |
        |  |   |   |
        |  |   |   |
        \--H---V---H
               |   |
               \---/


    MPS tensor network as per https://arxiv.org/abs/1405.4883 (s=stabilizer), except H and V qubit tensors are defined
    identically with the NE and SW (NW and SE) stabilizers applying Z (X) operators:
    ::

             s
            / \
           H   V   H
            \ / \ / \
             s   s   s
            / \ / \ /
           V   H   V
          / \ / \ /
         s   s   s
          \ / \ / \
           H   V   H
                \ /
                 s

    Links are rotated by splitting stabilizers and absorbing them into neighbouring qubits.
    For even columns of stabilizers (according to indexing defined in
    :class:`qecsim.models.planar.RotatedPlanarXZCode`), a 'lucky' horseshoe shape is used:
    ::

        H   V      H     V
         \ /        \   /       H V
          s    =>    s s    =>  | |
         / \         | |        V-H
        V   H        s-s
                    /   \
                   V     H

    For odd columns, an 'unlucky' horseshoe shape is used:
    ::

        H   V      H     V
         \ /        \   /       H-V
          s    =>    s-s    =>  | |
         / \         | |        V H
        V   H        s s
                    /   \
                   V     H

    Resultant MPS tensor network, where horizontal (vertical) bonds have dimension 2 (4) respectively.
    ::

          0 1 2
        0 H-V-H
          | | |
        1 V-H-V
          | | |
        2 H-V-H
    """

    @classmethod
    def sample_recovery(cls, code, syndrome):
        """
        Return a sample Pauli consistent with the syndrome, created by applying a path of X operators between each
        plaquette, identified by the syndrome, along a diagonal to an appropriate boundary.

        Since X operators are unchanged from Hadamard YZ operations, same sample recovery as for XZZX-code.

        :param code: Rotated planar XZZXdY code.
        :type code: RotatedPlanarXZZXdYCode
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :return: Sample recovery operation as rotated planar pauli.
        :rtype: RotatedPlanarXZZXdYPauli
        """
        # prepare sample
        sample_recovery = code.new_pauli()
        # ask code for syndrome plaquette_indices
        plaquette_indices = code.syndrome_to_plaquette_indices(syndrome)
        # for each plaquette
        max_site_x, max_site_y = code.site_bounds
        for plaq_index in plaquette_indices:
            # NOTE: plaquette index coincides with site on lower left corner
            plaq_x, plaq_y = plaq_index
            # if upper-left even diagonals or lower-right odd diagonals
            if (plaq_x < plaq_y and (plaq_x - plaq_y) % 2 == 0) or (plaq_x > plaq_y and (plaq_x - plaq_y) % 2 == 1):
                # join with X to lower-left boundary
                site_x, site_y = plaq_x, plaq_y
                while site_x >= 0 and site_y >= 0:
                    sample_recovery.site('X', (site_x, site_y))
                    site_x -= 1
                    site_y -= 1
            else:
                # join with X to upper-right boundary
                site_x, site_y = plaq_x + 1, plaq_y + 1
                while site_x <= max_site_x and site_y <= max_site_y:
                    sample_recovery.site('X', (site_x, site_y))
                    site_x += 1
                    site_y += 1
        # return sample
        return sample_recovery

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        params = [('chi', self._chi), ('mode', self._mode), ('tol', self._tol), ]
        return 'Rotated planar XZZXdY RMPS ({})'.format(', '.join('{}={}'.format(k, v) for k, v in params if v))

    class TNC(RotatedPlanarRMPSDecoder.TNC):
        """Tensor network creator"""
        
        def h_node_value(self, prob_dist, f, n, e, s, w):
            """Return horizontal edge tensor element value."""
            paulis = ('I', 'X', 'Y', 'Z')
            op_to_pr = dict(zip(paulis, prob_dist))
            f = pt.pauli_to_bsf(f)
            I, X, Y, Z = pt.pauli_to_bsf(paulis)
            # n, e, s, w are in {0, 1} so multiply op to turn on or off
            op = (f + (n * Z) + (e * X) + (s * Z) + (w * X)) % 2
            return op_to_pr[pt.bsf_to_pauli(op)]
        
        @functools.lru_cache()
        def create_h_node(self, prob_dist, f, compass_direction=None):
            """Return horizontal qubit tensor, i.e. has X plaquettes to left/right and Z plaquettes above/below."""

            def _shape(compass_direction=None):
                """Return shape of tensor including dummy indices."""
                return {  # (ne, se, sw, nw)
                    'n': (2, 2, 2, 1),
                    'ne': (1, 2, 2, 1),
                    'e': (1, 2, 2, 2),
                    'se': (1, 1, 2, 2),
                    's': (2, 1, 2, 2),
                    'sw': (2, 1, 1, 2),
                    'w': (2, 2, 1, 2),
                    'nw': (2, 2, 1, 1),
                }.get(compass_direction, (2, 2, 2, 2))

            # create bare h_node
            node = np.empty(_shape(compass_direction), dtype=np.float64)
            # fill values
            for n, e, s, w in np.ndindex(node.shape):
                node[(n, e, s, w)] = self.h_node_value(prob_dist, f, n, e, s, w)
            return node
        
        @functools.lru_cache()
        def v_node_value(self, prob_dist, f, n, e, s, w):
            """Return V-node qubit tensor element value."""
            # N.B. with XZ/ZX plaquettes, H-node and V-node values are both as per H-node values of the CSS code
            return self.h_node_value(prob_dist, f, n, e, s, w)
        def create_v_node(self, prob_dist, f, compass_direction=None):
            """Return vertical qubit tensor, i.e. has Z plaquettes to left/right and X plaquettes above/below."""

            def _shape(compass_direction=None):
                """Return shape of tensor including dummy indices."""
                return {  # (ne, se, sw, nw)
                    'n': (1, 2, 2, 2),
                    'ne': (1, 1, 2, 2),
                    'e': (2, 1, 2, 2),
                    'se': (2, 1, 1, 2),
                    's': (2, 2, 1, 2),
                    # 'sw': (2, 2, 1, 1),  # cannot happen
                    'w': (2, 2, 2, 1),
                    'nw': (1, 2, 2, 1),
                }.get(compass_direction, (2, 2, 2, 2))

            # create bare v_node
            node = np.empty(_shape(compass_direction), dtype=np.float64)
            # fill values
            for n, e, s, w in np.ndindex(node.shape):
                node[(n, e, s, w)] = self.v_node_value(prob_dist, f, n, e, s, w)
            return node
        
        @functools.lru_cache()
        def d_node_value(self, prob_dist, f, n, e, s, w):
            """Return diagonal tensor element value."""
            paulis = ('I', 'X', 'Y', 'Z')
            op_to_pr = dict(zip(paulis, prob_dist))
            f = pt.pauli_to_bsf(f)
            I, X, Y, Z = pt.pauli_to_bsf(paulis)
            # n, e, s, w are in {0, 1} so multiply op to turn on or off
            op = (f + (n * Y) + (e * X) + (s * Y) + (w * X)) % 2
            return op_to_pr[pt.bsf_to_pauli(op)]
        
        @functools.lru_cache()
        def create_d_node(self, prob_dist, f, compass_direction=None):
            """Return diagonal qubit tensor, i.e. has X plaquettes to left/right and Y plaquettes above/below."""

            def _shape(compass_direction=None):
                """Return shape of tensor including dummy indices."""
                return {  # (ne, se, sw, nw)
                    'n': (2, 2, 2, 1),
                    'ne': (1, 2, 2, 1),
                    'e': (1, 2, 2, 2),
                    'se': (1, 1, 2, 2),
                    's': (2, 1, 2, 2),
                    'sw': (2, 1, 1, 2),
                    'w': (2, 2, 1, 2),
                    'nw': (2, 2, 1, 1),
                }.get(compass_direction, (2, 2, 2, 2))

            # create bare h_node
            node = np.empty(_shape(compass_direction), dtype=np.float64)
            # fill values
            for n, e, s, w in np.ndindex(node.shape):
                node[(n, e, s, w)] = self.d_node_value(prob_dist, f, n, e, s, w)
            return node
        
        @functools.lru_cache()
        def create_s_node(self, compass_direction=None):
            """Return stabilizer tensor."""

            def _shape(compass_direction=None):
                """Return shape of tensor including dummy indices."""
                return {  # (ne, se, sw, nw)
                    'n': (1, 2, 2, 1),
                    'e': (1, 1, 2, 2),
                    's': (2, 1, 1, 2),
                    'w': (2, 2, 1, 1),
                }.get(compass_direction, (2, 2, 2, 2))

            node = tt.tsr.delta(_shape(compass_direction))
            return node
        
        def create_tn(self, prob_dist, sample_pauli):
            """Return a network (numpy.array 2d) of tensors (numpy.array 4d).
            Note: The network contracts to the coset probability of the given sample_pauli.
            """

            def _rotate_q_index(index, code):
                """Convert code site index in format (x, y) to tensor network q-node index in format (r, c)"""
                site_x, site_y = index  # qubit index in (x, y)
                site_r, site_c = code.site_bounds[1] - site_y, site_x  # qubit index in (r, c)
                return code.site_bounds[0] - site_c + site_r, site_r + site_c  # q-node index in (r, c)

            def _rotate_p_index(index, code):
                """Convert code plaquette index in format (x, y) to tensor network s-node index in format (r, c)"""
                q_node_r, q_node_c = _rotate_q_index(index, code)  # q-node index in (r, c)
                return q_node_r - 1, q_node_c  # s-node index in (r, c)

            def _compass_q_direction(index, code):
                """if the code site index lies on border of lattice then give that direction, else empty string."""
                direction = {code.site_bounds[1]: 'n', 0: 's'}.get(index[1], '')
                direction += {0: 'w', code.site_bounds[0]: 'e'}.get(index[0], '')
                return direction

            def _compass_p_direction(index, code):
                """if the code plaquette index lies on border of lattice then give that direction, else empty string."""
                direction = {code.site_bounds[1]: 'n', -1: 's'}.get(index[1], '')
                direction += {-1: 'w', code.site_bounds[0]: 'e'}.get(index[0], '')
                return direction

            # extract code
            code = sample_pauli.code
            # initialise empty tn
            tn_max_r, _ = _rotate_q_index((0, 0), code)
            _, tn_max_c = _rotate_q_index((code.site_bounds[0], 0), code)
            tn = np.empty((tn_max_r + 1, tn_max_c + 1), dtype=object)
            # iterate over
            max_site_x, max_site_y = code.site_bounds
            for code_index in itertools.product(range(-1, max_site_x + 1), range(-1, max_site_y + 1)):
                is_z_plaquette = code.is_z_plaquette(code_index)
                if code.is_in_site_bounds(code_index):
                    q_node_index = _rotate_q_index(code_index, code)
                    q_pauli = sample_pauli.operator(code_index)
                    if is_z_plaquette:
                        x, y = code_index
                        x_prim, y_prim = x, (max_site_y)-y
                        #All nodes along diagonal become d nodes
                        if x_prim == y_prim:
                            q_node = self.create_d_node(prob_dist, q_pauli, _compass_q_direction(code_index, code))
                        else:
                            q_node = self.create_h_node(prob_dist, q_pauli, _compass_q_direction(code_index, code))
                    else:
                        q_node = self.create_v_node(prob_dist, q_pauli, _compass_q_direction(code_index, code))
                    tn[q_node_index] = q_node
                if code.is_in_plaquette_bounds(code_index):
                    s_node_index = _rotate_p_index(code_index, code)
                    s_node = self.create_s_node(_compass_p_direction(code_index, code))
                    tn[s_node_index] = s_node
            return tn

