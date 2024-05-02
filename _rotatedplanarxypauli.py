from qecsim.models.rotatedplanar import RotatedPlanarPauli


class RotatedPlanarXYPauli(RotatedPlanarPauli):
    """
    Defines a Pauli operator on a rotated planar XZ/ZX lattice with Hadamard Y applied on the top left diagonal.
    Notes:
    * This is a utility class used by rotated planar implementations of the core models.
    * It is typically instantiated using :meth:`qecsim.models.rotatedplanarxz.RotatedPlanarXZCode.new_pauli`?
    Use cases:
    * Construct a rotated planar XZY Pauli operator by applying site, plaquette and logical operators:
      :meth:`site`, :meth:`plaquette`, :meth:`logical_x`, :meth:`logical_z`.
    * Get the single Pauli operator applied to a given site: :meth:`operator`
    * Convert to binary symplectic form: :meth:`to_bsf`.
    * Copy a rotated planar XZ Pauli operator: :meth:`copy`.
    """

    def plaquette(self, index):
        """
        Apply a plaquette operator at the given index.
        Notes:
        * Index is in the format (x, y).
        * Z operators are applied to SW and NE qubits. X operators are applied to NW and SE qubits. Z operators acting on 
        qubits along the upper-left to lower right diagonal are switched to Y.
        * Applying plaquette operators on plaquettes that lie outside the lattice have no effect on the lattice.
        :param index: Index identifying the plaquette in the format (x, y).
        :type index: 2-tuple of int
        :return: self (to allow chaining)
        :rtype: RotatedPlanarXZYPauli
        """
        x, y = index
        # apply if index within lattice
        if self.code.is_in_plaquette_bounds(index):
            #Plaquettes one up from diagonal have Y SW
            if (y + x) % 2 == 0:
                self.site('Y', (x, y)) #SW
                self.site('Y', (x, y + 1))  # NW
                self.site('Y', (x + 1, y + 1))  # NE
                self.site('Y', (x + 1, y))  # SE
            #Plaquettes one down from diagonal have Y NE
            else:
                self.site('X', (x, y))  # SW
                self.site('X', (x, y + 1))  # NW
                self.site('X', (x + 1, y + 1))  # NE
                self.site('X', (x + 1, y))  # SE
        return self

    def logical_x(self):
        """
        Apply a logical X operator, i.e. alternate X and Z between lower-left and lower-right corners.
        Notes:
        * X unchanged from Hadamard YZ operator
        * Operators are applied to the bottom row to allow optimisation of the MPS decoder.
        :return: self (to allow chaining)
        :rtype: RotatedPlanarXZYPauli
        """
        max_site_x, max_site_y = self.code.site_bounds
        self.site('X', *((x, 0) for x in range(0, max_site_x + 1)))
        return self

    def logical_y(self):
        """
        Apply a logical Z operator, i.e. alternate Z and X between lower-right and upper-right corners.
        Notes:
        * First Z operator applied on lower right qubit gets switched to Y.
        * Operators are applied to the rightmost column to allow optimisation of the MPS decoder.
        :return: self (to allow chaining)
        :rtype: RotatedPlanarXZPauli
        """
        max_site_x, max_site_y = self.code.site_bounds
        self.site('Y', *((max_site_x, y) for y in range(0, max_site_y + 1)))
        return self
    
    def logical_z(self):
        """
        Apply a logical Z operator simply by multiplying logical Y & X together.

        Notes:

        :return: self (to allow chaining)
        :rtype: RotatedPlanarXYPauli
        """
        max_site_x, max_site_y = self.code.site_bounds
        self.site('X', *((x, 0) for x in range(0, max_site_x + 1)))
        self.site('Y', *((max_site_x, y) for y in range(0, max_site_y + 1)))
        return self