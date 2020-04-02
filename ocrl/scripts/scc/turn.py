"""
    Simple Continuous Curvature Path Library

    Copyright (C) 2017, Gerd Gruenert

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import math

import numpy as np
import scipy.special
from property_manager import cached_property

from scc import State
from scc import TurnParams


class Turn(object):
    """This generates a turn that is always starting from angle 0, turning to an angle _delta"""

    def __init__(self, _params, _delta):
        """
        Parameters
        ----------
            _params : TurnParams
            _delta : float
                Deflection of th Turn; positive = left turn, negative = right turn
        """

        self.params = _params
        self.delta = math.fabs(_delta)
        self.dir = 1
        if _delta < 0:
            self.dir = -1

        assert(self.delta > 0)
        assert (self.delta < 2 * math.pi)

    def state(self, s):
        """
        Returns
        -------
            State
                includes :
                - position x,
                - position y,
                - angle theta
                - curvature kappa

        Parameters
        ----------
            s : numpy.array
                the 1-d position along the turn path

        """

        x = np.empty(len(s))
        y = np.empty(len(s))
        theta = np.empty(len(s))
        kappa = np.empty(len(s))

        if self.delta >= self.params.delta_min:
            first_clotho_cond = s < self.params.len_clothoid_part
            circ_seg_cond = (s > self.params.len_clothoid_part) & \
                            (s < self.params.len_clothoid_part + self.len_of_circular_part)
            second_clotho_cond = (s > self.params.len_clothoid_part + self.len_of_circular_part)

            first_clotho = self._state_clothoid_first(s[first_clotho_cond])
            x[first_clotho_cond] = first_clotho.x
            y[first_clotho_cond] = first_clotho.y
            theta[first_clotho_cond] = first_clotho.theta
            kappa[first_clotho_cond] = first_clotho.kappa

            circ_seg = self._state_circular(s[circ_seg_cond])
            x[circ_seg_cond] = circ_seg.x
            y[circ_seg_cond] = circ_seg.y
            theta[circ_seg_cond] = circ_seg.theta
            kappa[circ_seg_cond] = circ_seg.kappa

            second_clotho = self._state_clothoid_second(s[second_clotho_cond])
            x[second_clotho_cond] = second_clotho.x
            y[second_clotho_cond] = second_clotho.y
            theta[second_clotho_cond] = second_clotho.theta
            kappa[second_clotho_cond] = second_clotho.kappa

        else:  # delta < delta_min:
            first_smalldelta_cond = s < self._len_clothoid_part_smalldelta
            second_smalldelta_cond = ~ first_smalldelta_cond

            first_clotho = self._state_clothoid_smalldelta_first(s[first_smalldelta_cond])
            x[first_smalldelta_cond] = first_clotho.x
            y[first_smalldelta_cond] = first_clotho.y
            theta[first_smalldelta_cond] = first_clotho.theta
            kappa[first_smalldelta_cond] = first_clotho.kappa

            second_clotho = self._state_clothoid_smalldelta_second(s[second_smalldelta_cond])
            x[second_smalldelta_cond] = second_clotho.x
            y[second_smalldelta_cond] = second_clotho.y
            theta[second_smalldelta_cond] = second_clotho.theta
            kappa[second_smalldelta_cond] = second_clotho.kappa

        theta *= self.dir
        theta %= 2*math.pi  # show theta angle between 0 and 2*pi

        return State(x, self.dir * y, theta, self.dir * kappa)

    def _state_circular(self, s):
        """

        Parameters
        ----------
        s : numpy.array

        Returns
        -------

        """
        assert (self.delta >= self.params.delta_min)

        assert ((s >= self.params.len_clothoid_part).all())
        assert ((s <= self.params.len_clothoid_part + self.len_of_circular_part).all())

        angular_segment = self.delta - self.params.delta_min
        start_angle = self.params.delta_min / 2.0
        angles = start_angle + (s - self.params.len_clothoid_part) / self.len_of_circular_part * angular_segment
        x = self.params.omega[0] + self.params.inner_rad * np.sin(angles)
        y = self.params.omega[1] - self.params.inner_rad * np.cos(angles)
        kappa = np.full([len(s)], self.params.kappa_max)

        return State(x, y, angles, kappa)

    def _state_clothoid_first(self, s):
        """ left turn, first clothoid segment

        Parameters
        ----------
        s : numpy.array
        """
        assert((s >= 0).all())
        assert((s <= self.params.len_clothoid_part).all())

        scale = math.sqrt(math.pi / self.params.sigma_max)

        # s goes from 0 .. self.params.len_clothoid_part = 0 .. kappa_max / sigma_max
        ssa_csa = scipy.special.fresnel(s * math.sqrt(self.params.sigma_max / math.pi))

        # theta changes quadratically with s
        theta = 0.5 * self.params.sigma_max * (s*s)

        # curvature changes linear until reaching kappa_max for s=len_clothoid_part
        kappa = s/self.params.len_clothoid_part * self.params.kappa_max

        return State(scale*ssa_csa[1], scale*ssa_csa[0], theta, kappa)

    def _state_clothoid_second(self, s):
        """
        Parameters
        ----------
        s : numpy.array
        """
        assert ((s >= self.params.len_clothoid_part + self.len_of_circular_part).all())
        assert ((s <= 2*self.params.len_clothoid_part + self.len_of_circular_part).all())

        scale = math.sqrt(math.pi / self.params.sigma_max)

        # run from len_clothoid_part to 0
        # NOTE: Mathematically, these brackets are nonsense, but in the numpy.array sense, they are absolutely necessary
        inv_clothoid_s = (2 * self.params.len_clothoid_part + self.len_of_circular_part) - s

        ssa_csa = scipy.special.fresnel(inv_clothoid_s * math.sqrt(self.params.sigma_max / math.pi))

        # theta changes quadratically with s
        theta = - 0.5 * (inv_clothoid_s * inv_clothoid_s) * self.params.sigma_max

        # curvature changes linear until reaching kappa_max for s=len_clothoid_part
        kappa = inv_clothoid_s / self.params.len_clothoid_part * self.params.kappa_max

        state = State(-scale * ssa_csa[1], scale * ssa_csa[0], theta, kappa)
        state = state.rotate_then_translate(self.delta, self._state_qg.x,  self._state_qg.y)
        return state

    @cached_property
    def _sigma_smalldelta(self):
        """The sharpness of a small-delta turn, implicitly defining its kappa_min"""
        fr_sc = scipy.special.fresnel(math.sqrt(self.delta / math.pi))

        nomi = math.pi * (math.cos(self.delta/2)*fr_sc[1] +
                          math.sin(self.delta/2)*fr_sc[0])**2

        # one of the paper uses delta/2+gamma, one uses delta/2-gamma.
        denom = self.params.outer_rad**2 * math.sin(self.delta/2 + self.params.gamma)**2

        return nomi / denom

    @cached_property
    def _len_clothoid_part_smalldelta(self):
        """Length of a single smalldelta clothoid - two together make the smalldelta turn!"""
        return math.sqrt(self.delta/self._sigma_smalldelta)

    def _state_clothoid_smalldelta_first(self, s):
        """

        Parameters
        ----------
        s : numpy.array

        Returns
        -------

        """
        assert(self.delta <= self.params.delta_min)
        assert (self.delta >= 0)
        assert((s >= 0).all())
        assert((s <= self._len_clothoid_part_smalldelta).all())

        scale = math.sqrt(math.pi / self._sigma_smalldelta)
        ssa_csa = scipy.special.fresnel(s * math.sqrt(self._sigma_smalldelta / math.pi))
        theta = (s * s) * self._sigma_smalldelta * 0.5
        kappa = s * self._sigma_smalldelta

        return State(scale * ssa_csa[1], scale * ssa_csa[0], theta, kappa)

    def _state_clothoid_smalldelta_second(self, s):
        """

        Parameters
        ----------
        s : numpy.array

        Returns
        -------

        """
        assert(self.delta <= self.params.delta_min)
        assert (self.delta >= 0)
        assert((s >= self._len_clothoid_part_smalldelta).all())
        assert((s <= self._len_clothoid_part_smalldelta * 2).all())

        # original segments: s \in lp .. 2*lp
        # transform s: for the second clotho part go from lp to 0
        my_s = 2*self._len_clothoid_part_smalldelta - s

        scale = math.sqrt(math.pi / self._sigma_smalldelta)

        ssa_csa = scipy.special.fresnel(my_s * math.sqrt(self._sigma_smalldelta / math.pi))

        # a right-curve now:
        theta = -(my_s * my_s) * self._sigma_smalldelta * 0.5

        kappa = my_s * self._sigma_smalldelta

        st = State(-scale * ssa_csa[1], scale * ssa_csa[0], theta, kappa)
        st = st.rotate_then_translate(self.delta, self._state_qg.x, self._state_qg.y)
        return st

    @cached_property
    def len(self):
        if self.delta >= self.params.delta_min:
            return self.len_of_circular_part + 2*self.params.len_clothoid_part
        else:
            return 2*self._len_clothoid_part_smalldelta

    @cached_property
    def len_of_circular_part(self):
        # circumference = 2*pi*r
        # angular_fraction = 2*pi / (delta - delta_min)
        angular_fraction = (self.delta - self.params.delta_min) / (2*math.pi)
        if angular_fraction < 0:
            return 0

        return 2 * math.pi * self.params.inner_rad * angular_fraction

    @cached_property
    def state_qi(self):
        """Where the first clothoid intersects the inner circle (l/r-turn)"""
        st = self.params.state_qi
        return State(st.x, st.y*self.dir, st.theta*self.dir, st.kappa*self.dir)

    @cached_property
    def _state_qj(self):
        """Where the inner circle segment intersects the second clothoid (left-turn)"""

        ang = self.delta - self.params.delta_min/2
        x = self.params.omega[0] + self.params.inner_rad * np.sin(ang)
        y = self.params.omega[1] - self.params.inner_rad * np.cos(ang)
        kappa = self.params.kappa_max

        return State(x, y, ang, kappa)

    @cached_property
    def state_qj(self):
        """Where the inner circle segment intersects the second clothoid (l/r-turn)"""
        st = self._state_qj
        return State(st.x, st.y*self.dir, st.theta*self.dir, st.kappa*self.dir)

    @cached_property
    def state_qg(self):
        """The end of the second clothoid (turn left | right for dir=1 | -1)"""
        st = self._state_qg
        return State(st.x, st.y*self.dir, st.theta*self.dir, st.kappa*self.dir)

    @cached_property
    def _state_qg(self):
        """The end of the second clothoid (left-turn)"""

        st = State(-self.params.omega[0], -self.params.omega[1], self.delta, 0)
        st = st.rotate_then_translate(self.delta + 2 * self.params.gamma, self.params.omega[0], self.params.omega[1])

        return State(st.x, st.y, self.delta, 0)
