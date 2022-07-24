#!/usr/bin/env python3

"""
    response_optimization.py

    Contains classes to design the spectrum of mechanical networks
    using direct numerical optimization of the response function

  (c) Henrik Ronellenfitsch 2018
"""

from itertools import chain

import numpy as np
import scipy as sp
import scipy.optimize as optimize
import networkx as nx

from copy import copy

from . import bending

class PointMassOptimizer():
    """ Optimizes a number of band gaps at given mode indices
    for a network with equal point masses and massless springs.
    """
    def __init__(self, network, gaps, qs):
        """ Initialize with given MechanicalNetwork,
        band gap fractions, and wave vectors
        """
        self.network = network
        self.gaps = gaps
        self.gap_inds = self.gaps_from_input(gaps)
        self.qs = qs

        self.Qs = [self.network.equilibrium_matrix_ft(q).toarray() for q in qs]
        self.QHs = [Q.conj().transpose() for Q in self.Qs]

        self.estimate_jac = False

    def gaps_from_input(self, gaps):
        indices = []

        for g in gaps:
            if g < 1:
                indices.append(int(g*self.network.graph['Q'].shape[0]))
            else:
                indices.append(g)

        return np.array(indices)

    def loss_and_jac_gap(self, x, ωsqrs, summed=True):
        """ Spectral loss function for a gap defined by the given
        values of ω^2.
        """
        Ks = [np.dot(Q*x, QH) for Q, QH in zip(self.Qs, self.QHs)]

        Gs = [ωsqr*np.eye(K.shape[0]) - K for K, ωsqr in zip(Ks, ωsqrs)]
        G3s = [G.dot(G.conj().transpose()).dot(G) for G in Gs]
        invs = [np.linalg.inv(G) for G in Gs]

        response = [np.sum(np.abs(inv)**2) for inv in invs]

        grads = [2*np.sum(QH.transpose()*np.linalg.solve(G3, Q), axis=0).real
                 for G, G3, Q, QH in zip(Gs, G3s, self.Qs, self.QHs)]

        if summed:
            return np.sum(response), np.sum(grads, axis=0)
        else:
            return np.array(response), np.array(grads)

    def loss_and_jac(self, x):
        res, grads = zip(*[self.loss_and_jac_gap(x, oms)
                         for oms in self.ωsqrs])

        return np.sum(np.array(res)/self.res_0), np.sum(np.array(grads)/self.res_0[:,np.newaxis], axis=0)


    def num_optim_dof(self):
        return self.network.graph['Q'].shape[1]

    def initialize(self, x0):
        """ Initialize the state for optimization of given initial conditions
        """
        Ks = [np.dot(Q*x0, QH) for Q, QH in zip(self.Qs, self.QHs)]

        self.ωsqrs = [np.array([np.mean(sp.linalg.eigvalsh(K)[gap_ind:gap_ind+2]) for K in Ks])
                        for gap_ind in self.gap_inds]
        self.x0 = x0.copy()

        # normalization factors
        res, grads = zip(*[self.loss_and_jac_gap(x0, oms)
                         for oms in self.ωsqrs])
        self.res_0 = np.array(res)

    def optimize(self, seed=None, lower_bound=0.1, upper_bound=1,
                 method='L-BFGS-B', callback=None, tol=1e-6, options={},
                 x0=None):
        if seed is not None:
            np.random.seed(seed)

        n = self.num_optim_dof()

        if x0 is None:
            x0 = lower_bound + (upper_bound - lower_bound)*np.random.random(n)

        # check which type of loss function we have
        if self.estimate_jac:
            loss_func = self.loss_function
            loss_jac = False
        elif hasattr(self, 'loss_and_jac'):
            loss_func = self.loss_and_jac
            loss_jac = True
        else:
            loss_func = self.loss_function
            loss_jac = self.loss_jac

        self.initialize(x0)

        ret = optimize.minimize(loss_func, x0, jac=loss_jac,
                        method=method,
                        bounds=(n*[(lower_bound, upper_bound)]), tol=tol,
                        callback=callback,
                        options=options)

        return (ret, x0)

class MassPointOptimizer(PointMassOptimizer):
    """ Optimizes a number of band gaps at given mode indices
    for a network with variable point masses and fixed stiffness massless springs.
    """

    def loss_and_jac_gap(self, x, ωsqrs, summed=True):
        """ Spectral loss function for a gap defined by the given
        values of ω^2.
        """
        d = self.network.graph['dimension']
        n = self.network.number_of_nodes()

        Ks = [np.dot(Q, QH) for Q, QH in zip(self.Qs, self.QHs)]
        M = np.diag(np.tile(x, d))

        Gs = [ωsqr*M - K for K, ωsqr in zip(Ks, ωsqrs)]
        # G3s = [G.dot(G.conj().transpose()).dot(G) for G in Gs]
        invs = [np.linalg.inv(G) for G in Gs]

        response = [np.sum(np.abs(inv)**2) for inv in invs]

        # gradient
        diags = [np.diag(Gi.dot(Gi.T.conj()).dot(Gi)) for Gi in invs]

        grads = [-2*ω2*dia.reshape(d, n).sum(axis=0).real
                 for ω2, dia in zip(ωsqrs, diags)]

        if summed:
            return np.sum(response), np.sum(grads, axis=0)
        else:
            return np.array(response), np.array(grads)

    def num_optim_dof(self):
        return self.network.number_of_nodes()

    def initialize(self, x0):
        """ Initialize the state for optimization of given initial conditions
        """
        d = self.network.graph['dimension']
        Ks = [np.dot(Q, QH) for Q, QH in zip(self.Qs, self.QHs)]
        M = np.diag(np.tile(x0, d))

        self.ωsqrs = [np.array([np.mean(sp.linalg.eigvalsh(K, b=M)[gap_ind:gap_ind+2]) for K in Ks])
                        for gap_ind in self.gap_inds]
        self.x0 = x0.copy()

        # normalization factors
        res, grads = zip(*[self.loss_and_jac_gap(x0, oms)
                         for oms in self.ωsqrs])
        self.res_0 = np.array(res)


class PointMassIncreaseOptimizer(PointMassOptimizer):
    """ Increase the response instead of decreasing it.
    """
    def loss_and_jac(self, x):
        res, grads = zip(*[self.loss_and_jac_gap(x, oms)
                         for oms in self.ωsqrs])

        increase = 3

        res_scaled = np.array(res)/self.res_0
        grad_scaled = np.array(grads)/self.res_0[:,np.newaxis]

        res_increase_tot = (res_scaled - increase)

        res_increase = 0.5*np.mean(res_increase_tot**2)
        grad_increase = np.mean(res_increase_tot[:,np.newaxis]*grad_scaled, axis=0)

        return res_increase, grad_increase


class SpringMassOptimizer(PointMassOptimizer):
    """ Like point masses, but instead taking into account spring mass.
    Node and spring masses are fixed but spring constants can be varied.
    """
    def __init__(self, network, gaps, qs, node_masses=None,
                 spring_masses=None):
        super().__init__(network, gaps, qs)

        self.Es = [self.network.incidence_matrix_ft(q).toarray() for q in qs]
        self.EHs = [E.conj().transpose() for E in self.Es]

        self.Fs = [self.network.non_oriented_incidence_matrix_ft(q).toarray() for q in qs]
        self.FHs = [F.conj().transpose() for F in self.Fs]

        if node_masses is None:
            node_masses = np.ones(network.number_of_nodes())

        if spring_masses is None:
            spring_masses = np.ones(network.number_of_edges())

        self.node_masses = node_masses
        self.spring_masses = spring_masses

    def initialize(self, x0):
        """ Initialize the state for optimization of given initial conditions
        """
        Ks = [(Q*x0).dot(QH) for Q, QH in zip(self.Qs, self.QHs)]
        M_node = sp.sparse.diags(self.node_masses)
        Ms = [M_node + np.dot(E*self.spring_masses, EH)/12 + 3*np.dot(F*self.spring_masses, FH)/12
              for E, EH, F, FH in zip(self.Es, self.EHs, self.Fs, self.FHs)]
        Ms = [sp.linalg.block_diag(*(self.network.graph['dimension']*[M])) for M in Ms]


        self.ωsqrs = [np.array([np.mean(sp.linalg.eigvalsh(K, b=M)[gap_ind:gap_ind+2])
                                for K, M in zip(Ks, Ms)])
                        for gap_ind in self.gap_inds]
        self.x0 = x0.copy()

        # normalization factors
        res, grads = zip(*[self.loss_and_jac_gap(x0, oms)
                         for oms in self.ωsqrs])
        self.res_0 = np.array(res)

        return [np.array([sp.linalg.eigvalsh(D, b=M) for M, D in zip(Ms, Ks)])
                        for gap_ind in self.gap_inds]

    def loss_and_jac_gap(self, x, ωsqrs):
        """ Spectral loss function for a gap defined by the given
        values of ω^2.
        """
        d = self.network.graph['dimension']
        n = self.network.number_of_nodes()

        Ks = [np.dot(Q*x, QH) for Q, QH in zip(self.Qs, self.QHs)]

        M_node = sp.sparse.diags(self.node_masses)
        Ms = [M_node + np.dot(E*self.spring_masses, EH)/12 + 3*np.dot(F*self.spring_masses, FH)/12
              for E, EH, F, FH in zip(self.Es, self.EHs, self.Fs, self.FHs)]
        Ms = [sp.linalg.block_diag(*(d*[M])) for M in Ms]

        Gs = [ωsqr*M - K for K, M, ωsqr in zip(Ks, Ms, ωsqrs)]
        invs = [np.linalg.inv(G) for G in Gs]
        Gi3s = [np.dot(Gi, Gi.T.conj()).dot(Gi) for Gi in invs]

        response = np.sum([np.abs(inv)**2 for inv in invs])

        Q_grads = np.array([2*np.sum(QH.transpose()*np.dot(Gi3, Q), axis=0).real
                 for Gi3, Q, QH in zip(Gi3s, self.Qs, self.QHs)])

        # E_grads =  np.array([-2*ωsqr*np.sum(EH.transpose()*np.dot(Gi3[i*n:(i+1)*n,i*n:(i+1)*n], E)/12, axis=0).real
        #          for Gi3, E, EH, ωsqr in zip(Gi3s, self.Es, self.EHs, ωsqrs) for i in range(d)])
        #
        # F_grads =  np.array([-6*ωsqr*np.sum(FH.transpose()*np.dot(Gi3[i*n:(i+1)*n,i*n:(i+1)*n], F)/12, axis=0).real
        #          for Gi3, F, FH, ωsqr in zip(Gi3s, self.Fs, self.FHs, ωsqrs) for i in range(d)])

        return response, np.sum(Q_grads, axis=0) #+ np.sum(E_grads + F_grads, axis=0)

class BeamNetworkOptimizer(PointMassOptimizer):
    """ Optimizer for networks of slender beams which are able to bend
    and stretch, according to Kirchhoff's beam laws.
    The optimizatin variables are not the stiffnesses but instead the
    squared beam radii.
    The "equation of state" relating bending and stretching spring constants
    is then k_bend = r^2 k_stretch and k_stretch = r^2.
    Similarly, the mass of each rod is given by m = r^2.
    Fixed node masses may be specified as well.

    The network will be automatically refined.
    """
    def __init__(self, network, gaps, qs):
        refined = bending.refined_network(network)

        super().__init__(refined, gaps, qs)

        self.refined_network = refined
        self.original_network = network

        # FT of the mass matrices
        self.Es = [self.network.incidence_matrix_ft(q).toarray() for q in qs]
        self.EHs = [E.conj().transpose() for E in self.Es]

        self.Fs = [self.network.non_oriented_incidence_matrix_ft(q).toarray() for q in qs]
        self.FHs = [F.conj().transpose() for F in self.Fs]

        # FT of the bending equilibrium matrices
        D_ijs, D_is, degrees, einds = zip(*[bending.bending_equilibrium_matrix2(refined, q=q) for q in qs])

        self.D_is = D_is
        self.D_iHs = [D_i.conj().transpose() for D_i in self.D_is]

        self.D_ijs = D_ijs
        self.D_ijHs = [D_ij.conj().transpose() for D_ij in self.D_ijs]

        self.degrees = degrees[0]
        self.incident_edge_indices = list(chain(einds))

        self.estimate_jac = True

    def num_optim_dof(self):
        """ DOFs are edge radii squared in the original network.
        """
        return self.original_network.graph['Q'].shape[1]

    def initialize(self, x0):
        """ Initialize the state for optimization of given initial conditions
        """
        Ms, Ds = self.mass_and_dynamical_matrices(x0)

        self.ωsqrs = [np.array([np.mean(sp.linalg.eigvalsh(D, b=M)[gap_ind:gap_ind+2])
                                for M, D in zip(Ms, Ds)])
                        for gap_ind in self.gap_inds]
        self.x0 = x0.copy()

        # normalization factors
        res = [self.loss_gap(x0, oms) for oms in self.ωsqrs]
        self.res_0 = np.array(res)

        print(np.sqrt(self.ωsqrs))

        return [np.array([sp.linalg.eigvalsh(D, b=M) for M, D in zip(Ms, Ds)])
                        for gap_ind in self.gap_inds]

    def elastic_constants(self, x):
        old_nodes, mid_nodes, new_edges = self.refined_network.graph['old_node_inds'], \
            self.refined_network.graph['midpoint_node_inds'], \
            self.refined_network.graph['new_edge_inds']

        kappas = np.zeros(self.refined_network.number_of_nodes())

        kappas[old_nodes] = 0.01 # hinge_stiffness
        kappas[mid_nodes] = x/100 # bond bending

        kappas_nodal = np.repeat(kappas, self.degrees)

        Kappas = sp.sparse.diags(kappas/self.degrees)
        Kappas_nodal = sp.sparse.diags(kappas_nodal)

        k_stretch = np.ones(self.refined_network.number_of_edges())
        K_stretch = sp.sparse.diags(k_stretch)

        mass = np.ones_like(k_stretch)

        return Kappas, Kappas_nodal, K_stretch, mass

    def mass_and_dynamical_matrices(self, x):
        """ Return mass matrix and dynamical matrix
        for given rsqr.
        """
        d = self.refined_network.graph['dimension']

        Kappas, Kappas_nodal, K_stretch, mass = self.elastic_constants(x)

        Ms = []
        Ds = []
        for E, EH, F, FH, D_i, D_iH, D_ij, D_ijH, Q, QH in zip(self.Es, self.EHs,
                                                    self.Fs, self.FHs,
                                                    self.D_is, self.D_iHs,
                                                    self.D_ijs, self.D_ijHs,
                                                    self.Qs, self.QHs):
            D_bend = D_ij.dot(Kappas_nodal).dot(D_ijH)
            D_bend = D_bend - D_i.dot(Kappas).dot(D_iH)

            D_stretch = Q.dot(K_stretch.dot(QH))

            Ds.append(D_bend + D_stretch)

            # Mass matrix
            M = (E*mass).dot(EH)/12 + (F*mass).dot(FH)/4
            M = sp.linalg.block_diag(M, M)
            Ms.append(M)

        return Ms, Ds

    def loss_gap(self, x, ωsqrs):
        """ Spectral loss function for a gap defined by the given
        values of ω^2.
        """
        n = self.refined_network.number_of_nodes()

        Ms, Ds = self.mass_and_dynamical_matrices(x)

        # compute response function
        Gs = [ωsqr*M - D for D, M, ωsqr in zip(Ds, Ms, ωsqrs)]
        invs = [np.linalg.inv(G) for G in Gs]
        # Gi3s = [np.dot(Gi, Gi.T.conj()).dot(Gi) for Gi in invs]

        response = np.sum([np.abs(inv)**2 for inv in invs])

        # Q_grads = np.array([2*np.sum(QH.transpose()*np.dot(Gi3, Q), axis=0).real
        #          for Gi3, Q, QH in zip(Gi3s, self.Qs, self.QHs)])

        # E_grads =  np.array([-2*ωsqr*np.sum(EH.transpose()*np.dot(Gi3[i*n:(i+1)*n,i*n:(i+1)*n], E)/12, axis=0).real
        #          for Gi3, E, EH, ωsqr in zip(Gi3s, self.Es, self.EHs, ωsqrs) for i in range(d)])
        #
        # F_grads =  np.array([-6*ωsqr*np.sum(FH.transpose()*np.dot(Gi3[i*n:(i+1)*n,i*n:(i+1)*n], F)/12, axis=0).real
        #          for Gi3, F, FH, ωsqr in zip(Gi3s, self.Fs, self.FHs, ωsqrs) for i in range(d)])

        return response#, np.sum(Q_grads, axis=0) #+ np.sum(E_grads + F_grads, axis=0)

    def loss_function(self, x):
        res = [self.loss_gap(x, oms) for oms in self.ωsqrs]

        return np.sum(np.array(res)/self.res_0)

class RealSpringNetworkOptimizer(BeamNetworkOptimizer):
    def elastic_constants(self, x):
        old_nodes, mid_nodes, new_edges = self.refined_network.graph['old_node_inds'], \
            self.refined_network.graph['midpoint_node_inds'], \
            self.refined_network.graph['new_edge_inds']

        kappas = np.zeros(self.refined_network.number_of_nodes())

        kappas[old_nodes] = 0.6 # hinge_stiffness
        kappas[mid_nodes] = x # bond bending

        kappas_nodal = np.repeat(kappas, self.degrees)

        Kappas = sp.sparse.diags(kappas/self.degrees)
        Kappas_nodal = sp.sparse.diags(kappas_nodal)

        # linear interpolation
        # stretch = (17 - 27)*(x - 0.05)/(0.2 - 0.05) + 27.0
        stretch = 20*np.ones_like(x)
        k_stretch = np.repeat(stretch, 2)
        K_stretch = sp.sparse.diags(k_stretch)

        mass = 2.8*np.ones_like(x)
        mass = np.repeat(mass, 2)

        return Kappas, Kappas_nodal, K_stretch, mass

class UniformNetworkOptimizer(BeamNetworkOptimizer):
    def elastic_constants(self, x):
        old_nodes, mid_nodes, new_edges = self.refined_network.graph['old_node_inds'], \
            self.refined_network.graph['midpoint_node_inds'], \
            self.refined_network.graph['new_edge_inds']

        hinge_stiffness = x[0]
        bending_modulus = x[1]
        mass = x[2]
        stretching_modulus = 20

        kappas = np.zeros(self.refined_network.number_of_nodes())

        kappas[old_nodes] = hinge_stiffness # hinge_stiffness
        kappas[mid_nodes] = bending_modulus # bond bending

        kappas_nodal = np.repeat(kappas, self.degrees)

        Kappas = sp.sparse.diags(kappas/self.degrees)
        Kappas_nodal = sp.sparse.diags(kappas_nodal)

        k_stretch = np.repeat(stretching_modulus*np.ones(len(mid_nodes)), 2)
        K_stretch = sp.sparse.diags(k_stretch)

        mass = mass*np.ones_like(k_stretch)

        return Kappas, Kappas_nodal, K_stretch, mass

    def num_optim_dof(self):
        return 3


class PointMassResonanceOptimizer(PointMassOptimizer):
    def __init__(self, network, ω_sqr, qs, regularize=0.1):
        """ Initialize with given MechanicalNetwork,
        band gap fractions, and wave vectors
        """
        self.network = network
        self.gaps = [1.0]
        self.gap_inds = [1]
        self.ω_sqr = ω_sqr
        self.qs = qs
        self.ϵ = regularize

        self.Qs = [self.network.equilibrium_matrix_ft(q).toarray() for q in qs]
        self.QHs = [Q.conj().transpose() for Q in self.Qs]

    def initialize(self, x0):
        """ Initialize the state for optimization of given initial conditions
        """
        self.ωsqrs = [np.array([self.ω_sqr + 1j*self.ϵ for q in self.qs])
                        for gap_ind in self.gap_inds]
        self.x0 = x0.copy()

        # normalization factors
        res, grads = zip(*[self.loss_and_jac_gap(x0, oms)
                         for oms in self.ωsqrs])
        self.res_0 = np.array(res)

    def loss_and_jac(self, x, verbose=False):
        loss, jac = super().loss_and_jac(x)

        if verbose:
            print('loss:', loss)

        return -loss, -jac

    def loss_and_jac_gap(self, x, ωsqrs, summed=True):
        """ Spectral loss function for a gap defined by the given
        values of ω^2.
        """
        kd = np.diag(x)

        Ks = [np.dot(Q, np.dot(kd, QH)) for Q, QH in zip(self.Qs, self.QHs)]

        Gs = [ωsqr*np.eye(K.shape[0]) - K for K, ωsqr in zip(Ks, ωsqrs)]
        invs = [np.linalg.inv(G) for G in Gs]

        response = [np.abs(inv)**2 for inv in invs]

        grads = [2*np.sum(QH.transpose()*np.linalg.solve(
            G.dot(G.conj().transpose()).dot(G), Q), axis=0).real
                 for G, Q, QH in zip(Gs, self.Qs, self.QHs)]

        if summed:
            return np.sum(response), np.sum(grads, axis=0)
        else:
            return response, grads

def deformed_positions_jac(deformed_netw, k):
    """ Return the Jacobian of the node positions in the
    deformed network with respect to the spring stiffnesses
    """
    netw = deformed_netw.graph['undeformed_network']

    d = netw.graph['dimension']
#     ratios = deformed_netw.graph['deformed_length_ratios']

    # derivatives of the normal vectors
    e_aff = deformed_netw.graph['e_aff']

    # undeformed K
    Q = netw.graph['Q']
    K = Q.dot(sp.sparse.diags(k)).dot(Q.transpose())

    # vector components
    V = sp.linalg.lstsq(K.toarray(), Q.toarray(), cond=1e-10)[0]

    # prefactors
    prefactors = V.transpose().dot(Q.dot(k*e_aff)) - e_aff

    # jacobian of the node vectors
    u_jac = prefactors*V

    return u_jac

def deformed_lengths_jac(deformed_netw, k, u_jac):
    """ Return the Jacobian of the deformed edge lengths
    """
    d = deformed_netw.graph['dimension']
    E = deformed_netw.graph['E'].toarray()

    b_jac = E.transpose().dot(u_jac.reshape(d, deformed_netw.number_of_nodes(), deformed_netw.number_of_edges()))

    len_grad = np.einsum('ij,ijk->ik', deformed_netw.graph['b_hat'], b_jac)

    return len_grad, b_jac
#     lg = -k*ratios/netw.graph['lengths']

def deformed_incidence_jac(deformed_netw, k, q, u_jac):
    """ Return the Jacobian of the deformed Fourier transformed incidence matrix
    with respect to stiffnesses k and wave vector q.
    """
    x_s = deformed_netw.graph['edge_node_pos']
    x_ab = deformed_netw.graph['edge_pos']
    d = deformed_netw.graph['dimension']
    nn = deformed_netw.number_of_nodes()
    ne = deformed_netw.number_of_edges()

    phase_a = np.exp(-1j*np.dot(x_s[0,:,:] - x_ab, q))
    phase_b = np.exp(-1j*np.dot(x_s[1,:,:] - x_ab, q))

    edge_i = deformed_netw.graph['edge_ind_list']
    # edge_i_y = edge_i_x + nn

    edge_is_a = [edge_i[:,0] + i*nn for i in range(d)]
    edge_is_b = [edge_i[:,1] + i*nn for i in range(d)]

    Es = []
    E0 = deformed_netw.graph['E'].astype(complex)
    # this is where orientation is reversed
    rev = E0.data[::2] == 1

    for i in range(ne):
        # u_jac_1 = u_jac[[edge_i_x[:,0],edge_i_y[:,0]],i].T
        # u_jac_2 = u_jac[[edge_i_x[:,1],edge_i_y[:,1]],i].T

        u_jac_1 = u_jac[edge_is_a,i].T
        u_jac_2 = u_jac[edge_is_b,i].T

        u_ab_jac = 0.5*(u_jac_1 + u_jac_2)

        phase_ai = -1j*np.dot(u_jac_1 - u_ab_jac, q)*phase_a
        phase_bi = -1j*np.dot(u_jac_2 - u_ab_jac, q)*phase_b

        Ed = E0.data.copy()

        phase_aa = phase_ai.copy()
        phase_bb = phase_bi.copy()
        phase_aa[rev] = phase_bi[rev]
        phase_bb[rev] = phase_ai[rev]

        Ed[::2] *= phase_aa
        Ed[1::2] *= phase_bb

        E = copy(E0)
        E.data = Ed

        Es.append(E)

    return Es

def deformed_equilibrium_jac(deformed_netw, k, q, u_jac, b_jac, len_jac):
    """ Return the Jacobian of the deformed Fourier transformed equilibrium matrix
    with respect to stiffnesses k and wave vector q.
    """
    x_s = deformed_netw.graph['edge_node_pos']
    x_ab = deformed_netw.graph['edge_pos']
    d = deformed_netw.graph['dimension']
    lens = deformed_netw.graph['lengths']
    b_hat = deformed_netw.graph['b_hat']
    nn = deformed_netw.number_of_nodes()
    ne = deformed_netw.number_of_edges()

    # new phases
    phase_a = np.exp(-1j*np.dot(x_s[0,:,:] - x_ab, q))
    phase_b = np.exp(-1j*np.dot(x_s[1,:,:] - x_ab, q))

    E = deformed_netw.graph['E']
    # this is where orientation is reversed
    rev = E.data[::2] == 1

    tmp = phase_a[rev].copy()
    phase_ax = phase_a.copy()
    phase_bx = phase_b.copy()

    phase_ax[rev] = phase_bx[rev]
    phase_bx[rev] = tmp

    edge_i = deformed_netw.graph['edge_ind_list']
    # edge_i_y = edge_i_x + nn

    edge_is_a = [edge_i[:,0] + i*nn for i in range(d)]
    edge_is_b = [edge_i[:,1] + i*nn for i in range(d)]

    Qs = []
    Q0 = deformed_netw.graph['Q'].astype(complex)

    b_hat_dot_b_jac = np.einsum('ij,ijk->ik', b_hat, b_jac)

    for i in range(ne):
        # u_jac_1 = u_jac[[edge_i_x[:,0],edge_i_y[:,0]],i].T
        # u_jac_2 = u_jac[[edge_i_x[:,1],edge_i_y[:,1]],i].T

        u_jac_1 = u_jac[edge_is_a,i].T
        u_jac_2 = u_jac[edge_is_b,i].T

        u_ab_jac = 0.5*(u_jac_1 + u_jac_2)

        phase_ai = -1j*np.dot(u_jac_1 - u_ab_jac, q)*phase_a
        phase_bi = -1j*np.dot(u_jac_2 - u_ab_jac, q)*phase_b

        Qd = Q0.data.copy()

        phase_aa = phase_ai.copy()
        phase_bb = phase_bi.copy()
        phase_aa[rev] = phase_bi[rev]
        phase_bb[rev] = phase_ai[rev]

        phase_aa = np.repeat(phase_aa, d)
        phase_bb = np.repeat(phase_bb, d)

        Qd[::2] *= phase_aa
        Qd[1::2] *= phase_bb

        # append derivatives of b_hat
        b_hat_jac = (b_jac[:,:,i] - b_hat*b_hat_dot_b_jac[:,i,np.newaxis])/lens[:,np.newaxis]

        bj_xs_a = [E.data[::2]*b_hat_jac[:,i]*phase_ax for i in range(d)]
        bj_xs_b = [E.data[1::2]*b_hat_jac[:,i]*phase_bx for i in range(d)]

        bj_1 = np.empty(d*(bj_xs_a[0].size), dtype=complex)
        bj_2 = np.empty(d*(bj_xs_b[0].size), dtype=complex)

        for i in range(d):
            bj_1[i::d] = bj_xs_a[i]
            bj_2[i::d] = bj_xs_b[i]

        Qd[::2] += bj_1
        Qd[1::2] += bj_2

        Q = copy(Q0)
        Q.data = Qd
        Qs.append(Q)

    return Qs

class PointMassSwitchOptimizer(PointMassOptimizer):
    """ Optimize a switchable network. Only one gap is allowed.
    """
    def __init__(self, network, gap, qs, Γ, undeformed_response_weight=0.5, min_response=0.5):
        super().__init__(network, [gap], qs)
        self.Γ = Γ
        self.σ = undeformed_response_weight
        self.min_response = min_response

    def total_response_deformed(self, x, ωsqrs_deformed, return_gradient=True):
        """ Like total_response, for the deformed network.
        Returns only part of the gradient, not the one related to the equilibrium position
        of the deformed network.
        """
        deformed = self.network.deformed_periodic_network(self.Γ, x)
        d = deformed.graph['dimension']

        Ks, Qs, Es, qeffs = zip(*[deformed.deformed_dynamical_matrix_at(x, q,
                                            return_matrices=True, return_qeff=True)
                           for q in self.qs_def])

        Ks = [K.toarray() for K in Ks]
        Qs = [Q.toarray() for Q in Qs]
        Es = [E.toarray() for E in Es]

        Gs = [ωsqr*np.eye(K.shape[0]) - K for K, ωsqr in zip(Ks, ωsqrs_deformed)]
        # G3s = [np.linalg.matrix_power(G, 3) for G in Gs]
        G3s = [G.dot(G.conj().T).dot(G) for G in Gs]

        invs = [np.linalg.inv(G) for G in Gs]

        def_resp = np.sum([np.abs(inv)**2 for inv in invs])

        if return_gradient:
            u_jac = deformed_positions_jac(deformed, x)
            len_jac, b_jac = deformed_lengths_jac(deformed, x, u_jac)
            len_undef = self.network.graph['lengths']
            len_def = deformed.graph['lengths']
            len_rat = deformed.graph['deformed_length_ratios']

            k_eff = np.diag(x*len_rat)
            k_lap = np.diag(x*(1 - len_rat))

            l = x[:,np.newaxis]*np.einsum('i,ij->ij', -len_undef/len_def**2, len_jac)
            l_eff = l.copy()
            l_eff[np.diag_indices_from(l_eff)] += len_rat

            l_lap = -l
            l_lap[np.diag_indices_from(l_lap)] += 1 - len_rat

            grads = np.zeros(deformed.number_of_edges())
            for qeff, G3i, Qi, Ei in zip(qeffs, G3s, Qs, Es):
                Qjacs = [Qj.toarray() for Qj in deformed_equilibrium_jac(deformed, x, qeff, u_jac, b_jac, len_jac)]
                Ejacs = [Ej.toarray() for Ej in deformed_incidence_jac(deformed, x, qeff, u_jac)]

                QiH = Qi.transpose().conj()
                EiH = Ei.transpose().conj()
                k_eff_dot_Qi = k_eff.dot(QiH)
                k_lap_dot_Ei = k_lap.dot(EiH)

                # Qi_l_eff_QiH = np.einsum('ij,jk,jl->ilk', Qi, l_eff, QiH)
                # Ei_l_lap_EiH = np.einsum('ij,jk,jl->ilk', Ei, l_lap, EiH)
                G3i_lu = sp.linalg.lu_factor(G3i)

                for i in range(deformed.number_of_edges()):
                    # construct derivative of K
                    K_Qj = Qjacs[i].dot(k_eff_dot_Qi)
                    K_Qsym = K_Qj + K_Qj.conj().transpose()
                    Kgrad = K_Qsym + (Qi*l_eff[:,i]).dot(QiH)

                    L_Ej = Ejacs[i].dot(k_lap_dot_Ei)
                    L_Esym = L_Ej + L_Ej.conj().transpose()
                    Lgrad = L_Esym + (Ei*l_lap[:,i]).dot(EiH)

                    Lgrad = sp.linalg.block_diag(*(d*[Lgrad]))
                    # Lgrad = sp.sparse.block_diag(d*[Lgrad]).toarray()

                    grad = Kgrad + Lgrad

                    # A = np.linalg.solve(G3i, grad)
                    A = sp.linalg.lu_solve(G3i_lu, grad)

                    grads[i] += 2*np.trace(A).real

            return def_resp, grads

        return def_resp

    def initialize(self, x0):
        """ Initialize the state for optimization of given initial conditions
        """
        self.x0 = x0.copy()

        # initialize deformed configuration
        deformed = self.network.deformed_periodic_network(self.Γ, x0)

        Ksd = [deformed.deformed_dynamical_matrix_at(x0, q).toarray() for q in self.qs]

        gi = self.gap_inds[0]

        self.ωsqrs_def = np.array([np.mean(sp.linalg.eigvalsh(K)[gi:gi+2]) for K in Ksd])
        # self.ωsqrs = np.concatenate((self.ωsqrs_def, 0.96*self.ωsqrs_def, 1.04*self.ωsqrs_def))
        self.ωsqrs = self.ωsqrs_def

        self.qs_def = self.qs
        # self.qs = np.tile(self.qs, 3)
        self.qs = self.qs_def
        self.Qs = 3*self.Qs
        self.QHs = 3*self.QHs

        # normalization factors
        res, grads = self.loss_and_jac_gap(x0, self.ωsqrs, summed=False)
        self.res_0 = res

        res_def, grads_def = self.total_response_deformed(x0, self.ωsqrs_def)
        self.res_def0 = res_def

    def loss_and_jac(self, x, verbose=False, return_gradient=True):
        if return_gradient:
            defo, grad_def = self.total_response_deformed(x, self.ωsqrs_def, return_gradient=True)
        else:
            defo = self.total_response_deformed(x, self.ωsqrs_def, return_gradient=False)
            grad_def = 0

        undef, grad_undef = self.loss_and_jac_gap(x, self.ωsqrs, summed=False)

        defo /= self.res_def0
        grad_def /= self.res_def0

        undef /= self.res_0
        grad_undef = grad_undef/self.res_0[:,np.newaxis]

        # we penalize any response that dips below p% of the original value but allow higher response
        combined = 0.5*(defo**2 + self.σ*np.mean((undef - self.min_response)**2))#*np.heaviside(-undef + self.min_response, 0)))

        residual = undef - self.min_response
        combined_grad = grad_def*defo + self.σ*np.mean(residual[:,np.newaxis]*grad_undef, axis=0)#*np.heaviside(-undef + self.min_response, 0)[:,np.newaxis], axis=0)

        # combined = defo + undef
        # combined_grad = grad_def*defo + grad_undef*undef

        if verbose:
            print('combined:', combined)
            print('undef:', undef)
            print('def:', defo)

        return combined, combined_grad

class PointMassDeformedOptimizer(PointMassSwitchOptimizer):
    def loss_and_jac(self, x, verbose=False, return_gradient=True):
        if return_gradient:
            defo, grad_def = self.total_response_deformed(x, self.ωsqrs_def, return_gradient=True)
        else:
            defo = self.total_response_deformed(x, self.ωsqrs_def, return_gradient=False)
            grad_def = 0

        # undef, grad_undef = self.loss_and_jac_gap(x, self.ωsqrs, summed=False)

        defo /= self.res_def0
        grad_def /= self.res_def0

        if verbose:
            pass

        return defo, grad_def

class PointMassResonanceSwitchOptimizer(PointMassSwitchOptimizer):
    def __init__(self, network, ω_sqr, qs, Γ, σ, regularize=0.1):
        """ Initialize with given MechanicalNetwork,
        band gap fractions, and wave vectors
        """
        self.network = network
        self.gaps = [1.0]
        self.gap_inds = [1]
        self.ω_sqr = ω_sqr
        self.qs = qs
        self.ϵ = regularize
        self.Γ = Γ
        self.σ = σ

        self.Qs = [self.network.equilibrium_matrix_ft(q).toarray() for q in qs]
        self.QHs = [Q.conj().transpose() for Q in self.Qs]

    def initialize(self, x0):
        """ Initialize the state for optimization of given initial conditions
        """
        self.ωsqrs = np.array([self.ω_sqr + 1j*self.ϵ for q in self.qs])
        self.x0 = x0.copy()

        # normalization factors
        res, grads = self.loss_and_jac_gap(x0, self.ωsqrs)
        self.res_0 = res

        res_def, grads_def = self.total_response_deformed(x0, self.ωsqrs)
        self.res_def0 = res_def

    def loss_and_jac(self, x, verbose=False, return_gradient=True):
        if return_gradient:
            defo, grad_def = self.total_response_deformed(x, self.ωsqrs, return_gradient=True)
        else:
            defo = self.total_response_deformed(x, self.ωsqrs, return_gradient=False)
            grad_def = 0

        undef, grad_undef = self.loss_and_jac_gap(x, self.ωsqrs)

        defo /= self.res_def0
        grad_def /= self.res_def0

        undef /= self.res_0
        grad_undef /= self.res_0

        # combined = np.sqrt(defo**2 + self.σ*(undef - 1)**2)
        # combined_grad = (grad_def*defo + grad_undef*self.σ*(undef - 1))/combined

        combined = -defo**2 + self.σ*(undef - 1)**2
        combined_grad = -2*grad_def*defo + 2*grad_undef*self.σ*(undef - 1)

        if verbose:
            print('combined:', combined)
            print('undef:', undef)
            print('def:', defo)

        return combined, combined_grad
