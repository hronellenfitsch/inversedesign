 #!/usr/bin/env python3

"""
  networks.py

  contains code to generate various types of mechanical networks
  that can then be optimized using spectral.py.

  (c) Henrik Ronellenfitsch 2017-2018
"""

from copy import deepcopy, copy

import numpy as np
import scipy as sp
import scipy.optimize as optimize
from scipy.spatial import Delaunay, ConvexHull, Voronoi
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import pandas as pd

from collections import defaultdict

from .alphashapes import concave_hull

def remove_duplicate_digraph_edges(G):
    edges_to_rem = []

    for e in G.edges:
        if G.has_edge(*e[::-1]) and e not in edges_to_rem:
            edges_to_rem.append(e[::-1])

    G.remove_edges_from(edges_to_rem)

def fourier_transform_Q(G, Q, q):
    """ Fourier transform the Q matrix to Q(q)
    """
    dim = G.graph['dimension']
    n = len(G.graph['nodelist'])

    Q_ft = Q.copy().astype(complex).tolil()

    # transform each coordinate block independently.
    for e_i, (u, v) in enumerate(G.graph['edgelist']):
        u_i = G.node[u]['idx']
        v_i = G.node[v]['idx']

        R_u = G.node[u]['x']

        # b = np.array([G[u][v]['b_hat_{}'.format(i)] for i in range(dim)])
        b = G.graph['b_hat'][e_i,:]
        R_v = R_u + b*G[u][v]['length']

        # this is technically not necessary and should cancels out
        R_uv = R_u + 0.5*b*G[u][v]['length']

        # if G[u][v]['periodic']:
        for i in range(dim):
            Q_ft[u_i+i*n,e_i] *= np.exp(-1j*(R_u - R_uv).dot(q))
            Q_ft[v_i+i*n,e_i] *= np.exp(-1j*(R_v - R_uv).dot(q))

    return Q_ft.tocsc()


def fourier_transform_E(G, E, q):
    """ Fourier transform the incidence E matrix to E(q)
    """
    dim = G.graph['dimension']
    n = len(G.graph['nodelist'])

    E_ft = E.copy().astype(complex).tolil()

    # transform each coordinate block independently.
    for e_i, (u, v) in enumerate(G.graph['edgelist']):
        u_i = G.node[u]['idx']
        v_i = G.node[v]['idx']

        R_u = G.node[u]['x']

        b = G.graph['b_hat'][e_i,:]#np.array([G[u][v]['b_hat_{}'.format(i)] for i in range(dim)])
        R_v = R_u + b*G.graph['lengths'][e_i]#G[u][v]['length']

        # this is technically not necessary and should cancels out
        R_uv = R_u + 0.5*b*G.graph['lengths'][e_i]#G[u][v]['length']

        # if G[u][v]['periodic']:
        # for i in range(dim):
        E_ft[u_i,e_i] *= np.exp(-1j*(R_u - R_uv).dot(q))
        E_ft[v_i,e_i] *= np.exp(-1j*(R_v - R_uv).dot(q))

        # print('r_uv', R_uv)
        # print('r_uv', R_uv)

    return E_ft.tocsc()

class MechanicalNetwork(nx.DiGraph):
    """ Models a mechanical network using a networkx digraph.
    In particular, adds methods for the standard equilibrium matrix
    and spring mass matrix
    """
    def __init__(self, dimension=2):
        super().__init__(self)

        self.graph['dimension'] = dimension

    def init_attrs(self):
        """ Initialize default attributes like fixed edge and node list
        and construct the graph and network matrices
        """
        try:
            nodes = sorted(list(self.nodes()))
        except:
            nodes = list(self.nodes())

        for i, nd in enumerate(nodes):
            self.node[nd]['idx'] = i

        self.graph['edgelist'] = list(self.edges())
        self.graph['nodelist'] = nodes
        self.graph['b_hat'] = np.array([[self.edges[u,v]['b_hat_{}'.format(i)] for u, v in self.graph['edgelist']]
                  for i in range(self.graph['dimension'])]).T
        self.graph['E'] = self.incidence_matrix()
        self.graph['Et'] = self.graph['E'].transpose().tocsc()
        self.graph['F'] = np.abs(self.graph['E'])
        self.graph['Q'] = self.equilibrium_matrix()
        self.graph['pos'] = dict((n, self.nodes[n]['x']) for n in nodes)
        self.graph['x'] = np.array([self.nodes[n]['x'] for n in nodes])
        self.graph['lengths'] = np.array([self.edges[e]['length'] for e in self.graph['edgelist']])
        self.graph['edge_pos'] = self.edge_positions()
        self.graph['edge_node_pos'] = self.edge_node_positions()
        self.graph['edge_ind_list'] = self.edge_ind_list()

    def add_dimension(self):
        """ Return the network embedded into R^(d+1) by adding a dummy dimension
        to all geometric quantities.
        """
        H = deepcopy(self)

        # embed node positions
        for n in H.nodes:
            x_old = self.node[n]['x']
            x_new = np.zeros(x_old.shape[0] + 1)
            x_new[:-1] = x_old

            H.node[n]['x'] = x_new

        # embed b_hat
        d = self.graph['dimension']
        for u, v in self.edges:
            H.edges[u,v][f'b_hat_{d}'] = 0.0

        # update graph attributes
        H.graph['dimension'] = d + 1

        H.init_attrs()

        return H

    def edge_positions(self):
        """ Return vector of edge positions. The edge position is
        the mean of the associated node positions, while taking periodicity
        into account.
        """
        x_a = np.array([self.node[a]['x'] for a, b in self.graph['edgelist']])
        x_ab = x_a + 0.5*self.graph['lengths'][:,np.newaxis]*self.graph['b_hat']

        return x_ab


    def edge_node_positions(self):
        """ Return the oriented node positions belonging to each edge.
        These are the positions of the node where the edge begins and where it
        ends.
        The end node is defined using the normal vector, such that for
        periodic edges, it will lie in the adjacent unit cell.
        """
        x_a = np.array([self.node[a]['x'] for a, b in self.graph['edgelist']])
        x_b = x_a + self.graph['lengths'][:,np.newaxis]*self.graph['b_hat']

        return np.array([x_a, x_b])

    def edge_ind_list(self):
        return np.array([[self.graph['nodelist'].index(a), self.graph['nodelist'].index(b)]
         for a, b in self.graph['edgelist']], dtype=int)

    def incidence_matrix(self):
        """ Construct and return the oriented incidence matrix
        """
        E = nx.incidence_matrix(self, edgelist=self.graph['edgelist'],
                           nodelist=self.graph['nodelist'],
                           oriented=True)
        return E

    def equilibrium_matrix(self):
        """ Construct and return the equilibrium matrix Q from the spatially embedded graph G.
        Make sure that the first edge is always oriented along the x axis.
        """
        # dim = self.graph['dimension']
        # Es = [nx.incidence_matrix(self, oriented=True, edgelist=self.graph['edgelist'],
        #                           nodelist=self.graph['nodelist'],
        #                           weight='b_hat_{}'.format(i)) for i in range(dim)]
        #
        # Q = sp.sparse.vstack(Es)

        # b_hats = [[self.edges[u,v]['b_hat_{}'.format(i)] for u, v in self.graph['edgelist']]
        #           for i in range(self.graph['dimension'])]

        return self.equilibrium_matrix_b_hats(self.graph['b_hat'])

    def equilibrium_matrix_b_hats(self, b_hats, Et=None):
        """ Return the equilibrium matrix constructed by using fast
        sparse operations and the given vectors of [b_hat_0, b_hat_1, ...],
        where each entry is a N-dimensional b_hat vector for the i component
        of b_hat of each edge.
        """
        if Et is None:
            Et = self.graph['Et']

        Es = []
        for i in range(self.graph['dimension']):
            Ec = Et.copy()
            Ec.data *= np.take(b_hats[:,i], Ec.indices)
            Es.append(Ec)

        return sp.sparse.hstack(Es).transpose().tocsc()

    def incidence_matrix_ft(self, q):
        """ Return the Fourier transformed incidence matrix,
        using a fast method.
        """
        x_s = self.graph['edge_node_pos']
        x_ab = self.graph['edge_pos']

        phase_a = np.exp(-1j*np.dot(x_s[0,:,:] - x_ab, q))
        phase_b = np.exp(-1j*np.dot(x_s[1,:,:] - x_ab, q))

        E = self.graph['E'].astype(complex)

        # this is where orientation is reversed
        rev = E.data[::2] == 1

        phase_aa = phase_a.copy()
        phase_aa[rev] = phase_b[rev]
        phase_b[rev] = phase_a[rev]

        E.data[::2] *= phase_aa
        E.data[1::2] *= phase_b

        return E

    def non_oriented_incidence_matrix_ft(self, q):
        """ Return the Fourier transformed non-oriented incidence matrix,
        using a fast method.
        """
        x_s = self.graph['edge_node_pos']
        x_ab = self.graph['edge_pos']

        phase_a = np.exp(-1j*np.dot(x_s[0,:,:] - x_ab, q))
        phase_b = np.exp(-1j*np.dot(x_s[1,:,:] - x_ab, q))

        E = self.graph['E'].astype(complex)

        # this is where orientation is reversed
        rev = E.data[::2] == 1

        phase_aa = phase_a.copy()
        phase_aa[rev] = phase_b[rev]
        phase_b[rev] = phase_a[rev]

        # we don't multiply by the phase, we just write it.
        # this kills the orientation.
        E.data[::2] = phase_aa
        E.data[1::2] = phase_b

        return E

    def equilibrium_matrix_ft(self, q, Q=None):
        """ Return the Fourier transformed equilibrium matrix,
        using a fast method.
        """
        x_s = self.graph['edge_node_pos']
        x_ab = self.graph['edge_pos']
        d = self.graph['dimension']

        phase_a = np.exp(-1j*np.dot(x_s[0,:,:] - x_ab, q))
        phase_b = np.exp(-1j*np.dot(x_s[1,:,:] - x_ab, q))

        if Q is None:
            Q = self.graph['Q'].astype(complex)
        else:
            Q = Q.astype(complex)
        E = self.graph['E']

        # reverse phases where orientation is reversed
        rev = E.data[::2] == 1
        phase_aa = phase_a.copy()
        phase_aa[rev] = phase_b[rev]
        phase_b[rev] = phase_a[rev]

        phase_aa = np.repeat(phase_aa, d)
        phase_b = np.repeat(phase_b, d)

        # first node
        Q.data[::2] *= phase_aa
        Q.data[1::2] *= phase_b

        return Q

    def equilibrium_matrix_ft_old(self, q):
        """ Return the Fourier transformed equilibrium matrix.
        """
        Q = self.graph['Q']
        Q_ft = fourier_transform_Q(self, Q, q)

        return Q_ft

    def incidence_matrix_ft_old(self, q):
        """ Return the Fourier transformed incidence matrix.
        """
        E = self.graph['E']
        E_ft = fourier_transform_E(self, E, q)

        return E_ft

    def convex_hull(self):
        """ Return the ConvexHull of the network, defined as the convex hull
        of either the finite sample or the periodic unit cell
        """
        pts = np.array([self.nodes[nd]['x'] for nd in self.graph['nodelist']])
        ch = ConvexHull(pts, qhull_options='Qc')

        return ch

    def concave_hull(self, alpha):
        """ Return the concave hull, i.e., the indices of the points on the
        boundary of the alpha shape for given alpha
        """
        pts = np.array([self.nodes[nd]['x'] for nd in self.graph['nodelist']])
        hull_idx = concave_hull(pts, alpha)

        return hull_idx

    def bulk_modulus(self, k, return_modes=False):
        """ Calculate the elastic Bulk modulus B of the network with spring
        constants k.

        If the clamped system can't be solved (i.e., there are mechanisms
        in the bulk), return zero.

        If return_modes is set to True,
        return the original compression mode and the relaxed compression mode
        including the bulk response.
        """
        d = self.graph['dimension']
        Q = self.graph['Q'].toarray()
        K = self.stiffness_matrix(k).toarray()

        ch = self.convex_hull()

        # coplanar points must be included for networks
        # with a straight line boundary
        vertices = np.concatenate((ch.vertices, ch.coplanar[:,0]))

        # subtracting off the center of mass is technically not necessary
        # because translations will be projected out, but it looks better
        # when plotted.
        com = np.mean(ch.points[vertices], axis=0)
        # print(com)

        # uniform compression displacements
        u_compression = -(ch.points[vertices] - com).flatten('F')

        # projector onto the clamped points
        iis = np.arange(d*len(vertices))
        js = np.concatenate(tuple(i*self.number_of_nodes() + vertices for i in range(d)))
        data = np.ones(d*len(vertices))

        P = sp.sparse.coo_matrix((data, (iis, js)),
                                 shape=(d*len(vertices),
                                        d*self.number_of_nodes()))

        # construct bordered Hessian
        H = sp.sparse.bmat([[K, P.transpose()], [P, None]]).tocsr()

        # solve for linear displacements
        b = np.zeros(H.shape[0])
        b[d*self.number_of_nodes():] = u_compression

        x = sp.sparse.linalg.spsolve(H, b)

        # bulk displacements from clamping
        u = x[:d*self.number_of_nodes()]

        V_el = 0.5*np.dot(u.T, K.dot(u))

        # note, in 2D, Qhull's volume is the area and area is the perimeter.
        if return_modes:
            return np.nan_to_num(V_el/(2*ch.volume)), P.transpose().dot(u_compression), u
        else:
            return np.nan_to_num(V_el/(2*ch.volume))

    def affine_response(self, eta):
        """ Return the affine response (bond extensions) of the network to
        the finite deformation lambda = Id + eta.
        """
        b_hat = np.array([[self.edges[e]['b_hat_{}'.format(i)]
                                         for i in range(self.graph['dimension'])]
                          for e in self.graph['edgelist']])
        lengths = np.array([self.edges[e]['length']
                          for e in self.graph['edgelist']])

        b = b_hat*lengths[:,np.newaxis]

        epsilon = 0.5*(eta + eta.T)
        e_aff = np.sum(b*np.dot(epsilon, b_hat.T).T, axis=1)

        return e_aff


    def spring_mass_matrix(self, m_s=None):
        """ Construct the spring mass matrix for masses m_s
        using the oriented incidence matrix E and non-oriented incidence matrix F
        """
        nodelist = self.graph['nodelist']
        edgelist = self.graph['edgelist']
        dim = self.graph['dimension']

        E = self.graph['E']
        F = self.graph['F']

        if m_s is None:
            m_s = np.ones(len(edgelist))

        mass = sp.sparse.diags(m_s)
        block = E.dot(mass.dot(E.transpose()))/12 + 3*F.dot(mass.dot(F.transpose()))/12

        blocks = [block for i in range(dim)]
        return sp.sparse.block_diag(blocks)

    def stiffness_matrix(self, k, q=None):
        """ Construct and return the stiffness matrix for given stiffnesses k.
        If q is not None, return the Fourier transform K(q) for q in the first Brillouin zone
        """
        Q = self.graph['Q']
        stiff = sp.sparse.diags(k)

        if q is None:
            return Q.dot(stiff.dot(Q.transpose()))
        else:
            Q_ft = fourier_transform_Q(self, Q, q)
            return Q_ft.dot(stiff.dot(Q_ft.conj().transpose()))

    def compute_b_hat(self, u, v):
        """ Compute b_hat for the edge (u, v) and store as attribute
        """
        x = self.node[u]['x']
        y = self.node[v]['x']

        b = y - x
        b_hat = b / np.linalg.norm(b)

        for i in range(b_hat.shape[0]):
            self[u][v]['b_hat_{}'.format(i)] = b_hat[i]

    def compute_len(self, u, v):
        """ Compute the length for the edge (u, v) and store as attribute
        """
        x = self.node[u]['x']
        y = self.node[v]['x']

        b = y - x

        self[u][v]['length'] = np.linalg.norm(b)

    def recompute_edge_geometry(self):
        """ Recompute b_hat and lengths for all edges.
        This might be useful if you want to change the geometry on
        the fly
        """
        for u, v in self.edges:
            self.compute_len(u, v)
            self.compute_b_hat(u, v)

    def add_edge(self, *args, **kwargs):
        """ Adds a new edge to the mechanical network.
        if both nodes it connects have an associated position and
        no explicit b_hat is given, b_hat is explicitly calculated.
        """
        u, v = args
        compute = 'b_hat_0' not in kwargs
        compute_len = 'length' not in kwargs

        if 'periodic' not in kwargs:
            kwargs['periodic'] = False

        super().add_edge(*args, **kwargs)

        if compute and 'x' in self.node[u] and 'x' in self.node[v]:
            self.compute_b_hat(u, v)

        if compute_len and 'x' in self.node[u] and 'x' in self.node[v]:
            self.compute_len(u, v)

    def spectrum(self, k, m_s=None, m=1, return_eigenvectors=False, q=None):
        """ Return the vibrational spectrum of the
        mechanical network for given stiffnesses k with point masses m.
        If m_s is given, add the terms for spring masses.
        If q is not None, return the spectrum at the given value of
        q in the Brillouin zone.
        """
        n = self.graph['Q'].shape[0]

        M = m*sp.sparse.eye(n)

        if m_s is not None:
            M += self.spring_mass_matrix(m_s=m_s)

        K = self.stiffness_matrix(k, q=q)

        if return_eigenvectors:
            return sp.linalg.eigh(K.toarray(), b=M.toarray())
        else:
            return sp.linalg.eigvalsh(K.toarray(), b=M.toarray())

    def deformed_periodic_network(self, Γ, k):
        """ Return a new *periodic* mechanical network that represents
        the new equilibrated positions after applying the global
        deformation Γ.

        It is important to note that this function is for fully periodic
        networks only. Any other network must have Additional
        constraints applied that fix the boundary nodes.
        """
        η = Γ - np.eye(self.graph['dimension'])

        # affine stretched positions
        x_eq = self.graph['x']
        x_def = np.apply_along_axis(lambda x: np.dot(η, x), 1, x_eq)

        # bond vectors
        # b_hat = np.array([[self.edges[e][f"b_hat_{i}"] for i in range(self.graph['dimension'])]
        #                   for e in self.graph['edgelist']])
        b_hat = self.graph['b_hat']
        lens = self.graph['lengths']#np.array([self.edges[e]['length'] for e in self.graph['edgelist']])
        b = b_hat*lens[:,np.newaxis]

        η_b = np.apply_along_axis(lambda x: np.dot(η, x), 1, b)
        e_aff = np.sum(b_hat * η_b, axis=1)

        # nonaffine relaxed displacements
        Q = self.graph['Q']
        k_diag = sp.sparse.diags(k)
        K = Q.dot(k_diag).dot(Q.transpose())

        rhs = Q.dot(k_diag).dot(e_aff)

        # res = sp.sparse.linalg.lsqr(K, rhs, atol=1e-12, btol=1e-12)
        res = sp.linalg.lstsq(K.toarray(), rhs, cond=1e-11)
        # print(res)

        u_aff = -res[0].reshape(self.graph['dimension'], self.number_of_nodes()).T

        # total displacement and corrected positions
        u = x_def + u_aff
        x_new = x_eq + u

        # corrected b_hats
        Δu_aff = self.graph['E'].transpose().dot(u_aff)

        b_new = b + η_b + Δu_aff
        lens_new = np.sqrt(np.einsum('...i,...i', b_new, b_new))
        b_hat_new = b_new/lens_new[:,np.newaxis]

        length_ratios = lens/lens_new

        # now copy over to new network and add new attributes
        netw_def = deepcopy(self)

        for i, n in enumerate(netw_def.graph['nodelist']):
            netw_def.nodes[n]['x'] = x_new[i,:]

        for i, e in enumerate(netw_def.graph['edgelist']):
            netw_def.edges[e]['length'] = lens_new[i]

            for j in range(self.graph['dimension']):
                netw_def.edges[e][f'b_hat_{j}'] = b_hat_new[i,j]
            # netw_def.edges[e]['b_hat_1'] = b_hat_new[i,1]

        netw_def.graph['deformed_length_ratios'] = length_ratios
        netw_def.graph['undeformed_lengths'] = lens
        netw_def.graph['Γ'] = Γ
        netw_def.graph['e_aff'] = e_aff
        netw_def.graph['is_deformed'] = True
        netw_def.graph['undeformed_network'] = self

        netw_def.init_attrs()

        return netw_def

    def deformed_dynamical_matrix_at(self, k, q, return_qeff=False, return_matrices=False):
        """ Return the dynamical matrix for the deformed periodic network
        at the desired wavevector q. q is defined in the original,
        undeformed BZ, and then converted to an effective, deformed q_eff
        before computing the dynamical matrix.
        """
        length_ratios = self.graph['deformed_length_ratios']
        k_eff = k*length_ratios
        k_lap = k*(1 - length_ratios)

        # Q = self.graph['Q']
        # E = self.graph['E']

    #   effective q vector in the deformed unit cell is related to the original q vector
        q_eff = np.linalg.solve(self.graph['Γ'].T, q)

        Q_def = self.equilibrium_matrix_ft(q_eff)
        E_def = self.incidence_matrix_ft(q_eff)

        K_eff = Q_def.dot(sp.sparse.diags(k_eff)).dot(Q_def.conj().transpose())
        L_eff = E_def.dot(sp.sparse.diags(k_lap)).dot(E_def.conj().transpose())

        L_eff = sp.sparse.block_diag(self.graph['dimension']*[L_eff])

        D = K_eff + L_eff

        if return_qeff:
            if return_matrices:
                return D, Q_def, E_def, q_eff
            else:
                return D, q_eff

        if return_matrices:
            return D, Q_def, E_def

        return D

    def deformed_spectrum_at(self, k, q):
        """ Return the spectrum of the deformed network at
        desired wavevector q (which is defined in the undeformed
        BZ)
        """
        D_q = self.deformed_dynamical_matrix_at(k, q)
        u = sp.linalg.eigvalsh(D_q.toarray(), turbo=True)

        return u

    def deformed_spectrum_and_vecs_at(self, k, q):
        """ Return the spectrum of the deformed network at
        desired wavevector q (which is defined in the undeformed
        BZ)
        """
        D_q = self.deformed_dynamical_matrix_at(k, q)
        u, v = sp.linalg.eigh(D_q.toarray(), turbo=True)

        return u, v

    def spectrum_at(self, k, q):
        """ Return spectrum of the dynamical matrix at wavevector q.
        """
        K_q = self.dynamical_matrix_at(k, q)
        u = sp.linalg.eigvalsh(K_q.toarray(), turbo=True)

        return u

    def spectrum_and_vecs_at(self, k, q):
        """ Return spectrum of the dynamical matrix at wavevector q.
        """
        K_q = self.dynamical_matrix_at(k, q)
        u, v = sp.linalg.eigh(K_q.toarray(), turbo=True)

        return u, v

    def mass_spectrum_at(self, k, m_spring, q, m_node=None):
        """ Return spectrum of the dynamical matrix at wavevector q.
        """
        K_q = self.dynamical_matrix_at(k, q)
        M_q = self.mass_matrix_at(m_spring, q, m_node=m_node)
        u = sp.linalg.eigvalsh(K_q.toarray(), b=M_q.toarray(), turbo=True)

        return u

    def mass_matrix_at(self, m_spring, q, m_node=None):
        """ Return the Fourier transformed mass matrix at wavevetor q.
        """
        E = self.incidence_matrix_ft(q)
        F = self.non_oriented_incidence_matrix_ft(q)

        m = sp.sparse.diags(m_spring)
        M = E.dot(m).dot(E.transpose().conj())/12 + 3/12 * F.dot(m).dot(F.transpose().conj())

        if m_node is not None:
            M = M + sp.sparse.diags(m_node)

        return sp.sparse.block_diag(self.graph['dimension']*[M])

    def dynamical_matrix_at(self, k, q, return_matrices=False):
        """ Return the dynamical matrix at given wavevector
        """
        K = sp.sparse.diags(k)
        # Q = self.graph['Q']

        Q_q = self.equilibrium_matrix_ft(q)
        K_q = Q_q.dot(K).dot(Q_q.conj().transpose())

        if return_matrices:
            return K_q, Q_q
        else:
            return K_q

    def draw_edges_3d(self, width, ax, periodic_edges_mirror=True,
                      plot_periodic=True, coords=None, **kwargs):
        """ Like draw_edges_2d, but assumes that we are plotting to
        an Axes3D object
        """
        if 'color' not in kwargs:
            kwargs['color'] = 'k'

        if coords is None:
            coords = self.graph['x']

        pos = dict((n, xx) for n, xx in zip(self.graph['nodelist'], coords))

        for w, (u, v) in zip(width, self.graph['edgelist']):
            x = pos[u]
            y = pos[v]

            pts = np.stack((x,y))
            if not self.edges[u,v]['periodic']:
                ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=w, **kwargs)
            elif plot_periodic:
                b_hat = np.array([self.edges[u,v]['b_hat_{}'.format(i)]
                                                  for i in range(3)])

                ax.plot([pts[0,0], pts[0,0] + b_hat[0]],
                        [pts[0,1], pts[0,1] + b_hat[1]],
                        [pts[0,2], pts[0,2] + b_hat[2]],
                        linewidth=w, **kwargs)

                if periodic_edges_mirror:
                    ax.plot([pts[1,0], pts[1,0] - b_hat[0]],
                            [pts[1,1], pts[1,1] - b_hat[1]],
                            [pts[1,2], pts[1,2] - b_hat[2]],
                            linewidth=w, **kwargs)

    def draw_edges_2d_manual(self, width, ax, pos, b_hat, periodic_edges_mirror=True,
                      draw_node_labels=False, **kwargs):
        """ Draw the network edges with widths width into the
        matplotlib axis ax. This is for 2D networks only.
        """
        G = self
        for (u, v), w, bh in zip(G.graph['edgelist'], width, b_hat):
            # x = G.node[u]['x']
            # y = G.node[v]['x']
            x, y = pos[G.graph['nodelist'].index(u)], pos[G.graph['nodelist'].index(v)]

            # b_hat = np.array([G.edges[u,v]['b_hat_0'], G.edges[u,v]['b_hat_1']])
            b_hat = bh

            if not G.edges[u,v]['periodic']:
                ax.plot([x[0], y[0]], [x[1], y[1]], 'k-', linewidth=w, **kwargs)

                if draw_node_labels:
                    ax.text(x[0], x[1], '{}'.format(u))
                    ax.text(y[0], y[1], '{}'.format(v))

            else:
                # plot periodic edges
                e = b_hat*self.edges[u,v]['length']
                ax.plot([x[0], x[0] + e[0]], [x[1], x[1] + e[1]], 'k-', linewidth=w, **kwargs)

                if draw_node_labels:
                    ax.text(x[0] + e[0], x[1] + e[1], '{}'.format(v))

                if periodic_edges_mirror:
                    f = -e
                    ax.plot([y[0], y[0] + f[0]], [y[1], y[1] + f[1]], 'k-', linewidth=w, **kwargs)

                    if draw_node_labels:
                        ax.text(y[0] + f[0], y[1] + f[1], '{}'.format(u))

    def draw_edges_2d(self, width, ax, periodic_edges_mirror=True,
                      draw_node_labels=False, coords=None, **kwargs):
        """ Draw the network edges with widths width into the
        matplotlib axis ax. This is for 2D networks only.
        """
        G = self

        if coords is None:
            coords = self.graph['x']

        pos = dict((n, xx) for n, xx in zip(self.graph['nodelist'], coords))

        edges = []
        for (u, v), w in zip(G.graph['edgelist'], width):
            x = pos[u]
            y = pos[v]

            b_hat = np.array([G.edges[u,v]['b_hat_0'], G.edges[u,v]['b_hat_1']])

            if not G.edges[u,v]['periodic']:
                ed = ax.plot([x[0], y[0]], [x[1], y[1]], 'k-', linewidth=w, **kwargs)
                edges.append(ed[0])

                if draw_node_labels:
                    ax.text(x[0], x[1], '{}'.format(u))
                    ax.text(y[0], y[1], '{}'.format(v))

            else:
                # plot periodic edges
                e = b_hat*self.edges[u,v]['length']
                ed = ax.plot([x[0], x[0] + e[0]], [x[1], x[1] + e[1]], 'k-', linewidth=w, **kwargs)
                edges.append(ed[0])

                if draw_node_labels:
                    ax.text(x[0] + e[0], x[1] + e[1], '{}'.format(v))

                if periodic_edges_mirror:
                    f = -e
                    ed = ax.plot([y[0], y[0] + f[0]], [y[1], y[1] + f[1]], 'k-', linewidth=w, **kwargs)
                    edges.append(ed[0])

                    if draw_node_labels:
                        ax.text(y[0] + f[0], y[1] + f[1], '{}'.format(u))

        return edges

    def draw_mode_2d(self, v, ax, head_fact=0.4, **kwargs):
        """ Draw the mode v as arrows in the axis ax.
        Additional kwargs are passed to matplotlib's arrow function
        """
        # reshape mode vector into array of 2D displacements
        vv = v.reshape(self.graph['dimension'], self.number_of_nodes()).T

        for n, (dx, dy) in zip(self.graph['nodelist'], vv):
            x, y = self.nodes[n]['x']

            l = np.linalg.norm([dx, dy]) + 1e-8
            ax.arrow(x, y, dx, dy, head_width=head_fact*l, **kwargs)

    """ Return a "Kagomized" version of the network.
    The Kagomized network consists of the "dual" network in which
    each edge is represented by a node, and nodes in the Kagomized
    network are connected if their original edges shared a node
    in the original network
    """
    def kagomized(self):
        Lg = nx.line_graph(nx.Graph(self))

        Kag = MechanicalNetwork(dimension=self.graph['dimension'])

        # edges are nodes
        k_nodes = list(self.graph["edgelist"])
        for i, (a, b) in enumerate(k_nodes):
            x = 0.5*(self.nodes[a]["x"] + self.nodes[b]["x"])
            Kag.add_node((a, b), x=x)

        # connect the edge-nodes
        for (e1, e2) in Lg.edges:
            Kag.add_edge(e1, e2)

        # initialize network
        Kag.init_attrs()

        return Kag

    """ Return the "dual" network in which a Voronoi tesselation
    of the points is created, and then this is taken as the new network
    """
    def voronized(self):
        Vor = MechanicalNetwork(dimension=self.graph['dimension'])

        pts = np.array([self.nodes[n]['x'] for n in self.nodes])

        tess = Voronoi(pts)

        # add Voronoi vertices as new nodes
        for i, x in enumerate(tess.vertices):
            Vor.add_node(i, x=x)

        # connect via Voronoi ridges
        for ridge in tess.ridge_vertices:
            # don't connect to the outside
            if ridge[0] != -1 and ridge[1] != -1:
                Vor.add_edge(ridge[0], ridge[1])

        Vor.init_attrs()

        return Vor

    def tile_unit_cell_2d(self, k, a, b):
        """ Tile the unit cell a times in x direction and b times in
        y direction. Return the new network and copy over stiffnesses.
        """
        Gg = MechanicalNetwork(dimension=2)

        # create new points
        lattice_a = np.array([self.graph['periods'][0], 0.0])
        lattice_b = np.array([0.0, self.graph['periods'][1]])

        for i in range(a):
            for j in range(b):
                shift = i*lattice_a + j*lattice_b
                cell = (i, j)

                for n in self.nodes():
                    Gg.add_node((cell, n), x=self.nodes[n]['x'] + shift)

        # create edges between points and between unit cells
        for e in self.edges:
            e0, e1 = e
            attrs = self.edges[e]
            if not self.edges[e]['periodic']:
                # simply copy the edge within each unit cell
                for i in range(a):
                    for j in range(b):
                        cell = (i, j)
                        if not ((cell, e1), (cell, e0)) in Gg.edges:
                            Gg.add_edge((cell, e0), (cell, e1), **attrs)
            else:
                # check which two unit cells are being connected here.
                b_v = np.array([self[e0][e1]['b_hat_{}'.format(i)] for i in range(self.graph['dimension'])])
                x_u = self.nodes[e0]['x']

                # make b_v slightly longer for the case where nodes are exactly on the cell boundary
                x_v = x_u + (1 + 1e-6)*b_v*self[e0][e1]['length']

                cell_diff = tuple(np.array(np.floor(x_v/self.graph['periods']), dtype=int))

                # connect them appropriately
                for i in range(a):
                    for j in range(b):
                        cell0 = (i, j)
                        cell1 = (int(np.mod(i + cell_diff[0], a)), int(np.mod(j + cell_diff[1], b)))

                        if not ((cell1, e1), (cell0, e0)) in Gg.edges:
                            Gg.add_edge((cell0, e0), (cell1, e1), **attrs)

                            if (cell1[0] - cell0[0] == cell_diff[0]) and (cell1[1] - cell0[1] == cell_diff[1]):
                                Gg.edges[(cell0, e0), (cell1, e1)]['periodic'] = False

        # initialize network
        Gg.init_attrs()

        # copy over stiffnesses
        X = np.zeros(Gg.number_of_edges())
        for i, (e0, e1) in enumerate(Gg.graph['edgelist']):
            old_e = (e0[1], e1[1])
            j = self.graph['edgelist'].index(old_e)

            X[i] = k[j]

        Gg.graph['periods'] = np.array([a, b])*self.graph['periods']
        return Gg, X

    def tile_unit_cell_3d(self, k, a, b, c):
        """ Tile the unit cell a times in x direction and b times in
        y direction and c times in z directoon. Return the new network and copy over stiffnesses.
        """
        Gg = MechanicalNetwork(dimension=3)

        # create new points
        lattice_a = np.array([self.graph['periods'][0], 0.0, 0.0])
        lattice_b = np.array([0.0, self.graph['periods'][1], 0.0])
        lattice_c = np.array([0.0, 0.0, self.graph['periods'][2]])


        for i in range(a):
            for j in range(b):
                for r in range(c):
                    shift = i*lattice_a + j*lattice_b + r*lattice_c
                    cell = (i, j, r)

                    for n in self.nodes():
                        Gg.add_node((cell, n), x=self.nodes[n]['x'] + shift)

        # create edges between points and between unit cells
        for e in self.edges:
            e0, e1 = e
            attrs = self.edges[e]
            if not self.edges[e]['periodic']:
                # simply copy the edge within each unit cell
                for i in range(a):
                    for j in range(b):
                        for r in range(c):
                            cell = (i, j, r)
                            if not ((cell, e1), (cell, e0)) in Gg.edges:
                                Gg.add_edge((cell, e0), (cell, e1), **attrs)
            else:
                # check which two unit cells are being connected here.
                b_v = np.array([self[e0][e1]['b_hat_{}'.format(i)] for i in range(self.graph['dimension'])])
                x_u = self.nodes[e0]['x']

                # make b_v slightly longer for the case where nodes are exactly on the cell boundary
                x_v = x_u + (1 + 1e-6)*b_v*self[e0][e1]['length']

                cell_diff = tuple(np.array(np.floor(x_v/self.graph['periods']), dtype=int))

                # connect them appropriately
                for i in range(a):
                    for j in range(b):
                        for r in range(c):
                            cell0 = (i, j, r)
                            cell1 = (int(np.mod(i + cell_diff[0], a)),
                                     int(np.mod(j + cell_diff[1], b)),
                                     int(np.mod(r + cell_diff[2], c)))

                            if not ((cell1, e1), (cell0, e0)) in Gg.edges:
                                Gg.add_edge((cell0, e0), (cell1, e1), **attrs)

                                if (cell1[0] - cell0[0] == cell_diff[0]) and (cell1[1] - cell0[1] == cell_diff[1]) and (cell1[2] - cell0[2] == cell_diff[2]):
                                    Gg.edges[(cell0, e0), (cell1, e1)]['periodic'] = False


        # remove edges that are periodic but connect the same cells
        # edges_to_rem = []
        # for (cell0, n0), (cell1, n1) in Gg.edges:
        #     if (cell0 == cell1) and Gg.edges[(cell0, n0), (cell1, n1)]['periodic']:
        #         edges_to_rem.append(((cell0, n0), (cell1, n1)))
        #
        # Gg.remove_edges_from(edges_to_rem)

        # initialize network
        Gg.init_attrs()

        # copy over stiffnesses
        X = np.zeros(Gg.number_of_edges())
        for i, (e0, e1) in enumerate(Gg.graph['edgelist']):
            old_e = (e0[1], e1[1])
            j = self.graph['edgelist'].index(old_e)

            X[i] = k[j]

        Gg.graph['periods'] = np.array([a, b, c])*self.graph['periods']
        return Gg, X

    def non_periodic(self, k):
        """ Remove all periodic edegs and copy over stiffnesses
        """
        # remove periodic edges
        Gg = deepcopy(self)
        edges_to_rem = [e for e in Gg.edges() if Gg.edges[e]['periodic']]

        Gg.remove_edges_from(edges_to_rem)
        Gg.init_attrs()

        # copy over stiffnesses
        X = np.zeros(Gg.number_of_edges())
        for i, e in enumerate(Gg.graph['edgelist']):
            j = self.graph['edgelist'].index(e)

            X[i] = k[j]

        return Gg, X

    def non_periodic_directional(self, k, x_nonp=False, y_nonp=True):
        """ Remove all periodic edegs and copy over stiffnesses
        """
        # remove periodic edges
        Gg = deepcopy(self)
        edges_to_rem = []

        # find the max and min cells
        x_max, y_max = np.max(Gg.graph['x'], axis=0)
        x_min, y_min = np.min(Gg.graph['x'], axis=0)

        for e in Gg.edges():
            if Gg.edges[e]['periodic']:
                a, b = e

                x_a = Gg.nodes[a]['x']

                # construct true position of other node
                R = x_a + Gg.edges[e]['length']*np.array([Gg.edges[e]['b_hat_0'], Gg.edges[e]['b_hat_1']])

                # remove those that go over the limits
                if x_nonp and ((R[0] > x_max) or (R[0] < x_min)):
                    edges_to_rem.append(e)
                if y_nonp and ((R[1] > y_max) or (R[1] < y_min)):
                    edges_to_rem.append(e)

        Gg.remove_edges_from(edges_to_rem)
        Gg.init_attrs()

        # copy over stiffnesses
        X = np.zeros(Gg.number_of_edges())
        for i, e in enumerate(Gg.graph['edgelist']):
            j = self.graph['edgelist'].index(e)

            X[i] = k[j]

        return Gg, X

class DelaunayNetwork(MechanicalNetwork):
    """ Models a network defined through a non-periodic Delaunay triangulation
    of random points in N-D.
    """
    def __init__(self, n, dimension, seed=None, copy_unit_cell=[],
                 periods=None, tol=1e-2):
        """ Initialize a new Delaunay network. Allows you to copy the unit cell
        along indicated dimensions
        """
        super().__init__(dimension=dimension)
        self.graph['n'] = n
        self.graph['seed'] = seed

        points = self.generate_points(seed, tol=tol)

        if periods is None:
            periods = len(copy_unit_cell)*[1.0]

        points, nodes = self.add_points(points, zip(copy_unit_cell, periods))
        self.construct_triangulation(points, nodes)
        self.init_attrs()

        self.graph['periods'] = periods

    def add_points(self, points, copy_unit_cell):
        """ Add points as nodes to the network
        """
        # copy directions
        nodes = [(i, 0) for i in range(points.shape[0])]

        if copy_unit_cell is not None:
            # for each copy direction, we copy the unit cell points
            # and shift the coordinates

            num_cells = 1
            n_unit_pts = points.shape[0]
            for d, period in copy_unit_cell:
                n_pts = points.shape[0]
                points = np.tile(points, (3, 1))

                points[n_pts:2*n_pts,d] += period
                points[2*n_pts:,d] -= period

                # add new ids
                num_new_cells = 2*num_cells
                for i in range(num_new_cells):
                    nodes.extend([(j, num_cells+i) for j in range(n_unit_pts)])

                num_cells += num_new_cells

        for nd, pt in zip(nodes, points):
            self.add_node(nd, x=pt)

        return points, nodes

    def generate_points(self, seed, tol=1e-2):
        """ Generate pseudo-random but uniformized points
        """
        if seed is not None:
            np.random.seed(seed)

        n, d = self.graph['n'], self.graph['dimension']

        def periodic_dist(u, v):
            abs_dists = np.abs(u - v)
            wrapped_abs = 1 - abs_dists

            return np.sqrt(np.sum(np.minimum(abs_dists, wrapped_abs)**2))

        def cost_fun(pts):
            pts = np.reshape(pts, (int(pts.shape[0]/d), d))
            real_pts = (1 + np.sin(pts))/2
            pdist = sp.spatial.distance.pdist(real_pts, periodic_dist)

            return np.sum(1./pdist)

        x0 = np.random.rand(n*d)

        ret = sp.optimize.minimize(cost_fun, x0, jac=False,
                        method='L-BFGS-B', tol=tol)

        pts = (1 + np.sin(ret.x))/2
        pts = np.reshape(pts, (int(pts.shape[0]/d), d))

        pts -= np.mean(pts, axis=0)
        pts += np.array(d*[0.5])

        return pts

    def generate_points_old(self, seed, uniformize=20):
        """ Generate random points
        """
        if seed is not None:
            np.random.seed(seed)

        n, d = self.graph['n'], self.graph['dimension']
        points = np.random.random((n, d))

        # make points more uniform
        for i in range(uniformize):
            # compute force from all others
            dist = squareform(1./pdist(points)**3)
            for j in range(points.shape[0]):
                F = np.zeros(d)
                for k in range(points.shape[0]):
                    F += -(points[k,:] - points[j,:])*dist[k,j]

                # boundary
                for l in range(d):
                    left = points[j,:].copy()
                    right = points[j,:].copy()
                    right[l] = 1 - right[l]

                    left_nr = np.linalg.norm(left)
                    right_nr = np.linalg.norm(right)

                    F[l] += 1./left_nr**2 - 1./right_nr**2

                points[j,:] += np.clip(0.00001*F, -0.01, 0.01)
                points[j,:] = np.clip(points[j,:], 0., 1.)

        return points

    def construct_triangulation(self, points, nodes):
        """ Construct a Delaunay triangulation of n random points
        and store the resulting network as a MechanicalNetwork.
        """
        # find the edges via Delaunay triangulation
        def find_neighbors(tess):
            neighbors = defaultdict(set)

            for simplex in tess.simplices:
                for idx in simplex:
                    other = set(simplex)
                    other.remove(idx)
                    neighbors[idx] = neighbors[idx].union(other)
            return neighbors

        def find_edges(neighbors):
            edges = set()

            for k in neighbors.keys():
                for v in neighbors[k]:
                    edges.add(tuple(sorted([k, v])))

            return list(edges)

        tri = Delaunay(points)
        neighbors = find_neighbors(tri)
        edges = find_edges(neighbors)

        # add edges to mechanical network
        for u, v in edges:
            self.add_edge(nodes[u], nodes[v], periodic=False, a=np.array([0, 0]))

    def make_periodic(self):
        """ Take all the edges that connect unit cell 0 to other cells,
        of the form ((i, 0), (j, k)), create a new edge ((i, 0), (j, 0))
        with the same attributes (in particular, b_hat), as the original one,
        then finally delete all the non-basic unit cells.
        Returns a copy of the new, periodic network
        """
        G = MechanicalNetwork(dimension=self.graph['dimension'])

        # find edges to copy over
        periodic_edges = [(u, v) for u, v in self.edges
                          if (u[1] == 0 or v[1] == 0)]

        for nd in self.nodes:
            if nd[1] == 0:
                G.add_node(nd, **self.nodes[nd])

        # add new edges with appropriate attributes
        for u, v in periodic_edges:
            (nd_u, cell_u) = u
            (nd_v, cell_v) = v
            uu = (nd_u, 0)
            vv = (nd_v, 0)

            attrs = self.edges[u,v].copy()
            attrs['periodic'] = cell_u != 0 or cell_v != 0

            if (vv, uu) not in G.edges:
                G.add_edge(uu, vv, **attrs)

        G.graph.update(self.graph)
        G.init_attrs()

        return G

class SquareTriangularGrid2D(MechanicalNetwork):
    """ Models a 2D Triangular grid with unequal edge lengths
    """
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)

        self.graph['n'] = n
        self.construct_network()

        self.init_attrs()

    def construct_network(self):
        """ Construct triangular mesh points.
        """
        n = self.graph['n']

        for i in range(n):
            for j in range(n):
                x = np.array([i, j], dtype=float)
                self.add_node((i, j), x=x)

        # square grid
        for i in range(n):
            for j in range(n):
                if (i+1, j) in self.nodes:
                    self.add_edge((i, j), (i+1,j))
                if (i, j+1) in self.nodes:
                    self.add_edge((i,j),(i,j+1))
        # diagonals
        for i in range(1, n, 2):
            for j in range(1, n, 2):
                if (i+1, j+1) in self.nodes:
                    self.add_edge((i, j), (i+1,j+1))
                if (i+1, j-1) in self.nodes:
                    self.add_edge((i, j), (i+1,j-1))
                if (i-1, j+1) in self.nodes:
                    self.add_edge((i, j), (i-1,j+1))
                if (i-1, j-1) in self.nodes:
                    self.add_edge((i, j), (i-1,j-1))

class SquareGrid2D(MechanicalNetwork):
    """ Models a 2D square grid.
    """
    def __init__(self, n, m, x_periodic=True, y_periodic=True, **kwargs):
        super().__init__(**kwargs)

        self.graph['n'] = n
        self.graph['m'] = m
        self.graph['x_periodic'] = x_periodic
        self.graph['y_periodic'] = y_periodic

        self.graph['periods'] = np.array([n, m])

        self.construct_network()

        self.init_attrs()

    def construct_network(self):
        """ Construct triangular mesh points.
        """
        n = self.graph['n']
        m = self.graph['m']

        x_p = self.graph['x_periodic']
        y_p = self.graph['y_periodic']

        for i in range(n):
            for j in range(m):
                x = np.array([i, j], dtype=float)
                self.add_node((i, j), x=x)

        # square grid
        for i in range(n):
            for j in range(m):
                if (i+1, j) in self.nodes:
                    self.add_edge((i, j), (i+1,j))
                elif x_p and ( (np.mod(i+1, n), j) in self.nodes):
                    self.add_edge((i, j), (np.mod(i+1, n), j), periodic=True, b_hat_0=1.0, b_hat_1=0.0, length=1.0)
                if (i, j+1) in self.nodes:
                    self.add_edge((i,j), (i,j+1))
                elif y_p and ( (i, np.mod(j+1, m)) in self.nodes):
                    self.add_edge((i,j), (i, np.mod(j+1, m)), periodic=True, b_hat_0=0.0, b_hat_1=1.0, length=1.0)


class TriangularGrid3D(DelaunayNetwork):
    """ Models a 3D triangular grid by generating the appropriate points
    and then performing a Delaunay triangulation.
    """
    def __init__(self, n, **kwargs):
        if 'copy_unit_cell' in kwargs:
            l = len(kwargs['copy_unit_cell'])
            kwargs['periods'] = [1.0*n,
                                 n*np.sqrt(3)/2,
                                 n*np.sqrt(2/3)][:l]

        super().__init__(n, 3, **kwargs)

    def construct_triangulation(self, points, nodes):
        super().construct_triangulation(points, nodes)

        # remove long edges
        edges_to_rem = [e for e in self.edges
                        if np.abs(self.edges[e]['length'] - 1) > 1e-8]
        self.remove_edges_from(edges_to_rem)

    def generate_points(self, seed, uniformize=20):
        """ Construct triangular mesh points.
        """
        a = 1.0
        b = np.sqrt(3)/2
        c = np.sqrt(2/3)
        d = np.sqrt(3)/6

        n = self.graph['n']

        points = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    x = np.array([i + 0.5*np.mod(j,2) + 0.5*np.mod(k,2),
                                  j*b + d*np.mod(k,2), k*c])
                    points.append(x)

        return np.array(points)

class CubicGrid3D(DelaunayNetwork):
    """ Models a 3D cubic grid by generating the appropriate points
    and then performing a Delaunay triangulation.
    """
    def __init__(self, n, **kwargs):
        if 'copy_unit_cell' in kwargs:
            l = len(kwargs['copy_unit_cell'])
            kwargs['periods'] = [n,
                                 n,
                                 n][:l]

        super().__init__(n, 3, **kwargs)

    def construct_triangulation(self, points, nodes):
        super().construct_triangulation(points, nodes)

        # remove long edges
        edges_to_rem = [e for e in self.edges
                        if self.edges[e]['length'] - 1 > 1e-8]
        self.remove_edges_from(edges_to_rem)

    def generate_points(self, seed, uniformize=20):
        """ Construct triangular mesh points.
        """
        n = self.graph['n']

        points = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    x = np.array([i, j, k])
                    points.append(x)

        for i in range(n-1):
            for j in range(n-1):
                for k in range(n-1):
                    x = np.array([i+0.5, j+0.5, k+0.5])
                    points.append(x)

        return np.array(points)


class TriangularGrid2D(MechanicalNetwork):
    """ Models a triangular grid of given linear dimension n.
    n must be even, otherwise it is set to n+1.
    """
    def __init__(self, *args, **kwargs):
        x_periodic = kwargs['x_periodic']
        y_periodic = kwargs['y_periodic']
        n = kwargs['n']

        # # n must be even
        # if np.mod(n, 2) == 1:
        #     n = n+1

        del kwargs['x_periodic']
        del kwargs['y_periodic']
        del kwargs['n']

        MechanicalNetwork.__init__(self, dimension=2)

        self.graph['linear_length'] = n
        self.graph['x_periodic'] = x_periodic
        self.graph['y_periodic'] = y_periodic

        self.graph['periods'] = float(n), float(n)*np.sqrt(3)/2

        self.construct_lattice()

        self.init_attrs()

    def construct_lattice(self):
        G = self
        x_periodic = self.graph['x_periodic']
        y_periodic = self.graph['y_periodic']
        n = self.graph['linear_length']

        def period_map(k):
            if k == n:
                return 0
            elif k == -1:
                return n-1
            else:
                return k

        def period_map_remove(k):
            if k == n:
                return -1
            else:
                return k

        def is_periodic(k):
            if k == n or k == -1:
                return True
            else:
                return False

        # periodic edges get modulated by e^(i a q)
        def a_x(k):
            if not x_periodic:
                return 0

            if k == n:
                return 1
            elif k == -1:
                return -1
            else:
                return 0

        def a_y(k):
            if not y_periodic:
                return 0

            if k == n:
                return 1
            elif k == -1:
                return -1
            else:
                return 0

        if x_periodic:
            period_map_i = period_map
        else:
            period_map_i = period_map_remove

        if y_periodic:
            period_map_j = period_map
        else:
            period_map_j = period_map_remove

        b = np.sqrt(3)/2
        pos = {}
        for i in range(n):
            for j in range(n):
                G.add_edge((i,j), (period_map_i(i+1), j), length=1, b_hat_0=1, b_hat_1=0,
                          periodic=is_periodic(i+1), a=np.array([a_x(i+1), a_y(j)]))
                G.add_edge((i,j), (period_map_i(i-1), j), length=1, b_hat_0=-1, b_hat_1=0,
                          periodic=is_periodic(i-1), a=np.array([a_x(i-1), a_y(j)]))

                if np.mod(j, 2) == 0:
                    G.add_edge((i,j), (i, period_map_j(j+1)), length=1, b_hat_0=0.5, b_hat_1=b,
                              periodic=is_periodic(j+1), a=np.array([a_x(i), a_y(j+1)]))
                    G.add_edge((i,j), (period_map_i(i-1), period_map_j(j+1)), length=1, b_hat_0=-0.5, b_hat_1=b,
                              periodic=is_periodic(i-1) or is_periodic(j+1),
                             a=np.array([a_x(i-1), a_y(j+1)]))
                    G.add_edge((i,j), (i, period_map_j(j-1)), length=1, b_hat_0=0.5, b_hat_1=-b,
                              periodic=is_periodic(j-1),
                              a=np.array([a_x(i), a_y(j-1)]))
                    G.add_edge((i,j), (period_map_i(i-1), period_map_j(j-1)), length=1, b_hat_0=-0.5, b_hat_1=-b,
                              periodic=is_periodic(i-1) or is_periodic(j-1),
                              a=np.array([a_x(i-1), a_y(j-1)]))
                else:
                    G.add_edge((i,j), (i, period_map_j(j+1)), length=1, b_hat_0=-0.5, b_hat_1=b,
                              periodic=is_periodic(j+1),
                              a=np.array([a_x(i), a_y(j+1)]))
                    G.add_edge((i,j), (i, period_map_j(j-1)), length=1, b_hat_0=-0.5, b_hat_1=-b,
                              periodic=is_periodic(j-1),
                              a=np.array([a_x(i), a_y(j-1)]))
                    G.add_edge((i,j), (period_map_i(i+1), period_map_j(j+1)), length=1, b_hat_0=0.5, b_hat_1=b,
                              periodic=is_periodic(i+1) or is_periodic(j+1),
                              a=np.array([a_x(i+1), a_y(j+1)]))
                    G.add_edge((i,j), (period_map_i(i+1), period_map_j(j-1)), length=1, b_hat_0=0.5, b_hat_1=-b,
                              periodic=is_periodic(i+1) or is_periodic(j-1),
                              a=np.array([a_x(i+1), a_y(j-1)]))

                # add node coordinates
                x = np.array([i + 0.5*np.mod(j,2), j*b])
                G.add_node((i, j), x=x)

                pos[(i, j)] = x

        # remove periodic edges that we don't want
        nodes_to_rem = [u for u in G.nodes if u[0] == -1 or u[1] == -1]
        G.remove_nodes_from(nodes_to_rem)

        remove_duplicate_digraph_edges(G)

        nodes = sorted(list(G.nodes))

        for i, nd in enumerate(nodes):
            G.node[nd]['idx'] = i

        G.graph['edgelist'] = list(G.edges)
        G.graph['nodelist'] = nodes
        G.graph['pos'] = pos
        G.graph['lengths'] = np.ones(G.number_of_edges())
        G.graph['name'] = 'triang'
        G.graph['dimension'] = 2
        G.graph['linear_length']= n

    def double_unit_cell(self, x):
        """ Double the unit cell of the periodic network and copy over stiffnesses.
        This only works if the network is periodic in both directions.
        Return the new network.
        """
        n = self.graph['linear_length']
        N = 2*n
        # Gg = triang_grid(N, x_periodic=True, y_periodic=True)
        Gg = TriangularGrid2D(n=N, x_periodic=True, y_periodic=True)

        X = np.zeros(Gg.number_of_edges())
        G_nond = nx.Graph(self)

        for i, (u, v) in enumerate(Gg.graph['edgelist']):
            # map to original graph
            l, m = u
            k, p = v
            ll = np.mod(l, n)
            mm = np.mod(m, n)
            kk = np.mod(k, n)
            pp = np.mod(p, n)

            e = ((ll, mm), (kk, pp))
            if e in self.graph['edgelist']:
                j = self.graph['edgelist'].index(e)
            elif e[::-1] in self.graph['edgelist']:
                j = self.graph['edgelist'].index(e[::-1])
            else:
                print('err.. periodic edge?')

            X[i] = x[j]

        return Gg, X

    def non_periodic(self, x):
        """ Return a non-periodic version of this triangular grid.
        Effectively, remove all periodic edges.
        Also copy over stiffnesses x and return appropriate new vector.
        """
        G = TriangularGrid2D(n=self.graph['linear_length'],
                              x_periodic=False, y_periodic=False)

        # copy over stiffnesses
        x_new = np.zeros(G.number_of_edges())
        new_el = G.graph['edgelist']
        for e, k in zip(self.graph['edgelist'], x):
            if e in new_el:
                x_new[new_el.index(e)] = k
            elif e[::-1] in new_el:
                x_new[new_el.index(e[::-1])] = k

        return G, x_new

    def shift_upper_half(self, x, s=1):
        """ Shift the stiffnesses x in the upper half of the
        periodic network by s, approximately preserving
        the local structure but disrupting the global structure.
        """
        n = self.graph['linear_length']
        k = int(n/2)

        new_edges = dict()

        for u, v in self.graph['edgelist']:
            u_x, u_y = u
            v_x, v_y = v

            if u_y >= k and v_y >= k:
                new_edges[(u, v)] = ((np.mod(u_x + s, n), u_y), (np.mod(v_x + s, n), v_y))
            else:
                new_edges[(u, v)] = (u, v)

        new_indices = np.zeros(len(x), dtype=int)

        for i, e in enumerate(self.graph['edgelist']):
            try:
                j = self.graph['edgelist'].index(new_edges[e])
            except:
                j = self.graph['edgelist'].index(new_edges[e][::-1])
            new_indices[i] = j

        return x[new_indices].copy()

    def shift(self, x, s_x, s_y):
        """ Shift the stiffnesses in the x direction s_x times and in the y
        direction s_y times
        """
        n = self.graph['linear_length']

        new_edges = dict()

        for u, v in self.graph['edgelist']:
            u_x, u_y = u
            v_x, v_y = v

            new_edges[(u, v)] = ((np.mod(u_x + s_x, n), np.mod(u_y + s_y, n)),
                                 (np.mod(v_x + s_x, n), np.mod(v_y + s_y, n)))

        new_indices = np.zeros(len(x), dtype=int)

        for i, e in enumerate(self.graph['edgelist']):
            try:
                j = self.graph['edgelist'].index(new_edges[e])
            except:
                j = self.graph['edgelist'].index(new_edges[e][::-1])
            new_indices[i] = j

        return x[new_indices].copy()

    def shift_lattice_inds(self, a, b):
        """ Shift stiffnesses x by a units in the direction of
        (1, 0) and by b units in the direction of (1/2, sqrt(3)/2).

        Return indices idx such that
        x_shifted = x[idx]
        """
        n = self.graph['linear_length']

        new_edges = dict()

        def b_shift(x, y):
            if np.mod(y, 2) == 0:
                return int(x + np.floor(0.5*b)), y + b
            else:
                return int(x + np.ceil(0.5*b)), y + b


        for u, v in self.graph['edgelist']:
            u_x, u_y = u
            v_x, v_y = v

            # determine how many steps to go
            uu_x, uu_y = b_shift(u_x, u_y)
            uu_x += a

            vv_x, vv_y = b_shift(v_x, v_y)
            vv_x += a

            new_edges[(u, v)] = ((np.mod(uu_x, n), np.mod(uu_y, n)),
                                 (np.mod(vv_x, n), np.mod(vv_y, n)))

        new_indices = np.zeros(self.number_of_edges(), dtype=int)

        for i, e in enumerate(self.graph['edgelist']):
            try:
                j = self.graph['edgelist'].index(new_edges[e])
            except:
                j = self.graph['edgelist'].index(new_edges[e][::-1])
            new_indices[i] = j

        return new_indices

class HoneycombGrid2D(MechanicalNetwork):
    """ Models a non-periodic honeycomb lattice
    """
    def __init__(self, n=5, m=5):
        MechanicalNetwork.__init__(self, dimension=2)

        self.graph['n'] = n
        self.graph['m'] = m

        self.generate_grid()

        self.init_attrs()

    def generate_grid(self):
        n = self.graph['n']
        m = self.graph['m']

        h = 1/2
        a = np.sqrt(3)/2

        # construct nodes
        offset = 0.0
        for i in range(m):
            for j in range(n):
                if np.mod(j, 2) == 0:
                    y = i
                elif np.mod(i, 2) == 0:
                    y = y - h
                else:
                    y = y + h

                x = j*a

                self.add_node((i, j), x=np.array([x, y+offset]))

            if np.mod(i, 2) == 1:
                offset += 1.0

        # construct horizontal edges
        for i in range(m):
            for j in range(n-1):
                self.add_edge((i,j), (i,j+1))

        # construct vertical edges
        for i in range(m):
            if np.mod(i, 2) == 0:
                for j in range(0, n, 2):
                    if (i+1, j) in self.nodes:
                        self.add_edge((i, j), (i+1, j))
            else:
                for j in range(1, n, 2):
                    if (i+1, j) in self.nodes:
                        self.add_edge((i, j), (i+1, j))

        # remove danglers
        nodes_to_rem = [n for n in self.nodes if self.degree(n) == 1]
        self.remove_nodes_from(nodes_to_rem)

if __name__ == '__main__':
    # save triangular grids in npz format
    grid = TriangularGrid2D(n=10, x_periodic=True, y_periodic=True)

    from time import time

    t1 = time()
    Q1 = grid.equilibrium_matrix()

    t2 = time()
    bh_0 = np.array([grid.edges[u,v]['b_hat_0'] for u, v in grid.graph['edgelist']])
    bh_1 = np.array([grid.edges[u,v]['b_hat_1'] for u, v in grid.graph['edgelist']])

    Q2 = grid.equilibrium_matrix_b_hats([bh_0, bh_1])

    t3 = time()
    print((Q1 - Q2).toarray())

    print(t2 - t1)
    print(t3 - t2)

    print((t2-t1)/(t3-t2))
