#!/usr/bin/env python3

#
# Contains helper functions for network refinement and bending elasticity
#

import numpy as np
import scipy as sp
import networkx as nx

from . import networks

def refined_network(netw):
    """ Return a refined mechanical network that has a node
    added to the midpoint of each edge.
    """
    edges = netw.graph['edgelist']
    new_edges = []

    dim = netw.graph['dimension']
    rnet = networks.MechanicalNetwork(dimension=dim)

    new_edges = []
    for u, v in edges:
        if netw.edges[u,v]['periodic']:
            x_u = netw.nodes[u]['x']
            x_v = netw.nodes[v]['x']

            b_hat = np.array([netw.edges[u,v][f'b_hat_{i}'] for i in range(dim)])
            length = netw.edges[u,v]['length']

            # construct midpoint starting from u
            w = (u, v)
            x_w = x_u + 0.5*length*b_hat

            rnet.add_node(u, x=x_u)
            rnet.add_node(v, x=x_v)
            rnet.add_node(w, x=x_w)

            rnet.add_edge(w, u)
            rnet.add_edge(w, v, periodic=True,
                         length=0.5*length,
                         **dict((f'b_hat_{i}', netw.edges[u,v][f'b_hat_{i}']) for i in range(dim)))

            new_edges.append([(w, u), (w, v)])
        else:
            x_u = netw.nodes[u]['x']
            x_v = netw.nodes[v]['x']

            w = (u, v)
            x_w = 0.5*(x_u + x_v)

            rnet.add_node(u, x=x_u)
            rnet.add_node(v, x=x_v)
            rnet.add_node(w, x=x_w)

            rnet.add_edge(w, u)
            rnet.add_edge(w, v)

            new_edges.append([(w, u), (w, v)])

    rnet.init_attrs()

    # find indices of original nodes in new network
    old_nodes = np.array([i for i, n in enumerate(rnet.graph['nodelist']) if n in netw.nodes()], dtype=int)
    midpoint_nodes = np.array([i for i, n in enumerate(rnet.graph['nodelist']) if n not in netw.nodes()], dtype=int)

    new_edges = [[rnet.graph['edgelist'].index(f) for f in e] for e in new_edges]

    # find edge indices for the two subedges of each full edge
    edge_inds  = []
    for n in midpoint_nodes:
        edge_inds.append([i for i, e in enumerate(rnet.graph['edge_ind_list'])
                    if n in e])

    edge_inds_a, edge_inds_b = zip(*edge_inds)

    rnet.graph['refined'] = True
    rnet.graph['old_node_inds'] = old_nodes
    rnet.graph['midpoint_node_inds'] = midpoint_nodes
    rnet.graph['new_edge_inds'] = new_edges
    rnet.graph['edge_inds_a'] = edge_inds_a
    rnet.graph['edge_inds_b'] = edge_inds_b

    return rnet

def bending_equilibrium_matrix2(netw, q=None):
    """ Construct the equilibrium matrix for bending
    of all nodes. This can be used to implement both bond
    bending and hinge stiffness.
    If q is not None, also Fourier transform at the given wavevector.
    """
    b_hats = netw.graph['b_hat']

    # rotate by 90 deg
    b_hats_rot = np.zeros_like(b_hats)
    b_hats_rot[:,0] = b_hats[:,1]
    b_hats_rot[:,1] = -b_hats[:,0]

    # corresponds to D vectors in the notes
    Q_rot = netw.equilibrium_matrix_b_hats(b_hats_rot)

    if q is not None:
        Q_rot = netw.equilibrium_matrix_ft(q, Q=Q_rot)

    # nodes
    degrees = []
    incident_edge_indices = []
    D_nodal = None
    D_is = []
    for i, n in enumerate(netw.graph['nodelist']):
        # find neighbors
        neighbors = list(netw.predecessors(n)) + list(netw.successors(n))
        degrees.append(len(neighbors))

        i_incident = []
        D_ijs = []
        for m in neighbors:
            j = netw.graph['nodelist'].index(m)

            if (n, m) in netw.graph['edgelist']:
                e = (n, m)
                orientation = 1
            else:
                e = (m, n)
                orientation = -1

            ei = netw.graph['edgelist'].index(e)
            i_incident.append(ei)

            # construct new D matrix
            D_ij = -orientation*Q_rot[:,ei]
            if D_nodal is None:
                D_nodal = D_ij
            else:
                D_nodal = sp.sparse.hstack([D_nodal, D_ij])

            D_ijs.append(D_ij)

        incident_edge_indices.append(i_incident)
        D_i = sp.sparse.csc_matrix(sp.sparse.hstack(D_ijs).sum(axis=1))
        D_is.append(D_i)

    D_i_nodal = sp.sparse.hstack(D_is)

    return D_nodal, D_i_nodal, np.array(degrees), incident_edge_indices

def bending_equilibrium_matrix(netw, q=None):
    """ Construct the equilibrium matrix for bending
    using mid_nodes, the node indices of the edge midpoints
    from the refined network.
    If q is not None, also Fourier transform at the given wavevector.
    """
    b_hats = netw.graph['b_hat']

    # rotate by 90 deg
    b_hats_rot = np.zeros_like(b_hats)
    b_hats_rot[:,0] = -b_hats[:,1]
    b_hats_rot[:,1] = b_hats[:,0]

    # corresponds to D vectors in the notes
    Q_rot = netw.equilibrium_matrix_b_hats(b_hats_rot)

    if q is not None:
        Q_rot = netw.equilibrium_matrix_ft(q, Q=Q_rot)

    edge_inds_a, edge_inds_b = netw.graph['edge_inds_a'], netw.graph['edge_inds_b']

    Q_a = Q_rot[:,edge_inds_a]
    Q_b = Q_rot[:,edge_inds_b]

    Q_sum = Q_a + Q_b

    return Q_rot, Q_sum
