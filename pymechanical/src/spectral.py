#!/usr/bin/env python3

"""
    spectral.py

    Contains classes to design the spectrum of mechanical networks
    using direct numerical optimization

  (c) Henrik Ronellenfitsch 2017
"""

import numpy as np
import scipy as sp
import scipy.optimize as optimize
import networkx as nx

import cvxpy as cvx

from scipy.spatial.distance import pdist, squareform

class PointMassOptimizer():
    """ Optimizes a number of band gaps at given mode indices
    for a network with equal point masses and massless springs.
    """
    def __init__(self, network, gaps):
        """ Initialize with given MechanicalNetwork.
        and band gap fractions
        """
        self.network = network
        self.gaps = gaps
        self.gap_inds = self.gaps_from_input(gaps)

    def gaps_from_input(self, gaps):
        indices = []

        for g in gaps:
            if g < 1:
                indices.append(int(g*self.network.graph['Q'].shape[0]))
            else:
                indices.append(g)

        return np.array(indices)

    def loss_function(self, x):
        """ Spectral loss function for a number of gaps at given
        mode indices in the spectrum
        """
        K = self.network.stiffness_matrix(x).toarray()
        u = sp.linalg.eigvalsh(K, turbo=True)

        repulsion = np.sum(1./(u[self.gap_inds] - u[self.gap_inds+1])**2)
        return repulsion

    def loss_jac(self, x):
        Q = self.network.graph['Q']
        K = self.network.stiffness_matrix(x).toarray()
        u, v = sp.linalg.eigh(K, turbo=True)

        residuals = -2./(u[self.gap_inds] - u[self.gap_inds+1])**3
        grads = Q.transpose().dot(v[:,self.gap_inds])**2 - \
                    Q.transpose().dot(v[:,self.gap_inds+1])**2

        return np.sum(grads*residuals, axis=1)

    def num_optim_dof(self):
        return self.network.graph['Q'].shape[1]

    def optimize(self, seed=None, lower_bound=0.1, upper_bound=1,
                 method='L-BFGS-B', callback=None, tol=1e-6, options={},
                 x0=None):
        if seed is not None:
            np.random.seed(seed)

        n = self.num_optim_dof()

        if x0 is None:
            x0 = lower_bound + (upper_bound - lower_bound)*np.random.random(n)

        # check which type of loss function we have
        if hasattr(self, 'loss_and_jac'):
            loss_func = self.loss_and_jac
            loss_jac = True
        else:
            loss_func = self.loss_function
            loss_jac = self.loss_jac

        ret = optimize.minimize(loss_func, x0, jac=loss_jac,
                        method=method,
                        bounds=(n*[(lower_bound, upper_bound)]), tol=tol,
                        callback=callback,
                        options=options)

        return (ret, x0)

    def optimize_adam(self, x0=None, lower_bound=0.1, upper_bound=1.0,
                      max_it=10000):
        """ Perform gradient descent with Adam adaptive learning rate.
        Adam is described here:
        https://arxiv.org/pdf/1412.6980.pdf
        """
        n = self.network.graph['Q'].shape[1]

        if x0 is None:
            x0 = lower_bound + (upper_bound - lower_bound)*np.random.random(n)

        x_cur = x0.copy()
        x_new = x0.copy()

        last_v = 0
        last_m = 0

        # algorithm parameters
        beta_1 = 0.9
        beta_2 = 0.999
        eps = 1e-8
        alpha = 0.005

        last_en = 0.0

        #Ks = []
        #ens = []
        for i in range(max_it):
            # en, grad = optfunc(x_cur, *args)
            en = self.loss_function(x_cur)
            grad = self.loss_jac(x_cur)

            # project gradient
            over = (x_cur >= upper_bound) & (grad < 0)
            under = (x_cur <= lower_bound) & (grad > 0)

            grad[over] = 0.0
            grad[under] = 0.0

            # ADAM step
            m = beta_1*last_m + (1 - beta_1)*grad
            v = beta_2*last_v + (1 - beta_2)*grad**2

            m_hat = m/(1 - beta_1**(i+1))
            v_hat = v/(1 - beta_2**(i+1))

            #step = np.clip(alpha/(np.sqrt(v_hat) + eps)*m_hat, -alpha, alpha)
            step = alpha/(np.sqrt(v_hat) + eps)*m_hat
            x_new = np.clip(x_cur - step, lower_bound, upper_bound)

            # print('en', en)
            # print('grad', np.max(np.abs(grad)))
            #print('argmax', np.argmax(np.abs(grad)))
            # break if energy is zero or projected gradient
            # is zero
            if np.max(np.abs(grad)) < 1e-6 or last_en < en:
                break
            x_cur = x_new

            last_v = v
            last_m = m

            last_en = en

        ret = optimize.OptimizeResult()
        ret.fun = en
        ret.message = 'success' if (i < max_it - 1) else 'fail'
        ret.x = x_cur
        ret.nit = i + 1
        ret.nfev = ret.nit
        ret.success = i < max_it - 1
        ret.jac = grad

        return ret, x0

    def basinhop(self, lower_bound=0.1, upper_bound=1, iterations=100):
        """ Use simulated annealing to find better optimum
        """
        def cb(x, en, accept):
            print(en)

        n = self.network.graph['Q'].shape[1]
        x0 = lower_bound + (upper_bound - lower_bound)*np.random.random(n)

        minimizer_kwargs = {"method":"L-BFGS-B", "jac":self.loss_jac, "tol":1e-6,
                            "bounds":(n*[(lower_bound, upper_bound)])}
        ret = optimize.basinhopping(self.loss_function, x0, stepsize=0.1, niter=iterations,
                                    minimizer_kwargs=minimizer_kwargs,
                                    take_step=BoundedRandomDisplacement(lower_bound=lower_bound,
                                                                        upper_bound=upper_bound),
                                    callback=cb)
        return ret, x0

class PointMassNodePosOptimizerMid(PointMassOptimizer):
    """ Models the optimization of the entire network including node positions
    """

    loss_jac = None

    def loss_function(self, x, return_data=False):
        """ Spectral loss function for a number of gaps at given
        mode indices in the spectrum
        """
        # unpack variables
        d = self.network.graph['dimension']
        n = self.network.number_of_nodes()
        E = self.network.graph['E']

        node_pos = x[:d*n].reshape((d, n)).T
        k = x[d*n:]

        # compute new b_hat
        b_hat = -E.transpose().dot(node_pos)
        b_hat /= np.linalg.norm(b_hat, axis=1)[:,np.newaxis]

        # new stiffness matrix from new positions
        Q = self.network.equilibrium_matrix_b_hats(b_hat.T)
        K = Q.dot(sp.sparse.diags(k)).dot(Q.transpose())

        u = sp.linalg.eigvalsh(K.toarray(), turbo=True)

        repulsion = np.sum((u[self.gap_inds] - u[self.gap_inds+1])/(u[self.gap_inds] + u[self.gap_inds+1]))

        if return_data:
            return repulsion, Q, K, b_hat
        else:
            return repulsion

    def optimize(self, seed=None, lower_bound=0.1, upper_bound=1,
                 method='L-BFGS-B', callback=None, tol=1e-6, options={},
                 x0=None):
        if seed is not None:
            np.random.seed(seed)

        n = self.network.graph['Q'].shape[1]
        m = self.network.graph['Q'].shape[0]
        # d = self.network.graph['dimension']

        pos = np.array([self.network.node[nd]['x'] for nd in self.network.graph['nodelist']])
        if x0 is None:
            k0 = lower_bound + (upper_bound - lower_bound)*np.random.random(n)
            pos0 = pos.T.flatten()

            x0 = np.concatenate((pos0, k0))
            print(x0.shape)
            print(n, m)

        # check which type of loss function we have
        if hasattr(self, 'loss_and_jac'):
            loss_func = self.loss_and_jac
            loss_jac = False
        else:
            loss_func = self.loss_function
            loss_jac = self.loss_jac

        def cb(x):
            print(self.loss_function(x))

        # find spatial constraints
        nodes = self.network.number_of_nodes()
        pos_lower_limits = []
        pos_upper_limits = []

        for i in range(self.network.graph['dimension']):
            mmin, mmax = np.min(pos[:,i]), np.max(pos[:,i])

            # shift and center, max box 2% larger to avoid
            # points sitting exactly on the constraint surface
            le = mmax - mmin
            mmin = mmin - 0.01*le
            mmax = mmax + 0.01*le

            pos_lower_limits.extend(nodes*[mmin])
            pos_upper_limits.extend(nodes*[mmax])

            print(mmin, mmax)

        pos_limits = list(zip(pos_lower_limits, pos_upper_limits))
        # pos_limits = list(zip(*pos_limits))

        ret = optimize.minimize(loss_func, x0, jac=loss_jac,
                        method=method,
                        bounds=(pos_limits + n*[(lower_bound, upper_bound)]), tol=tol,
                        callback=cb,
                        options=options)

        return (ret, x0)

    def unpack_result(self, x):
        """ Take the result of an optimization and return the stiffnesses and
        positions.
        """
        d = self.network.graph['dimension']
        n = self.network.number_of_nodes()

        node_pos = x[:d*n].T.reshape((d, n)).T
        k = x[d*n:]

        return node_pos, k

class PointMassOptimizerRel(PointMassOptimizer):
    """ Optimize the relative band gaps (omega_j - omega_i)/omega_i
    """
    def loss_function(self, x):
        """ Spectral loss function for a number of gaps at given
        mode indices in the spectrum
        """
        K = self.network.stiffness_matrix(x).toarray()
        u = sp.linalg.eigvalsh(K, turbo=True)

        repulsion = np.sum((u[self.gap_inds] - u[self.gap_inds+1])/u[self.gap_inds])

        return repulsion

    def loss_jac(self, x):
        Q = self.network.graph['Q']
        K = self.network.stiffness_matrix(x).toarray()
        u, v = sp.linalg.eigh(K, turbo=True)

        # residuals = -2./(u[self.gap_inds] - u[self.gap_inds+1])**3
        grads_i = Q.transpose().dot(v[:,self.gap_inds])**2
        grads_j = Q.transpose().dot(v[:,self.gap_inds+1])**2

        u_i = u[self.gap_inds]
        u_j = u[self.gap_inds+1]

        return np.sum((u_j*grads_i - u_i*grads_j)/u_i**2, axis=1)




class PointMassOptimizerMid(PointMassOptimizer):
    """ Optimize the relative band gaps (omega_j - omega_i)/(omega_i + omega_j)
    """
    def loss_and_jac(self, x):
        Q = self.network.graph['Q']
        K = self.network.stiffness_matrix(x).toarray()
        u, v = sp.linalg.eigh(K, turbo=True)

        # loss function
        loss = np.sum((u[self.gap_inds] - u[self.gap_inds+1])/(u[self.gap_inds] + u[self.gap_inds+1]))

        # gradient
        grads_i = Q.transpose().dot(v[:,self.gap_inds])**2
        grads_j = Q.transpose().dot(v[:,self.gap_inds+1])**2

        u_i = u[self.gap_inds]
        u_j = u[self.gap_inds+1]

        return loss, 2*np.sum((u_j*grads_i - u_i*grads_j)/(u_i + u_j)**2, axis=1)

    def optimize_sdp_single(self, seed=None, lower_bound=0.1, upper_bound=1,
                     callback=None, solver=cvx.MOSEK, x0=None,
                     frac_states=0.3, maxiter=200, tol=1e-4,
                     solver_options=None, verbose=False):
        """ Use the sequential semidefinite programming algorithm
        from Freund et al. to optimize a single band gap.

        We use fractional linear programming to deal with the
        scale invariance of the problem.

        The fractional objective is:

        Maximize (lambda_u - lambda_l)/(lambda_u + lambda_l)

        it will be converted to a linear program and then solved.

        """
        # Implement a subspace optimization method
        QT = self.network.graph['Q'].transpose()
        n = QT.shape[0]
        n_eigvals = QT.shape[1]

        # number of states above and below the gap
        n_states = int(frac_states*QT.shape[1])

        assert(len(self.gap_inds) == 1)

        gap_ind = self.gap_inds[0]

        # make sure we do not over or underflow
        above_lim = min(gap_ind + 1 + n_states, n_eigvals)
        below_lim = max(gap_ind - n_states + 1, 0)

        n_above_states = above_lim - gap_ind - 1
        n_below_states = gap_ind + 1 - below_lim

        # this many states are added in addition to the ones near the gap
        n_above_extra = 0

        # set up the problem
        def setup_problem(n_states=20):
            # construct fractional semidefinite program
            lambda_u = cvx.Variable()
            lambda_l = cvx.Variable()

            t = cvx.Variable()

            k = cvx.Variable(n)

            k_diag = cvx.diag(k)

            # for each gap consider the space above and below.
            # TODO: CHECK that n_states does not over/underflow
            Q_above = cvx.Parameter(n, n_above_states + n_above_extra)
            Q_below = cvx.Parameter(n, n_below_states)

            # * is matrix multiplication
            K_above = Q_above.T*k_diag*Q_above - lambda_u*np.eye(n_above_states + n_above_extra)
            K_below = Q_below.T*k_diag*Q_below - lambda_l*np.eye(n_below_states)

            objective = cvx.Maximize(lambda_u - lambda_l)

            constraints = [k >= lower_bound*t,
                           k <= upper_bound*t,
                           lambda_u >= 0.0,
                           lambda_l + lambda_u == 1.0,
                           t >= 0.0,
                           K_above >> 0,
                           K_below << 0]

            prob = cvx.Problem(objective, constraints)

            return prob, Q_above, Q_below, k, lambda_u, lambda_l, t

        prob, Q_above, Q_below, k, lambda_u, lambda_l, t = setup_problem(n_states=n_states)

        # setup starting point
        if x0 is None:
            x0 = lower_bound + (upper_bound - lower_bound)*np.random.random(n)

        if verbose:
            print("states:", below_lim, above_lim, "+", n_above_extra)
            print("i\tobjective\tSDP\t|k_{n+1} - k_n|\tt")

        # run iterative solver
        x_cur = x0.copy()
        x_norm = 0
        j = 0
        for i in range(maxiter):
            # compute current spectrum for rescaled variables
            K = self.network.stiffness_matrix(x_cur).toarray()
            u, v = sp.linalg.eigh(K, turbo=True)

            # construct subspaces above and below eigengap(s)
            v_above = v[:,gap_ind+1:above_lim]
            v_below = v[:,below_lim:gap_ind+1]

            # # add extra states
            extra_inds = np.linspace(above_lim+1, v.shape[1] - 1, n_above_extra, dtype=int)

            v_above = np.hstack((v_above, v[:,extra_inds]))

            # construct subspace projected stiffness matrices
            Q_above.value = QT.dot(v_above)
            Q_below.value = QT.dot(v_below)

            # use SCS for faster but less accurate solutions
            if solver_options is not None:
                prob.solve(solver=solver, **solver_options)
            else:
                prob.solve(solver=solver)

            # Check convergence conditions
            x_new = np.array(k.value).flatten()/t.value

            x_norm = np.linalg.norm(x_cur - x_new)
            x_cur = x_new.copy()

            j += 1

            # callback if desired
            if callback is not None:
                callback(x_cur, prob)

            objective = (u[gap_ind+1] - u[gap_ind])/(u[gap_ind+1] + u[gap_ind])

            if verbose:
                print("{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}".format(j + 1, objective, prob.value, x_norm, t.value))

            if x_norm < tol:
                break

        ret = optimize.OptimizeResult()
        ret.fun = objective
        ret.fun_sdp = prob.value
        ret.message = 'SUCCESS: |k_n - k_{n+1}| < tol.' if (j < maxiter - 1) else 'FAIL: iteration limit exceeded.'
        ret.x = x_cur
        ret.nit = j + 1
        ret.nfev = j + 1
        ret.success = j < maxiter - 1
        ret.jac = x_norm.copy()
        ret.n_states = n_states

        if verbose:
            print(ret.message)

        return ret, x0

    def optimize_sdp_multi(self, seed=None, lower_bound=0.1, upper_bound=1,
                     callback=None, solver=cvx.CVXOPT, x0=None,
                     frac_states=0.3, maxiter=200, tol=1e-4,
                     solver_options=None, verbose=False):
        """ Use the sequential semidefinite programming algorithm
        from Freund et al. to optimize the band gap(s).
        The objective is

        Max( min_i (lambda_i^up - lambda_i^l)/(lambda_i^l) )

        such that the eigenvalues are max and min eigenvalues of the
        surrounding eigenspaces.

        Technically, a tradeoff parameter can be introduced but
        we consider all eigengaps equally.

        In contrast to Freund, we do not include the midgap in the problem
        and solve a fractional linear program but instead take the midgap
        as a fixed parameter at each iteration.
        """
        # Implement a subspace optimization method
        QT = self.network.graph['Q'].transpose()
        n = QT.shape[0]
        n_eigvals = QT.shape[1]

        # number of states above and below the gap
        n_states = int(frac_states*QT.shape[1])

        # make sure we do not over or underflow
        above_lims = []
        below_lims = []

        for gap_ind in self.gap_inds:
            above_lims.append(min(gap_ind + 1 + n_states, n_eigvals))
            below_lims.append(max(gap_ind - n_states + 1, 0))

        # set up the problem
        def setup_problem(n_states=20):
            # construct semidefinite program
            lambda_u = cvx.Variable(len(self.gap_inds))
            lambda_l = cvx.Variable(len(self.gap_inds))

            k = cvx.Variable(n)

            k_diag = cvx.diag(k)

            # for each gap consider the space above and below.
            # TODO: CHECK that n_states does not over/underflow
            Q_aboves = [cvx.Parameter(n, above_lims[i] - self.gap_inds[i] - 1)
                        for i in range(len(self.gap_inds))]
            Q_belows = [cvx.Parameter(n, self.gap_inds[i] + 1 - below_lims[i])
                        for i in range(len(self.gap_inds))]

            # * is matrix multiplication
            K_aboves = [Q_above.T*k_diag*Q_above - lambda_u[i]*np.eye(above_lims[i] - self.gap_inds[i] - 1)
                        for i, Q_above in enumerate(Q_aboves)]
            K_belows = [Q_below.T*k_diag*Q_below - lambda_l[i]*np.eye(self.gap_inds[i] + 1 - below_lims[i])
                        for i, Q_below in enumerate(Q_belows)]

            midgaps = cvx.Parameter(len(self.gap_inds))
            objective = cvx.Maximize(midgaps.T*(lambda_u - lambda_l))

            constraints = [k >= lower_bound,
                           k <= upper_bound,
                           lambda_u >= 0,
                           lambda_l >= 0] \
                          + [K_above >> 0 for K_above in K_aboves] \
                          + [K_below << 0 for K_below in K_belows]

            prob = cvx.Problem(objective, constraints)

            return prob, Q_aboves, Q_belows, midgaps, k, lambda_u, lambda_l

        prob, Q_aboves, Q_belows, midgaps, k, lambda_u, lambda_l = setup_problem(n_states=n_states)

        # setup starting point
        if x0 is None:
            x0 = lower_bound + (upper_bound - lower_bound)*np.random.random(n)

        if verbose:
            print("objective\t\t |k_{n+1} - k_n|")

        # run iterative solver
        x_cur = x0.copy()
        x_norm = 0
        j = 0
        for i in range(maxiter):
            # compute current spectrum
            K = self.network.stiffness_matrix(x_cur).toarray()
            u, v = sp.linalg.eigh(K, turbo=True)

            # construct subspaces above and below eigengap(s)
            for i, gap_ind in enumerate(self.gap_inds):
                v_above = v[:,gap_ind+1:above_lims[i]]
                v_below = v[:,below_lims[i]:gap_ind+1]

                # construct subspace projected stiffness matrices
                Q_aboves[i].value = QT.dot(v_above)
                Q_belows[i].value = QT.dot(v_below)

            midgaps.value = 1./(u[self.gap_inds] + u[self.gap_inds+1])

            # use SCS for faster but less accurate solutions
            if solver_options is not None:
                prob.solve(solver=solver, **solver_options)
            else:
                prob.solve(solver=solver)

            print(prob.value)

            # Check convergence conditions
            x_new = np.array(k.value).flatten()

            x_norm = np.linalg.norm(x_cur - x_new)
            x_cur = x_new.copy()

            j += 1

            # callback if desired
            if callback is not None:
                callback(x_cur, prob)

            if verbose:
                print(prob.value, "\t", x_norm)

            if x_norm < tol:
                break

        ret = optimize.OptimizeResult()
        ret.fun = prob.value
        ret.message = 'SUCCESS: |k_n - k_{n+1}| < tol.' if (j < maxiter - 1) else 'FAIL: iteration limit exceeded.'
        ret.x = x_cur
        ret.nit = j + 1
        ret.nfev = j + 1
        ret.success = j < maxiter - 1
        ret.jac = x_norm.copy()
        ret.n_states = n_states

        if verbose:
            print(ret.message)

        return ret, x0

class PointMassOptimizerAbs(PointMassOptimizer):
    """ Optimize the absolute band gaps (omega_j - omega_i)
    """
    def loss_function(self, x):
        """ Spectral loss function for a number of gaps at given
        mode indices in the spectrum
        """
        K = self.network.stiffness_matrix(x).toarray()
        u = sp.linalg.eigvalsh(K, turbo=True)

        repulsion = np.sum(u[self.gap_inds] - u[self.gap_inds+1])
        # print(repulsion)
        return repulsion

    def loss_jac(self, x):
        Q = self.network.graph['Q']
        K = self.network.stiffness_matrix(x).toarray()
        u, v = sp.linalg.eigh(K, turbo=True)

        # residuals = -2./(u[self.gap_inds] - u[self.gap_inds+1])**3
        grads_i = Q.transpose().dot(v[:,self.gap_inds])**2
        grads_j = Q.transpose().dot(v[:,self.gap_inds+1])**2

        return np.sum(grads_i - grads_j, axis=1)

class BoundedRandomDisplacement():
    """
    Add a random displacement of maximum size `stepsize` to each coordinate
    Calling this updates `x` in-place.
    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    random_state : None or `np.random.RandomState` instance, optional
        The random number generator that generates the displacements
    """
    def __init__(self, stepsize=0.5, lower_bound=0.1, upper_bound=1.0):
        self.stepsize = stepsize
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, x):
        x += np.random.uniform(-self.stepsize, self.stepsize,
                                       np.shape(x))
        return np.clip(x, self.lower_bound, self.upper_bound)

class ZeroModeOptimizer(PointMassOptimizer):
    """ Optimizes additional zero modes into the network
    """
    def __init__(self, network, mode_id):
        """ Takes a network, the id of the mode we want to set to zero,
        """
        self.network = network
        self.mode_id = mode_id

    def loss_function(self, x):
        """ Spectral loss function for a number of gaps at given
        mode indices in the spectrum
        """
        K = self.network.stiffness_matrix(x)#.toarray()
        # u = sp.linalg.eigvalsh(K, turbo=True)
        u = sp.sparse.linalg.eigsh(K, k=self.mode_id+1, which='SM', return_eigenvectors=False)

        repulsion = u[0]
        return repulsion

    def loss_jac(self, x):
        Q = self.network.graph['Q']
        K = self.network.stiffness_matrix(x)
        # u, v = sp.linalg.eigh(K, turbo=True)
        u, v = sp.sparse.linalg.eigsh(K, k=self.mode_id+1, which='SM', return_eigenvectors=True)

        grad = Q.transpose().dot(v[:,self.mode_id])**2

        return grad

class PinnedZeroModeOptimizer(PointMassOptimizer):
    """ Optimizes additional zero modes into the network,
    where a number of nodes have been pinned
    """
    def __init__(self, network, mode_id, pinned_nodes):
        """ Takes a network, the id of the mode we want to set to zero,
        """
        self.network = network
        self.mode_id = mode_id
        self.pinned_nodes = pinned_nodes

        d = self.network.graph['dimension']
        n = self.network.number_of_nodes()
        self.free_idx = np.array([i for i in range(n*d)
                                  if np.mod(i, n) not in pinned_nodes])

    def loss_function(self, x):
        """ Spectral loss function for a number of gaps at given
        mode indices in the spectrum
        """
        K = self.network.stiffness_matrix(x)[:,self.free_idx][self.free_idx,:]

        # u = sp.linalg.eigvalsh(K, turbo=True)
        u = sp.sparse.linalg.eigsh(K, k=self.mode_id+1, which='SM', return_eigenvectors=False)

        repulsion = u[self.mode_id] #+ 1e-3/u[self.mode_id + 1]
        return repulsion

    def loss_jac(self, x):
        Q = self.network.graph['Q'].tocsc()[self.free_idx,:]
        K = self.network.stiffness_matrix(x)[:,self.free_idx][self.free_idx,:]
        # u, v = sp.linalg.eigh(K, turbo=True)
        u, v = sp.sparse.linalg.eigsh(K, k=self.mode_id+1, which='SM', return_eigenvectors=True)

        grad = Q.transpose().dot(v[:,self.mode_id])**2 #- 1e-3*Q.transpose().dot(v[:,self.mode_id+1])**2/u[self.mode_id+1]**2

        return grad

class PointMassBrillouinOptimizer(PointMassOptimizer):
    """ Optimizes the given points in the Brillouin zone of
    the periodic network instead of just the Gamma point.
    """
    def __init__(self, network, gaps, qs):
        super().__init__(network, gaps)
        self.qs = qs

        # compute and cache equilibrium_matrices at the needed
        # values of q
        self.Qs = [self.network.equilibrium_matrix_ft(q).tocsc() for q in qs]
        self.QHs = [Q.conj().transpose() for Q in self.Qs]

    def loss_function_q(self, x, q_idx):
        """ Spectral loss function for a number of gaps at given
        mode indices in the spectrum and given index of the wave vector
        """
        # K = self.network.stiffness_matrix(x, q=q).toarray()
        Q = self.Qs[q_idx]
        QH = self.QHs[q_idx]
        K = Q.dot(sp.sparse.diags(x)).dot(QH).toarray()

        u = sp.linalg.eigvalsh(K, turbo=True)

        repulsion = np.sum(1./(u[self.gap_inds] - u[self.gap_inds+1])**2)
        return repulsion

    def loss_jac_q(self, x, q_idx):
        # Q = self.network.equilibrium_matrix_ft(q)
        # K = self.network.stiffness_matrix(x, q).toarray()

        Q = self.Qs[q_idx]
        QH = self.QHs[q_idx]
        K = Q.dot(sp.sparse.diags(x)).dot(QH).toarray()

        u, v = sp.linalg.eigh(K, turbo=True)

        residuals = -2./(u[self.gap_inds] - u[self.gap_inds+1])**3
        grads = np.abs(QH.dot(v[:,self.gap_inds]))**2 - \
                    np.abs(QH.dot(v[:,self.gap_inds+1]))**2

        return np.sum(grads*residuals, axis=1)

    def loss_function(self, x):
        loss = np.sum([self.loss_function_q(x, q_idx) for q_idx in range(len(self.qs))], dtype=float)
        return loss

    def loss_jac(self, x):
        # print([self.loss_jac_q(x, q) for q in self.qs])
        return np.sum([self.loss_jac_q(x, q_idx) for q_idx in range(len(self.qs))], axis=0, dtype=float)

class PointMassBrillouinOptimizerRel(PointMassBrillouinOptimizer):
    """ Optimizes the relative loss function in the brillouin zone.
    """
    def loss_function_q(self, x, q_idx):
        """ Spectral loss function for a number of gaps at given
        mode indices in the spectrum and given index of the wave vector
        """
        # K = self.network.stiffness_matrix(x, q=q).toarray()
        Q = self.Qs[q_idx]
        QH = self.QHs[q_idx]
        K = Q.dot(sp.sparse.diags(x)).dot(QH).toarray()

        u = sp.linalg.eigvalsh(K, turbo=True)

        repulsion = np.sum((u[self.gap_inds] - u[self.gap_inds+1])/u[self.gap_inds])
        return repulsion

    def loss_jac_q(self, x, q_idx):
        # Q = self.network.equilibrium_matrix_ft(q)
        # K = self.network.stiffness_matrix(x, q).toarray()

        Q = self.Qs[q_idx]
        QH = self.QHs[q_idx]
        K = Q.dot(sp.sparse.diags(x)).dot(QH).toarray()

        u, v = sp.linalg.eigh(K, turbo=True)

        grads_i = np.abs(QH.dot(v[:,self.gap_inds]))**2
        grads_j = np.abs(QH.dot(v[:,self.gap_inds+1]))**2

        u_i = u[self.gap_inds]
        u_j = u[self.gap_inds+1]

        return np.sum((u_j*grads_i - u_i*grads_j)/u_i**2, axis=1)

class PointMassBrillouinOptimizerMid(PointMassBrillouinOptimizer):
    """ Optimizes the gap-midgap loss function in the brillouin zone.
    """
    def loss_and_jac_q(self, x, q_idx):
        Q = self.Qs[q_idx]
        QH = self.QHs[q_idx]
        K = Q.dot(sp.sparse.diags(x)).dot(QH).toarray()

        u, v = sp.linalg.eigh(K, turbo=True)

        # loss function
        loss = np.sum((u[self.gap_inds] - u[self.gap_inds+1])/(u[self.gap_inds] + u[self.gap_inds+1]))

        grads_i = np.abs(QH.dot(v[:,self.gap_inds]))**2
        grads_j = np.abs(QH.dot(v[:,self.gap_inds+1]))**2

        u_i = u[self.gap_inds]
        u_j = u[self.gap_inds+1]

        return loss, np.sum(2*(u_j*grads_i - u_i*grads_j)/(u_i + u_j)**2, axis=1)

    def loss_and_jac(self, x):
        losses, jacs = zip(*[self.loss_and_jac_q(x, q_idx) for q_idx in range(len(self.qs))])

        return np.sum(losses), np.sum(jacs, axis=0)

    def optimize_sdp_single(self, seed=None, lower_bound=0.1, upper_bound=1,
                     callback=None, solver=cvx.CVXOPT, x0=None,
                     frac_states=0.3, maxiter=200, tol=1e-4,
                     solver_options=None, verbose=False):
        """ Use the sequential semidefinite programming algorithm
        from Freund et al. to optimize the band gap(s).
        The objective is

        Max( sum_i (lambda_i^up - lambda_i^l)/(lambda_i^l) )

        such that the eigenvalues are max and min eigenvalues of the
        surrounding eigenspaces.

        The eigenspaces are chosen at the desired values of the wavevector q.
        """
        # Implement a subspace optimization method
        n = self.QHs[0].shape[0]
        n_eigvals = self.QHs[0].shape[1]

        # number of states above and below the gap
        n_states = int(frac_states*n_eigvals)

        # make sure we do not over or underflow
        above_lims = []
        below_lims = []

        gap_ind = self.gap_inds[0]

        for q_val in self.qs:
            above_lims.append(min(gap_ind + 1 + n_states, n_eigvals))
            below_lims.append(max(gap_ind - n_states + 1, 0))

        # set up the problem
        def setup_problem(n_states=20):
            # construct semidefinite program
            lambda_u = cvx.Variable()#len(self.qs))
            lambda_l = cvx.Variable()#len(self.qs))

            t = cvx.Variable()

            k = cvx.Variable(n)

            k_diag = cvx.diag(k)

            # for each gap consider the space above and below.
            # TODO: CHECK that n_states does not over/underflow
            Q_aboves_Re = [cvx.Parameter(n, above_lims[i] - gap_ind - 1)
                        for i in range(len(self.qs))]
            Q_belows_Re = [cvx.Parameter(n, gap_ind + 1 - below_lims[i])
                        for i in range(len(self.qs))]

            Q_aboves_Im = [cvx.Parameter(n, above_lims[i] - gap_ind - 1)
                        for i in range(len(self.qs))]
            Q_belows_Im = [cvx.Parameter(n, gap_ind + 1 - below_lims[i])
                        for i in range(len(self.qs))]

            # * is matrix multiplication
            # must deal with real and complex parts seperately
            K_aboves_Re = [Q_above_Re.T*k_diag*Q_above_Re + Q_above_Im.T*k_diag*Q_above_Im - lambda_u*np.eye(above_lims[i] - gap_ind - 1)
                        for i, (Q_above_Re, Q_above_Im) in enumerate(zip(Q_aboves_Re, Q_aboves_Im))]
            K_aboves_Im = [Q_above_Im.T*k_diag*Q_above_Re - Q_above_Re.T*k_diag*Q_above_Im
                        for i, (Q_above_Re, Q_above_Im) in enumerate(zip(Q_aboves_Re, Q_aboves_Im))]

            K_belows_Re = [Q_below_Re.T*k_diag*Q_below_Re + Q_below_Im.T*k_diag*Q_below_Im - lambda_l*np.eye(-below_lims[i] + gap_ind + 1)
                        for i, (Q_below_Re, Q_below_Im) in enumerate(zip(Q_belows_Re, Q_belows_Im))]
            K_belows_Im = [Q_below_Im.T*k_diag*Q_below_Re - Q_below_Re.T*k_diag*Q_below_Im
                        for i, (Q_below_Re, Q_below_Im) in enumerate(zip(Q_belows_Re, Q_belows_Im))]

            # Construct combined matrices with equivalent semidef properties
            K_aboves = [cvx.vstack(cvx.hstack(K_Re, -K_Im), cvx.hstack(K_Im, K_Re))
                        for K_Re, K_Im in zip(K_aboves_Re, K_aboves_Im)]
            K_belows = [cvx.vstack(cvx.hstack(K_Re, -K_Im), cvx.hstack(K_Im, K_Re))
                        for K_Re, K_Im in zip(K_belows_Re, K_belows_Im)]

            midgaps = cvx.Parameter(len(self.qs))
            objective = cvx.Maximize((lambda_u - lambda_l))

            constraints = [k >= lower_bound*t,
                           k <= upper_bound*t,
                           t >= 0,
                           lambda_u + lambda_l == 1,
                           lambda_u >= 0,
                           lambda_l >= 0] \
                          + [K_above >> 0 for K_above in K_aboves] \
                          + [K_below << 0 for K_below in K_belows]

            prob = cvx.Problem(objective, constraints)

            return prob, Q_aboves_Re, Q_aboves_Im, Q_belows_Re, Q_belows_Im, midgaps, k, lambda_u, lambda_l, t

        prob, Q_aboves_Re, Q_aboves_Im, Q_belows_Re, Q_belows_Im, midgaps, k, lambda_u, lambda_l, t = setup_problem(n_states=n_states)

        # setup starting point
        if x0 is None:
            x0 = lower_bound + (upper_bound - lower_bound)*np.random.random(n)

        if verbose:
            print("objective\t\t |k_{n+1} - k_n|")

        # run iterative solver
        x_cur = x0.copy()
        x_norm = 0
        j = 0
        for i in range(maxiter):
            # construct subspaces above and below eigengap
            # for the different q vectors
            midgaps_cur = np.zeros(len(self.qs))
            for i, (q, Q, QH) in enumerate(zip(self.qs, self.Qs, self.QHs)):
                K = Q.dot(sp.sparse.diags(x_cur)).dot(QH).toarray()
                u, v = sp.linalg.eigh(K, turbo=True)

                v_above = v[:,gap_ind+1:above_lims[i]]
                v_below = v[:,below_lims[i]:gap_ind+1]

                # construct subspace projected stiffness matrices
                Q_abov = self.QHs[i].dot(v_above)
                Q_belo = self.QHs[i].dot(v_below)

                Q_aboves_Re[i].value = Q_abov.real
                Q_belows_Re[i].value = Q_belo.real

                Q_aboves_Im[i].value = Q_abov.imag
                Q_belows_Im[i].value = Q_belo.imag

                midgaps_cur[i] = 1./(u[gap_ind] + u[gap_ind+1])

            midgaps.value = midgaps_cur

            # use SCS for faster but less accurate solutions
            if solver_options is not None:
                prob.solve(solver=solver, **solver_options)
            else:
                prob.solve(solver=solver)

            # Check convergence conditions
            x_new = np.array(k.value).flatten()/t.value

            x_norm = np.linalg.norm(x_cur - x_new)
            x_cur = x_new.copy()

            j += 1

            # callback if desired
            if callback is not None:
                callback(x_cur, prob)

            if verbose:
                print(prob.value, "\t", x_norm)

            if x_norm < tol:
                break

        ret = optimize.OptimizeResult()
        ret.fun = prob.value
        ret.message = 'SUCCESS: |k_n - k_{n+1}| < tol.' if (j < maxiter - 1) else 'FAIL: iteration limit exceeded.'
        ret.x = x_cur
        ret.nit = j + 1
        ret.nfev = j + 1
        ret.success = j < maxiter - 1
        ret.jac = x_norm.copy()
        ret.n_states = n_states

        if verbose:
            print(ret.message)

        return ret, x0


class ElectricalBrillouinOptimizerMid(PointMassOptimizer):
    """ Optimizes the gap-midgap loss function in the brillouin zone,
    but uses the network as an electrical network, with a regular graph
    Laplacian encoding capacitively grounded nodes, connected by inductances.
    """
    def __init__(self, network, gaps, qs):
        super().__init__(network, gaps)
        self.qs = qs

        # compute and cache equilibrium_matrices at the needed
        # values of q
        E = self.network.graph['E']
        self.Es = [self.network.incidence_matrix_ft(q).tocsc() for q in qs]
        self.EHs = [E.conj().transpose() for E in self.Es]

    def num_optim_dof(self):
        return 2*self.network.graph['Q'].shape[1]

    def loss_and_jac_q(self, x, q_idx):
        E = self.Es[q_idx]
        EH = self.EHs[q_idx]

        L = x[:E.shape[1]] # inductances
        C = x[E.shape[1]:] # capacitances
        ground = 1.0

        K = E.dot(sp.sparse.diags(L)).dot(EH).toarray()

        # ground capacitively
        M = (E.dot(sp.sparse.diags(C)).dot(EH) + ground*sp.sparse.eye(E.shape[0])).toarray()

        u, v = sp.linalg.eigh(K, M, turbo=True)

        # loss function
        loss = np.sum((u[self.gap_inds] - u[self.gap_inds+1])/(u[self.gap_inds] + u[self.gap_inds+1]))

        u_i = u[self.gap_inds]
        u_j = u[self.gap_inds+1]

        grads_i = np.concatenate((np.abs(EH.dot(v[:,self.gap_inds]))**2, -u_i*(np.abs(EH.dot(v[:,self.gap_inds]))**2)))
        grads_j = np.concatenate((np.abs(EH.dot(v[:,self.gap_inds+1]))**2, -u_j*(np.abs(EH.dot(v[:,self.gap_inds+1]))**2)))

        return loss, np.sum(2*(u_j*grads_i - u_i*grads_j)/(u_i + u_j)**2, axis=1)

    def loss_and_jac(self, x):
        losses, jacs = zip(*[self.loss_and_jac_q(x, q_idx) for q_idx in range(len(self.qs))])

        return np.sum(losses), np.sum(jacs, axis=0)

class ElectricalOnlyCBrillouinOptimizerMid(PointMassOptimizer):
    """ Optimizes the gap-midgap loss function in the brillouin zone,
    but uses the network as an electrical network, with a regular graph
    Laplacian encoding inductively grounded nodes, connected by capacitances.
    """
    def __init__(self, network, gaps, qs):
        super().__init__(network, gaps)
        self.qs = qs

        # compute and cache equilibrium_matrices at the needed
        # values of q
        E = self.network.graph['E']
        self.Es = [self.network.incidence_matrix_ft(q).tocsc() for q in qs]
        self.EHs = [E.conj().transpose() for E in self.Es]

    def num_optim_dof(self):
        return self.network.graph['Q'].shape[1]

    def loss_and_jac_q(self, x, q_idx):
        E = self.Es[q_idx]
        EH = self.EHs[q_idx]

        # L = x[:E.shape[1]] # inductances
        C = x#[E.shape[1]:] # capacitances
        ground = 1.0

        K = E.dot(sp.sparse.diags(C)).dot(EH).toarray()

        # ground capacitively
        # M = (E.dot(sp.sparse.diags(C)).dot(EH) + ground*sp.sparse.eye(E.shape[0])).toarray()

        u, v = sp.linalg.eigh(K, turbo=True)

        # loss function
        loss = np.sum((u[self.gap_inds] - u[self.gap_inds+1])/(u[self.gap_inds] + u[self.gap_inds+1]))

        u_i = u[self.gap_inds]
        u_j = u[self.gap_inds+1]

        grads_i = np.abs(EH.dot(v[:,self.gap_inds]))**2
        grads_j = np.abs(EH.dot(v[:,self.gap_inds+1]))**2

        return loss, np.sum(2*(u_j*grads_i - u_i*grads_j)/(u_i + u_j)**2, axis=1)

    def loss_and_jac(self, x):
        losses, jacs = zip(*[self.loss_and_jac_q(x, q_idx) for q_idx in range(len(self.qs))])

        return np.sum(losses), np.sum(jacs, axis=0)


class ElectricalResonatorGroundedBrillouinOptimizerMid(PointMassOptimizer):
    """ Optimizes the gap-midgap loss function in the brillouin zone,
    but uses the network as an electrical network, with a regular graph
    Laplacian encoding inductive-capacitively grounded nodes, connected by capacitances.
    """
    def __init__(self, network, gaps, qs):
        super().__init__(network, gaps)
        self.qs = qs

        # compute and cache equilibrium_matrices at the needed
        # values of q
        E = self.network.graph['E']
        self.Es = [self.network.incidence_matrix_ft(q).tocsc() for q in qs]
        self.EHs = [E.conj().transpose() for E in self.Es]

    def num_optim_dof(self):
        return 2*self.network.graph['Q'].shape[1]

    def loss_and_jac_q(self, x, q_idx):
        E = self.Es[q_idx]
        EH = self.EHs[q_idx]

        L = x[:E.shape[1]] # inductances
        C = x[E.shape[1]:] # capacitances
        ground = 1.0

        K = (E.dot(sp.sparse.diags(L)).dot(EH) + 0.1*ground*sp.sparse.eye(E.shape[0])).toarray()

        # ground capacitively
        M = (E.dot(sp.sparse.diags(C)).dot(EH) + ground*sp.sparse.eye(E.shape[0])).toarray()

        u, v = sp.linalg.eigh(K, M, turbo=True)

        # loss function
        loss = np.sum((u[self.gap_inds] - u[self.gap_inds+1])/(u[self.gap_inds] + u[self.gap_inds+1]))

        u_i = u[self.gap_inds]
        u_j = u[self.gap_inds+1]

        grads_i = np.concatenate((np.abs(EH.dot(v[:,self.gap_inds]))**2, -u_i*(np.abs(EH.dot(v[:,self.gap_inds]))**2)))
        grads_j = np.concatenate((np.abs(EH.dot(v[:,self.gap_inds+1]))**2, -u_j*(np.abs(EH.dot(v[:,self.gap_inds+1]))**2)))

        return loss, np.sum(2*(u_j*grads_i - u_i*grads_j)/(u_i + u_j)**2, axis=1)


class ElectricalNoCapacitanceBrillouinOptimizerMid(PointMassOptimizer):
    """ Optimizes the gap-midgap loss function in the brillouin zone,
    but uses the network as an electrical network, with a regular graph
    Laplacian encoding nodes connected by resistors and inductors, but
    no capacitances
    """
    def __init__(self, network, gaps, qs):
        super().__init__(network, gaps)
        self.qs = qs

        # compute and cache equilibrium_matrices at the needed
        # values of q
        E = self.network.graph['E']
        self.Es = [self.network.incidence_matrix_ft(q).tocsc() for q in qs]
        self.EHs = [E.conj().transpose() for E in self.Es]

    def num_optim_dof(self):
        return 2*self.network.graph['Q'].shape[1]

    def modes(self, x, q, q_is_idx=True):
        if q_is_idx:
            E = self.Es[q]
            EH = self.EHs[q]
        else:
            E = fourier_transform_E(self.network, self.network.graph['E'], q)
            EH = E.conj().transpose()

        Sigma = x[:E.shape[1]] # conductances
        W = x[E.shape[1]:] # inverse inductances

        Sigma_mat = (E.dot(sp.sparse.diags(Sigma)).dot(EH)).toarray()

        # ground capacitively
        W_mat = (E.dot(sp.sparse.diags(W)).dot(EH)).toarray()

        u, v = sp.linalg.eigh(W_mat, -1j*Sigma_mat, turbo=True)            

        return u, v

    def loss_and_jac_q(self, x, q_idx):
        E = self.Es[q_idx]
        EH = self.EHs[q_idx]

        L = x[:E.shape[1]] # inductances
        C = x[E.shape[1]:] # capacitances
        ground = 1.0

        K = (E.dot(sp.sparse.diags(L)).dot(EH) + 0.1*ground*sp.sparse.eye(E.shape[0])).toarray()

        # ground capacitively
        M = (E.dot(sp.sparse.diags(C)).dot(EH) + ground*sp.sparse.eye(E.shape[0])).toarray()

        u, v = sp.linalg.eigh(K, M, turbo=True)

        # loss function
        loss = np.sum((u[self.gap_inds] - u[self.gap_inds+1])/(u[self.gap_inds] + u[self.gap_inds+1]))

        u_i = u[self.gap_inds]
        u_j = u[self.gap_inds+1]

        grads_i = np.concatenate((np.abs(EH.dot(v[:,self.gap_inds]))**2, -u_i*(np.abs(EH.dot(v[:,self.gap_inds]))**2)))
        grads_j = np.concatenate((np.abs(EH.dot(v[:,self.gap_inds+1]))**2, -u_j*(np.abs(EH.dot(v[:,self.gap_inds+1]))**2)))

        return loss, np.sum(2*(u_j*grads_i - u_i*grads_j)/(u_i + u_j)**2, axis=1)

    def loss_and_jac(self, x):
        losses, jacs = zip(*[self.loss_and_jac_q(x, q_idx) for q_idx in range(len(self.qs))])

        return np.sum(losses), np.sum(jacs, axis=0)


class SpringMassOptimizerMid(PointMassOptimizer):
    """ Same as spectral optimizer, but sets the point masses to 0
    and instead assumes that the spring masses scale with their
    stiffness.
    """
    def _spring_mass_dot(self, v):
        """ Compute the dot product v^T S' v
        """
        E = self.network.graph['E']
        F = self.network.graph['F']
        nodes = len(self.network.graph['nodelist'])
        dim = self.network.graph['dimension']

        vdim = v.reshape((dim, nodes)).T
        dp = np.sum(E.transpose().dot(vdim)**2 + 3*F.transpose().dot(vdim)**2, axis=1)

        return dp/12

    def loss_and_jac(self, x):
        Q = self.network.graph['Q']
        K = self.network.stiffness_matrix(x).toarray()
        S = self.network.spring_mass_matrix(m_s=x).toarray()
        u, v = sp.linalg.eigh(K, b=S, turbo=True)

        loss = np.sum((u[self.gap_inds] - u[self.gap_inds+1])/(u[self.gap_inds] + u[self.gap_inds+1]))

        grad1 = Q.transpose().dot(v[:,self.gap_inds])**2
        grad2 = Q.transpose().dot(v[:,self.gap_inds+1])**2

        grad_springs1 = np.apply_along_axis(self._spring_mass_dot, 0, v[:,self.gap_inds])
        grad_springs2 = np.apply_along_axis(self._spring_mass_dot, 0, v[:,self.gap_inds+1])

        dlambda1 = grad1 - u[self.gap_inds]*grad_springs1
        dlambda2 = grad2 - u[self.gap_inds+1]*grad_springs2

        return loss, np.sum((u[self.gap_inds+1]*dlambda1 - u[self.gap_inds]*dlambda2)/(u[self.gap_inds] - u[self.gap_inds+1])**2, axis=1)

    def optimize_sdp_single(self, seed=None, lower_bound=0.1, upper_bound=1,
                     callback=None, solver=cvx.MOSEK, x0=None,
                     frac_states=0.3, maxiter=200, tol=1e-4,
                     solver_options=None, verbose=False):
        """ Use the sequential semidefinite programming algorithm
        from Freund et al. to optimize a single band gap.

        We use fractional linear programming to deal with the
        scale invariance of the problem.

        The fractional objective is:

        Maximize (lambda_u - lambda_l)/(lambda_u + lambda_l)

        it will be converted to a linear program and then solved.

        """
        # Implement a subspace optimization method
        QT = self.network.graph['Q'].transpose()
        E = self.network.graph['E']
        F = self.network.graph['F']

        nodes = len(self.network.graph['nodelist'])
        dim = self.network.graph['dimension']

        n = QT.shape[0]
        n_eigvals = QT.shape[1]

        # number of states above and below the gap
        n_states = int(frac_states*QT.shape[1])

        assert(len(self.gap_inds) == 1)

        gap_ind = self.gap_inds[0]

        # make sure we do not over or underflow
        above_lim = min(gap_ind + 1 + n_states, n_eigvals)
        below_lim = max(gap_ind - n_states + 1, 0)

        # set up the problem
        def setup_problem(n_states=20):
            # construct fractional semidefinite program
            lambda_u = cvx.Variable()
            lambda_l = cvx.Variable()

            t = cvx.Variable()

            k = cvx.Variable(n)

            k_diag = cvx.diag(k)

            # for each gap consider the space above and below.
            # TODO: CHECK that n_states does not over/underflow
            Q_above = cvx.Parameter(n, above_lim - gap_ind - 1)
            Q_below = cvx.Parameter(n, gap_ind + 1 - below_lim)

            # Mass matrix, with lambda as constant
            E_above_x = cvx.Parameter(n, above_lim - gap_ind - 1)
            E_below_x = cvx.Parameter(n, gap_ind + 1 - below_lim)

            F_above_x = cvx.Parameter(n, above_lim - gap_ind - 1)
            F_below_x = cvx.Parameter(n, gap_ind + 1 - below_lim)

            E_above_y = cvx.Parameter(n, above_lim - gap_ind - 1)
            E_below_y = cvx.Parameter(n, gap_ind + 1 - below_lim)

            F_above_y = cvx.Parameter(n, above_lim - gap_ind - 1)
            F_below_y = cvx.Parameter(n, gap_ind + 1 - below_lim)

            # * is matrix multiplication
            K_above = Q_above.T*k_diag*Q_above - 0.5*(lambda_u*np.eye(above_lim - gap_ind - 1) + (E_above_x.T*k_diag*E_above_x + E_above_y.T*k_diag*E_above_y + 3*F_above_x.T*k_diag*F_above_x + 3*F_above_y.T*k_diag*F_above_y)/12)
            K_below = Q_below.T*k_diag*Q_below - 0.5*(lambda_l*np.eye(gap_ind + 1 - below_lim) + (E_below_x.T*k_diag*E_below_x + E_below_y.T*k_diag*E_below_y + 3*F_below_x.T*k_diag*F_below_x + 3*F_below_y.T*k_diag*F_below_y)/12)

            objective = cvx.Maximize(lambda_u - lambda_l)

            constraints = [k >= lower_bound*t,
                           k <= upper_bound*t,
                           lambda_u >= 0.0,
                           lambda_l + lambda_u == 1.0,
                           t >= 0.0,
                           K_above >> 0,
                           K_below << 0]

            prob = cvx.Problem(objective, constraints)

            return prob, Q_above, Q_below, E_above_x, E_below_x, F_above_x, F_below_x, E_above_y, E_below_y, F_above_y, F_below_y, k, lambda_u, lambda_l, t

        prob, Q_above, Q_below, E_above_x, E_below_x, F_above_x, F_below_x, E_above_y, E_below_y, F_above_y, F_below_y, \
            k, lambda_u, lambda_l, t = setup_problem(n_states=n_states)

        # setup starting point
        if x0 is None:
            x0 = lower_bound + (upper_bound - lower_bound)*np.random.random(n)

        if verbose:
            print("states:", below_lim, above_lim)
            print("i\tobjective\tSDP\t|k_{n+1} - k_n|\tt")

        # run iterative solver
        x_cur = x0.copy()

        x_norm = 0
        j = 0
        xs = [x_cur.copy()]
        for i in range(maxiter):
            # compute current spectrum for rescaled variables
            K = self.network.stiffness_matrix(x_cur).toarray()
            S = self.network.spring_mass_matrix(m_s=x_cur).toarray()

            u, v = sp.linalg.eigh(K, b=S, turbo=True)

            # construct subspaces above and below eigengap(s)
            v_above = v[:,gap_ind+1:above_lim]
            v_below = v[:,below_lim:gap_ind+1]

            v_above_x = v_above[:int(v_above.shape[0]/2),:]
            v_above_y = v_above[int(v_above.shape[0]/2):,:]

            v_below_x = v_below[:int(v_below.shape[0]/2),:]
            v_below_y = v_below[int(v_below.shape[0]/2):,:]

            # construct subspace projected stiffness matrices
            Q_above.value = QT.dot(v_above)
            Q_below.value = QT.dot(v_below)


            # subspace mass matrix with k as dof.
            E_above_x.value = E.transpose().dot(v_above_x)*np.sqrt(u[gap_ind+1])
            E_below_x.value = E.transpose().dot(v_below_x)*np.sqrt(u[gap_ind])

            F_above_x.value = F.transpose().dot(v_above_x)*np.sqrt(u[gap_ind+1])
            F_below_x.value = F.transpose().dot(v_below_x)*np.sqrt(u[gap_ind])

            E_above_y.value = E.transpose().dot(v_above_y)*np.sqrt(u[gap_ind+1])
            E_below_y.value = E.transpose().dot(v_below_y)*np.sqrt(u[gap_ind])

            F_above_y.value = F.transpose().dot(v_above_y)*np.sqrt(u[gap_ind+1])
            F_below_y.value = F.transpose().dot(v_below_y)*np.sqrt(u[gap_ind])

            # use SCS for faster but less accurate solutions
            if solver_options is not None:
                prob.solve(solver=solver, **solver_options)
            else:
                prob.solve(solver=solver)

            # Check convergence conditions
            x_new = np.array(k.value).flatten()/t.value

            # x_new = 0.5*(x_new + x_cur)

            x_norm = np.linalg.norm(x_cur - x_new)
            x_cur = x_new.copy()

            xs.append(x_cur.copy())

            j += 1

            # callback if desired
            if callback is not None:
                callback(x_cur, prob)

            objective = (u[gap_ind+1] - u[gap_ind])/(u[gap_ind+1] + u[gap_ind])

            if verbose:
                print("{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}".format(j + 1, objective, prob.value, x_norm, t.value))

            if x_norm < tol:
                break

        ret = optimize.OptimizeResult()
        ret.fun = objective
        ret.fun_sdp = prob.value
        ret.message = 'SUCCESS: |k_n - k_{n+1}| < tol.' if (j < maxiter - 1) else 'FAIL: iteration limit exceeded.'
        ret.x = x_cur
        ret.xs = xs
        ret.nit = j + 1
        ret.nfev = j + 1
        ret.success = j < maxiter - 1
        ret.jac = x_norm.copy()
        ret.n_states = n_states

        if verbose:
            print(ret.message)

        return ret, x0

if __name__ == '__main__':
    import networks

    threed = networks.TriangularGrid2D(n=4, x_periodic=True, y_periodic=True)

    opt = PointMassNodePosOptimizerMid(threed, [0.5])
    ret, x0 = opt.optimize(seed=42, options={'maxiter':100000, 'maxfun':100000})

    print(ret)

    pts_new, k = opt.unpack_result(ret.x)
    pts = np.array([threed.node[nd]['x'] for nd in threed.graph['nodelist']])

    print(pts)
    print(pts_new)
    print(k)

    loss, Q, K = opt.loss_function(ret.x, return_matrices=True)

    print(sp.linalg.eigvalsh(K.toarray()))
