import time

import numpy as np
from numpy import transpose as tr
from numpy.linalg import inv, pinv
from scipy import linalg


class PPCA:

    def __init__(self, dl=1, tol=10**-6, max_iter=1000, method='EM', verbose=True, debug=False):

        self.dl = dl
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.verbose = verbose
        self.debug = debug
        self.eps = np.finfo(float).eps

    def fit(self, Y):

        Y = Y.T

        self.Y = Y
        self.d, self.n = self.Y.shape

        # present indicator, 0 if missing, 1 if present
        S = np.ones((self.d, self.n))
        S[np.isnan(self.Y)] = 0
        self.S = S

        if np.sum(self.S) < self.d * self.n:
            if self.dl == 1:
                self.__fit_missing1d()
            else:
                self.__fit_missing()
        else:
            if self.method == 'EM':
                self.__fit_em()
            else:
                self.__fit()

    def __fit_missing1d(self):
        """ PPCA for missing when dl=1 """

        if self.verbose:
            print("1D EM PPCA for missing")

        Y0 = self.Y.copy()
        Y0[self.S == 0] = 0
        nObs = np.sum(self.S)

        # mu = np.nanmean(self.Y, axis=1)
        mu = np.zeros(self.d)
        A = np.random.random((self.d, self.dl))
        # A = Y0[:, np.random.randint(0, self.n)][:, np.newaxis]
        # A = np.ones((self.d, 1))
        # sig2 = np.var(self.Y[self.S != 0])
        sig2 = np.random.random()

        Aold = np.random.rand(self.d, 1)
        start = time.time()

        for i in range(self.max_iter):

            if self.verbose and i % 20 == 0:
                took = time.time() - start
                print("{0}/{1} updates, {2:.2f} s".format(i, self.max_iter, took))
                start = time.time()

            Adiff = np.sum(np.abs(A - Aold)) / self.d
            if Adiff < self.tol and i > 20:
                if self.verbose:
                    print("PPCA weights converged after {} iterations...".format(i))
                break
            Aold = A.copy()

            y = (Y0 - mu[:, np.newaxis]) * self.S
            yy = y*y

            M = np.dot((A*A).T, self.S) + sig2 + self.eps

            Ez = np.dot(A.T, y) / M
            Ezz = sig2 / M + Ez ** 2

            errMx = (Y0 - np.dot(A, Ez)) * self.S
            mu = np.sum(errMx, axis=1) / np.sum(self.S, axis=1)

            A = y.dot(Ez.T) / self.S.dot(Ezz.T)

            sig2 = np.sum(yy - y * A.dot(Ez)) / nObs

            self.A = A
            self.sig2 = sig2
            self.mu = mu[:, np.newaxis]
            self.Z = Ez

    def __fit_missing(self):

        if self.verbose:
            print("Missing data, doing EM PPCA")

        Y0 = self.Y.copy()
        Y0[self.S == 0] = 0
        nObs = np.sum(self.S)

        # initial values
        A = np.random.random((self.d, self.dl))
        # A = Y0[:, np.random.randint(0, self.n)][:, np.newaxis]
        mu = np.random.random(self.d)
        sig2 = np.random.random()

        Aold = np.ones((self.d, self.dl))
        ollh = []
        start = time.time()

        for count in np.arange(self.max_iter):

            Adiff = np.sum(np.abs(A - Aold)) / (self.d * self.dl)
            if Adiff < self.tol and count > 20:
                if self.verbose:
                    print("PPCA converged after {} iterations...".format(count))
                break
            Aold = A.copy()

            if self.verbose and count % 20 == 0:
                took = time.time() - start
                print("{0}/{1} updates, {2:.2f} s, {3:.2e} Adiff".format(count, self.max_iter, took, Adiff))
                start = time.time()

            invM = np.zeros((self.dl, self.dl, self.n))
            Z = np.zeros((self.dl, self.n))

            ll = 0.0

            for i in np.arange(self.n):
                s = self.S[:, i]
                nobs = int(np.sum(s))
                m = mu[s == 1]
                y = self.Y[s == 1, i] - m
                a = A[s == 1, :]

                invMi = inv(sig2*np.eye(self.dl) + np.dot(a.T, a))
                Z[:, i] = invMi.dot(a.T).dot(y)
                invM[:, :, i] = invMi

                if self.debug:
                    # only calculate ollh when debugging, since it doubles the runtime
                    invC = (np.eye(nobs) - a.dot(invMi).dot(a.T)) / sig2
                    logDetInvC = np.log(np.linalg.det(np.eye(nobs) - a.dot(invMi).dot(a.T))) - nobs * np.log(sig2)
                    ll += - 0.5 * nobs * np.log(2 * np.pi) + 0.5 * logDetInvC - 0.5 * np.sum(y * invC.dot(y))

            ollh.append(ll)

            errMx = (Y0 - np.dot(A,Z))*self.S
            mu = np.sum(errMx, axis=1)/np.sum(self.S, axis=1)

            # update A and sig2
            ssum = 0.0
            for j in np.arange(self.d):
                s = self.S[j, :]
                m = mu[j]
                y = self.Y[j, s == 1] - m

                bottom = np.dot(Z[:, s == 1], Z[:, s == 1].T) + sig2*np.sum(invM[:, :, s == 1], axis=2)
                top = np.dot(Z[:, s == 1], y.T)

                A[j, :] = np.dot(inv(bottom), top)

                ssum += np.dot(y.T, y) - y.T.dot(np.dot(A[j, :], Z[:, s == 1]))

            sig2 = ssum / nObs

            # speed up convergence a lot
            (A, _, _) = np.linalg.svd(A, full_matrices=False)

            self.A = A
            self.sig2 = sig2
            self.mu = mu[:, np.newaxis]
            self.Z = Z
            self.ollh = ollh

    def __fit(self):
        """ML PPCA in the no-missing case"""
        if self.verbose:
            print("No missing data, doing maximum likelihood PCA")
        self.mu = np.mean(self.Y, 1)[:, np.newaxis]
        [u, s, v] = np.linalg.svd(np.cov(self.Y - self.mu))
        self.sig2 = 1 / (self.d - self.dl) * sum(s[self.dl:])
        # self.A = u[:, :self.dl].dot(np.diag(s[:self.dl]) - self.sig2*np.eye(self.dl))
        self.A = u[:, :self.dl]

    def __fit_em(self):
        """EM PPCA in the no-missing case """
        if self.verbose:
            print("No missing data, using EM PPCA")

        self.mu = np.mean(self.Y, axis=1)[:, np.newaxis]
        y = self.Y - self.mu
        yy = y*y
        A = np.random.random((self.d, self.dl))
        Aold = np.ones((self.d, self.dl))
        sig2 = np.random.random()

        ollh = np.zeros(self.max_iter)
        start = time.time()

        for i in range(self.max_iter):

            if self.verbose and i%20 == 0:
                took = time.time() - start
                print("{0}/{1} updates, {2:.2f} s".format(i, self.max_iter, took))
                start = time.time()

            invM = inv(np.dot(A.T, A) + sig2*np.eye(self.dl))

            invCi = (np.eye(self.d) - A.dot(invM).dot(A.T)) / sig2
            logDetInvC = np.log(np.linalg.det(np.eye(self.d) - A.dot(invM).dot(A.T))) -self.d * np.log(sig2)
            ollh[i] = -self.n * 0.5 * (self.d * np.log(2 * np.pi) - logDetInvC + np.sum(y * invCi.dot(y) / self.n))

            if np.sum(np.abs(A - Aold)) / (self.d * self.dl) < self.tol:
                if self.verbose:
                    print("PPCA converged after {} iterations...".format(i))
                break
            Aold = A.copy()

            Ez = invM.dot(A.T).dot(y)
            Ezz = self.n*sig2*invM + Ez.dot(Ez.T)

            A = y.dot(Ez.T).dot(inv(Ezz))
            sig2 = np.sum(yy - y*A.dot(Ez))/ (self.n * self.d)

        # (A, _, _) = np.linalg.svd(A, full_matrices=False)

        self.A = A
        self.sig2 = sig2
        self.ollh = ollh
        self.Z = Ez

    def __ollh(self, A, mu, sigma2):
        """Observed data log likelihood"""

        ll = 0.0

        for i in np.arange(self.n):

            s = self.S[:, i]
            nobs = int(sum(s))
            m = mu[s==1]
            y = self.Y[s == 1, i] - m
            a = A[s == 1, :]

            if nobs > 0:

                C = np.dot(a,tr(a)) + sigma2*np.eye(nobs)
                logDetC = np.log(linalg.det(C))
                #logDetC = sum(np.log(linalg.eigvals(C)))
                ll = ll - 0.5*nobs*np.log(2*np.pi) - 0.5*logDetC - 0.5*tr(y).dot(pinv(C)).dot(y)

        return ll

    def get_params(self):
        return self.A, self.sig2, self.mu

    def transform(self, Y):

        Y = Y.T

        d, n = Y.shape

        S = np.ones_like(Y)
        S[np.isnan(Y)] = 0
        Yhat = np.nan * np.ones_like(Y)

        invM = np.zeros((self.dl, self.dl, n))
        Z = np.zeros((self.dl, n))

        for i in np.arange(n):
            s = S[:, i]
            nobs = int(np.sum(s))

            if nobs == 0:
                Yhat[:, i] = self.mu.squeeze()
                continue

            m = self.mu[s == 1].squeeze()
            y = Y[s == 1, i] - m
            a = self.A[s == 1, :]

            invMi = inv(self.sig2 * np.eye(self.dl) + np.dot(a.T, a))
            Z[:, i] = invMi.dot(a.T).dot(y)
            invM[:, :, i] = invMi

            Yhat[:, i] = np.dot(self.A, Z[:, i])

        Ymixed = np.copy(Y)
        Ymixed[np.isnan(Y)] = Yhat[np.isnan(Y)]

        return Ymixed.T

    @staticmethod
    def subspace(A, B):
        """ Use the 2-norm to compare the angle between two subspaces A and B"""
        a = linalg.orth(A)
        b = linalg.orth(B)
        b = b - a.dot( np.dot(a.T, b) )
        return np.arcsin(np.linalg.norm(b, 2))

    @staticmethod
    def eval_llh(Ytest, A, sig2, mu):

        D, N = Ytest.shape
        _, dl = A.shape
        y = Ytest - mu

        invM = inv(A.T.dot(A) + sig2 * np.eye(dl))
        invC = (np.eye(D) - A.dot(invM).dot(A.T)) / sig2

        logDetInvC = np.log(np.linalg.det(np.eye(D) - A.dot(invM).dot(A.T))) - D * np.log(sig2)

        llh = - 0.5 * (D * np.log(2 * np.pi) - logDetInvC + np.sum(y * invC.dot(y), axis=0))
        llh_mean = np.sum(llh) / N

        return llh_mean, np.asarray(llh)
