# region imports
from AlgorithmImports import *
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from numba import jit
import pandas as pd
import numpy as np
import scipy as sp
import time
# endregion


class IPCA(BaseEstimator):
    """
    IPCA initialization parameters:
        n_factors: int, default=1
            total number of factors to estimate, including pre-specified factors (ex. fama-french factors)
        
        intercept: boolean, default=False
            determines whether the model is estimated with or without an intercept
        
        max_iter: int, default=10000
            maximum number of alternating least square updates before estimation is stopped

        iter_tol : float, default=10e-6
            tolerance threshold for stopping the ALS

        alpha : scalar
            Regularizing constant for Gamma estimation.  If this is set to
            zero then the estimation defaults to non-regularized.

        l1_ratio : scalar
            Ratio of l1 and l2 penalties for elastic net Gamma fit.

        n_jobs : scalar
            If greater than one, factor estimation distrubted across many CPU cores

        backend : str
            parallelization backend when n_jobs is greater than one
    """
    def __init__(self, algorithm, n_factors=1, intercept=False, max_iter=10000,
                iter_tol=10e-6, alpha=0., l1_ratio=1., n_jobs=1,
                backend="loky"):
    
        
        # paranoid parameter checking
        if not isinstance(n_factors, int) or n_factors < 1:
            algorithm.debug('n_factors for IPCA must be an int greater / equal 1.')
            return
        if not isinstance(intercept, bool):
            algorithm.debug('IPCA intercept must be boolean')
            return
        if not isinstance(iter_tol, float) or iter_tol >= 1:
            algorithm.debug('IPCA iteration tolerance must be smaller than 1.')
            return 
        if l1_ratio > 1. or l1_ratio < 0.:
            algorithm.debug("IPCA l1_ratio must be between 0 and 1")
            return 
        if alpha < 0.:
            algorithm.debug("IPCA alpha must be greater than or equal to 0")
            return 

        # Save parameters to self
        self.algorithm = algorithm
        self.n_factors = n_factors
        self.intercept = intercept
        self.max_iter = max_iter
        self.iter_tol = iter_tol
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_jobs = n_jobs
        self.backend = backend

    def fit(self, X, y, indices=None, PSF=None, Gamma=None,
        Factors=None, data_type="portfolio"):
        #fits the IPCA model (Gamma and Factors) to the data using the alternating least squares method
        
        """
        Parameters:
            X: matrix of fundamental characteristics. As the matrix is stacked, each row corresponds
                to a entity-time pair in indices

            y: numpy array or pandas series of dependent or target variables (returns)

            indices: numpy array, optional, made of two columns, a entity/security id index and a time index
            
            PSF : numpy array, optional, set of pre-specified factors as matrix of dimension (# of factors, # of periods)

            Gamma : numpy array, optional, set initial values for gamma

            Factors : numpy array, optional, sets initial factor values

            data_type : str, represents data-type used for ALS estimation. One of the following:

                1. panel: ALS uses the untransformed X and y for the estimation, marginally slower but necessary
                        when performing regularized estimation (alpha > 0)

                2. portfolio

                ALS uses a matrix of characteristic weighted portfolios (Q)
                as well as a matrix of weights (W) and count of non-missing
                observations for each time period (val_obs) for the estimation.

                See _build_portfolio for details on how these variables are formed
                from the initial X and y.

                Currently, the bootstrap procedure is only implemented in terms
                of the portfolio data_type.
        """
    
        X, y, indices, metadata = self._prep_input(X, y, indices)
        N, L, T = metadata["N"], metadata["L"], metadata["T"]

        # panel data_type if doing regularized estimation
        if self.alpha > 0.:
            data_type = "panel"

        # Handle pre-specified factors
        if PSF is not None:
            if np.size(PSF, axis=1) != np.size(metadata["dates"]):
                self.algorithm.debug("Number of PSF observations must match number of unique dates")
                return
            self.has_PSF = True
        else:
            self.has_PSF = False

        if self.has_PSF:
            if np.size(PSF, axis=0) == self.n_factors:
                self.algorithm.debug("n_factors = pre-specified factors. Estimate additonal factors by increasing n_factors")
                raise ValueError("No additional factors estimateable")

        #  Treating intercept as if was a prespecified factor, define effective n_factors
        if self.intercept:
            self.n_factors_eff = self.n_factors + 1
            if PSF is not None:
                PSF = np.concatenate((PSF, np.ones((1, T))), axis=0)
            elif PSF is None:
                PSF = np.ones((1, T))
        else:
            self.n_factors_eff = self.n_factors

        # Check that enough features provided
        if np.size(X, axis=1) < self.n_factors_eff:
            self.algorithm.debug("You cannot derive a greater number of independent underlying components than the number of independent observable inputs you have.")
            raise ValueError("Adjust number of characteristics")

    
        #case to account for both PSF or intercept
        self.PSFcase = True if self.has_PSF or self.intercept else False

        # store data
        self.X, self.y, self.indices, self.PSF = X, y, indices, PSF
        Q, W, val_obs = self._build_portfolio(X, y, indices, metadata)
        self.Q, self.W, self.val_obs = Q, W, val_obs
        self.metadata = metadata

        # Run IPCA
        Gamma, Factors = self._fit_ipca(X=X, y=y, indices=indices, Q=Q,
                                    W=W, val_obs=val_obs, PSF=PSF,
                                    Gamma=Gamma, Factors=Factors,
                                    data_type=data_type)

        # Store estimates
        if self.PSFcase:
            date_ln = len(metadata["dates"])
            if self.intercept and self.has_PSF:
                PSF = np.concatenate((PSF, np.ones((1, date_ln))),
                                    axis=0)
            elif self.intercept:
                PSF = np.ones((1, date_ln))
            if Factors is not None:
                Factors = np.concatenate((Factors, PSF), axis=0)
            else:
                Factors = PSF

        self.Gamma, self.Factors = Gamma, Factors

        return self

    def _prep_input(self, X, y=None, indices=None):
        #processes data and extracts metadata

        """
        returns metadata. a dictionary that contains:
            dates: array-like unique dates in panel
            ids: array-like unique ids in panel
            chars: array-like labels for X chars/columns
            T: scalar number of time periods
            N: scalar number of ids
            L: scalar total number of characteristics
        """
        
        # Check fundamental data
        if X is None:
            self.algorithm.debug("No fundamental data inputted")
            raise ValueError("No fundamental characteristic data")
        else:
            # remove any rows of fundamental data with missing observations + corresponding rows in target variables
            non_nan_ind = ~np.any(np.isnan(X), axis=1)
            X = X[non_nan_ind]
            if y is not None:
                y = y[non_nan_ind]

        # flexibility in data based, break out index/multi-indices and fundamental data indices from data
        if isinstance(X, pd.DataFrame) and not isinstance(y, pd.Series):
            indices = X.index
            chars = X.columns
            X = X.values
        elif not isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            indices = y.index
            y = y.values
            chars = np.arange(X.shape[1])
        elif isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            Xind = X.index
            chars = X.columns
            yind = y.index
            X = X.values
            y = y.values
            if not np.array_equal(Xind, yind):
                self.algorithm.debug("Indices provided by both X and y must be the same")
                raise ValueError("Check X and y indices")
            indices = Xind
        else:
            chars = np.arange(X.shape[1])

        if indices is None:
            self.algorithm.debug("Indices must be available")
            raise ValueError("X and y must have indices")

        # transforms dates and potential string equity IDs into numerical 0-based integer code
        if isinstance(indices, pd.MultiIndex):
            indices = indices.to_frame().values
        ids = np.unique(indices[:, 0])
        dates = np.unique(indices[:, 1])
        indices[:,0] = np.unique(indices[:,0], return_inverse=True)[1]
        indices[:,1] = np.unique(indices[:,1], return_inverse=True)[1]

        # find data dimensions
        T = np.size(dates, axis=0)
        N = np.size(ids, axis=0)
        L = np.size(chars, axis=0)

        # Metadata
        metadata = {}
        metadata["dates"] = dates
        metadata["ids"] = ids
        metadata["characteristics"] = chars
        metadata["T"] = T
        metadata["N"] = N
        metadata["L"] = L

        return X, y, indices, metadata

    def _build_portfolio(self, X, y, indices, metadata):
        """
        Breaks data down into:
            W: a characteristic weighting matrix which describes how related the characteristics
            are to each other (like a covariance matrix). Dimensions (L, T), L charactertics, T unique dates

            Q: characteristic weighted portfolios, where each row represents a portfolio weighted by the 
            characteristics of each security. Dimensions (L ,L ,T)

            Also returns how many securities have valid data for a particular time period. Matrix of dimension T. 

            Summarizing this data allows the IPCA calculations to be much faster
        """

        N, L, T = metadata["N"], metadata["L"], metadata["T"]
        self.algorithm.debug(f"The data dimensions are - n_samples: {N}, L: {L}, T: {T}")

        # initialize Q, W, and val_obs
        if y is not None:
            Q = np.full((L, T), np.nan)
        else:
            Q = None
        W = np.full((L, L, T), np.nan)
        val_obs = np.full((T), np.nan)

        self.algorithm.debug("Beginning portfolio aggregation")

        #build in flexbility, allow for compuing W only
        if y is None:
            for t in range(T):
                ixt = (indices[:, 1] == t)
                val_obs[t] = np.sum(ixt)
                W[:, :, t] = X[ixt, :].T.dot(X[ixt, :])/val_obs[t]
                if  t % (T // 10 or 1) == 0: # Log every 10% progress
                    self.algorithm.debug(f"Processing time slice {t+1}/{T} ({(t+1)/T:.1%})")

        else:
            for t in range(T):
                ixt = (indices[:, 1] == t)
                val_obs[t] = np.sum(ixt)
                Q[:, t] = X[ixt, :].T.dot(y[ixt])/val_obs[t]
                W[:, :, t] = X[ixt, :].T.dot(X[ixt, :])/val_obs[t]
                if  t % (T // 10 or 1) == 0: # Log every 10% progress
                    self.algorithm.debug(f"Processing time slice {t+1}/{T} ({(t+1)/T:.1%})")

        self.algorithm.debug("Finished portfolio aggregation")
        return Q, W, val_obs

    def _fit_ipca(self, X=None, y=None, indices=None, PSF=None, Q=None,
                W=None, val_obs=None, Gamma=None, Factors=None, quiet=False,
                data_type="portfolio"):
    
        #Fits the regressor to the data using alternating least squares method. Returns gamma and factors
        if data_type == "panel":
            ALS_inputs = (X, y, indices)
            ALS_fit = self._ALS_fit_panel
        elif data_type == "portfolio":
            ALS_inputs = (Q, W, val_obs)
            ALS_fit = self._ALS_fit_portfolio
        else:
            self.algorithm.debug("data_type not supported")
            raise ValueError("Unsupported ALS method: %s" % data_type)

        # Initialize the Alternating Least Squares Procedure
        #uses SVD on Q to find inital components that explain most variance in returns, initial components used
        #as a starting point for the ALS method
        if Gamma is None or Factors is None:
            Gamma_Old, s, v = np.linalg.svd(Q)
            Gamma_Old = Gamma_Old[:, :self.n_factors_eff]
            s = s[:self.n_factors_eff]
            v = v[:self.n_factors_eff, :]
            Factor_Old = np.diag(s).dot(v)
        if Gamma is not None:
            Gamma_Old = Gamma
        if Factors is not None:
            Factors_Old = Factors

        #Begin ALS Iteration
        tol_current = 1

        iter = 0

        while((iter <= self.max_iter) and (tol_current > self.iter_tol)):

            Gamma_New, Factor_New = ALS_fit(Gamma_Old, *ALS_inputs,
                                            PSF=PSF)

            if self.PSFcase:
                tol_current = np.max(np.abs(Gamma_New - Gamma_Old))
            else:
                tol_current_G = np.max(np.abs(Gamma_New - Gamma_Old))
                tol_current_F = np.max(np.abs(Factor_New - Factor_Old))
                tol_current = max(tol_current_G, tol_current_F)

            # Update factors and loadings
            Factor_Old, Gamma_Old = Factor_New, Gamma_New

            iter += 1

            self.algorithm.debug(f"ALS Interation: {iter}, Adjustment: {tol_current}")
        
        self.algorithm.debug('ALS Convergence Reached')

        return Gamma_New, Factor_New

    def score(self, X, y=None, indices=None, mean_factor=False, data_type="panel"):
        #Find R^2 value for model performance

        if data_type == "panel":

            X, y, indices, metadata = self._prep_input(X, y, indices)
            if y is None:
                y = self.y

            yhat = self.predict(X=X, indices=indices,
                                mean_factor=mean_factor,
                                data_type="panel")

            return r2_score(y, yhat)

        elif data_type == "portfolio":

            X, y, indices, metadata = self._prep_input(X, y, indices)
            if y is None:
                y = self.y
            Q, W, val_obs = self._build_portfolio(X, y, indices, metadata)

            Qhat = self.predict(W=W, mean_factor=mean_factor,
                                data_type="portfolio")
            return r2_score(Q, Qhat)

        else:
            return ValueError("Unsupported data_type: %s" % data_type)

    def project_factors(self, X, y, indices, data_type="panel"):
        """
        Recompute Gamma and Factors on new data without altering the current model.
        
        Returns:
        --------
        Gamma_new : np.ndarray
            Recomputed factor loadings.
        Factors_new : np.ndarray
            Recomputed latent factors.
        """
        X, y, indices, metadata = self._prep_input(X, y, indices)

        if data_type == "panel":
            Gamma_init = self.Gamma.copy()
            F_new, _ = self._ALS_fit_panel(Gamma_init, X, y, indices)
            return Gamma_init, F_new

        elif data_type == "portfolio":
            Q, W, val_obs = self._build_portfolio(X, y, indices, metadata)
            Gamma_init = self.Gamma.copy()
            F_new, _ = self._ALS_fit_portfolio(Gamma_init, Q, W, val_obs)
            return Gamma_init, F_new

        else:
            raise ValueError(f"Unsupported data_type: {data_type}")


    def _ALS_fit_panel(self, Gamma_Old, X, y, indices, PSF=None):
        #performs one interation of the alternating least squares procedure with panel data. 

        T, dates = self.metadata["T"], self.metadata["dates"]

        #Ktilde: total number of factors in the model - latent factors, PSF, and intercept
        #K is number of latent factors

        if PSF is None:
            L, K = np.shape(Gamma_Old)
            Ktilde = K
        else:
            L, Ktilde = np.shape(Gamma_Old)
            K_PSF, _ = np.shape(PSF)
            K = Ktilde - K_PSF

        # prep Time-index for iteration
        Tind = [np.where(indices[:,1] == t)[0] for t in range(T)]

        # ALS Step 1
        if K > 0:

            #no prespecified factors. calculate factors for each time t, store in new factors matrix 
            if PSF is None:
                if self.n_jobs > 1:
                    F_New = Parallel(n_jobs=self.n_jobs,
                                    backend=self.backend)(
                                delayed(self._Ft_fit_panel)(
                                    Gamma_Old, X[tind,:], y[tind])
                                for t, tind in enumerate(Tind))
                    F_New = np.stack(F_New, axis=1)

                else:
                    F_New = np.full((K, T), np.nan)
                    for t, tind in enumerate(Tind):
                        F_New[:,t] = _Ft_fit_panel(Gamma_Old, X[tind,:],
                                                y[tind])

            #if given prespecified factors
            else:
                if self.n_jobs > 1:
                    F_New = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                                delayed(_Ft_fit_PSF_panel)(
                                    Gamma_Old, X[tind,:], y[tind],
                                    PSF[:,t], K, Ktilde)
                                for t, tind in enumerate(Tind))
                    F_New = np.stack(F_New, axis=1)

                else:
                    F_New = np.full((K, T), np.nan)
                    for t, tind in enumerate(Tind):
                        F_New[:,t] = _Ft_fit_PSF_panel(Gamma_Old, X[tind,:],
                                                    y[tind], PSF[:,t],
                                                    K, Ktilde)               
        else:
            F_New = None
        
        # ALS Step 2
        Gamma_New = _Gamma_fit_panel(F_New, X, y, indices, PSF, L, Ktilde,
                                    self.alpha, self.l1_ratio)
        
        # condition checks
        if K > 0:
            # Enforce Orthogonality in Gamma_alpha (loadings of intercept) and Gamma_beta (loadings of latent factors): 
            # ensure that they are uncorrelated

            #Compute and stabilize gram matrix of first K loadings
            M = Gamma_New[:, :K].T @ Gamma_New[:, :K]
            M = (M + M.T) * 0.5
            eps = 1e-6
            M += np.eye(K) * eps
            
            #Cholesky orthogonalization
            R1 = _numba_chol(M).T
            Gamma_New[:, :K] = Gamma_New[:, :K] @ np.linalg.inv(R1)
            F_New[:K, :]   = np.linalg.inv(R1).T @ F_New[:K, :]

            #SVD re-orthonormalization to preserve reconstruction
            R2, _, _ = _numba_svd(R1 @ F_New[:K, :] @ F_New[:K, :].T @ R1.T)
            Gamma_New[:, :K] = (_numba_lstsq(Gamma_New[:, :K].T, R1.T)[0]
                                .dot(R2))
            F_New[:K, :] = _numba_solve(R2, R1 @ F_New[:K, :])

            #Enforce sign convention
            sg = np.sign(np.mean(F_New[:K, :], axis=1, keepdims=True))
            sg[sg == 0] = 1
            Gamma_New[:, :K] *= sg.T
            F_New[:K, :]   *= sg

            #Adjust for pre-specified factors (if provided)
            if PSF is not None:
                # Project out the latent-factor space
                P_perp = np.eye(Gamma_New.shape[0]) - Gamma_New[:, :K] @ Gamma_New[:, :K].T
                Gamma_New[:, K:] = P_perp @ Gamma_New[:, K:]
                # Inject PSF contributions into factors
                F_New += Gamma_New[:, :K].T @ Gamma_New[:, K:] @ PSF

                # Re-enforce sign convention after PSF adjustment
                sg = np.sign(np.mean(F_New, axis=1, keepdims=True))
                sg[sg == 0] = 1
                Gamma_New[:, :K] *= sg.T
                F_New[:K, :]   *= sg

            # #adjust for prespeciified factors
            # if PSF is not None:
            #     Gamma_New[:, K:] = (np.identity(Gamma_New.shape[0]) - Gamma_New[:, :K].dot(Gamma_New[:, :K].T)).dot(Gamma_New[:, K:])
            #     F_New += Gamma_New[:, :K].T.dot(Gamma_New[:, K:]).dot(PSF)

            #     sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
            #     sg[sg == 0] = 1
            #     Gamma_New[:, :K] = np.multiply(Gamma_New[:, :K], sg.T)
            #     F_New = np.multiply(F_New, sg)

        return Gamma_New, F_New

    def _ALS_fit_portfolio(self, Gamma_Old, Q, W, val_obs, PSF=None):
        #performs one interation of the alternating least squares procedure using aggregated Q and W data, not raw data
        T = self.metad["T"]

        if PSF is None:
            L, K = np.shape(Gamma_Old)
            Ktilde = K
        else:
            L, Ktilde = np.shape(Gamma_Old)
            K_PSF, _ = np.shape(PSF)
            K = Ktilde - K_PSF

        # ALS Step 1
        if K > 0:
            # case with no observed factors
            if PSF is None:
                if self.n_jobs > 1:
                    F_New = Parallel(n_jobs=self.n_jobs,
                                    backend=self.backend)(
                                delayed(_Ft_fit_portfolio)(
                                    Gamma_Old, W[:,:,t], Q[:,t])
                                for t in range(T))
                    F_New = np.stack(F_New, axis=1)

                else:
                    F_New = np.full((K, T), np.nan)
                    for t in range(T):
                        F_New[:,t] = _Ft_fit_portfolio(Gamma_Old, W[:,:,t],
                                                    Q[:,t])

            #given prespecified factors
            else:
                if self.n_jobs > 1:
                    F_New = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                                delayed(_Ft_fit_PSF_portfolio)(
                                    Gamma_Old, W[:,:,t], Q[:,t], PSF[:,t],
                                    K, Ktilde)
                                for t in range(T))
                    F_New = np.stack(F_New, axis=1)

                else:
                    F_New = np.full((K, T), np.nan)
                    for t in range(T):
                        F_New[:,t] = _Ft_fit_PSF_portfolio(Gamma_Old,
                                                        W[:,:,t], Q[:,t],
                                                        PSF[:,t], K,
                                                        Ktilde)
        else: 
            F_New = None

        Gamma_New = _Gamma_fit_portfolio(F_New, Q, W, val_obs, PSF, L, K,
                                        Ktilde, T)
        # condition checks

        if K > 0:
            # Enforce Orthogonality in Gamma_alpha (loadings of intercept) and Gamma_beta (loadings of latent factors): 
            # ensure that they are uncorrelated
            R1 = _numba_chol(Gamma_New[:, :K].T.dot(Gamma_New[:, :K])).T
            R2, _, _ = _numba_svd(R1.dot(F_New).dot(F_New.T).dot(R1.T))
            Gamma_New[:, :K] = _numba_lstsq(Gamma_New[:, :K].T,
                                            R1.T)[0].dot(R2)
            F_New = _numba_solve(R2, R1.dot(F_New))

            # Enforce sign convention for Gamma_Beta and F_New
            sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
            sg[sg == 0] = 1
            Gamma_New[:, :K] = np.multiply(Gamma_New[:, :K], sg.T)
            F_New = np.multiply(F_New, sg)

            if PSF is not None:
                Gamma_New[:, K:] = (np.identity(Gamma_New.shape[0]) - Gamma_New[:, :K].dot(Gamma_New[:, :K].T)).dot(Gamma_New[:, K:])
                F_New += Gamma_New[:, :K].T.dot(Gamma_New[:, K:]).dot(PSF)

                sg = np.sign(np.mean(F_New, axis=1)).reshape((-1, 1))
                sg[sg == 0] = 1
                Gamma_New[:, :K] = np.multiply(Gamma_New[:, :K], sg.T)
                F_New = np.multiply(F_New, sg)

        return Gamma_New, F_New

        #incomplete
    
    def predict(self, X=None, indices=None, W=None, mean_factor=False,
                data_type="panel", label_ind=False):
        #predicts future returns
        """
        Parameters:
            X: matrix of characteristics, assumed to be multiindex mapping to each entity-time pair

            mean_factor: boolean, if true, estimated factors are averaged before prediction.

            label_ind : boolean, if true, applies the indices to fitted values and returned pandas Series

        Returns a series of values for the panel data type or a matrix of portfolio Qs for the portoflio data type.
        If label_in is True, returns pandas variants of the predicted values. Otherwise, the underlying numpy arrays
        """

        if data_type == "panel":
            if X is None:
                X, indices, metadata = self.X, self.indices, self.metadata
            else:
                X, y, indices, metadata = self._prep_input(X, None, indices)

            N, L, T = metadata["N"], metadata["L"], metadata["T"]

            pred = self.predict_panel(X, indices, T, mean_factor)

            if label_ind:
                pred = pd.DataFrame(pred, columns=["yhat"])
                ind = pd.DataFrame(indices, columns=["ids", "dates"])
                pred = pd.concat([ind, pred]).set_index(["ids", "dates"])
        
        elif data_type == "portfolio":

            if W is not None:
                L = W.shape[0]
                T = W.shape[2]
            elif X is not None:
                X, y, indices, metadata = self._prep_input(X, None, indices)
                Q, W, val_obs = self._build_portfolio(X, None, indices, metadata)
                L = W.shape[0]
                T = W.shape[2]
            elif hasattr(self, "W"):
                W = self.W
                L = W.shape[0]
                T = W.shape[2]
            else:
                X, indices, metadata = self.X, self.indices, self.metadata
                N, L, T = metadata["N"], metadata["L"], metadata["T"]
                Q, W, val_obs = self._build_portfolio(X, None, indices, metadata)
            
            pred = self.predict_portfolio(W, L, T, mean_factor)

            if label_ind:
                pred = pd.DataFrame(pred, index=metadata["chars"],
                                    columns=metadata["dates"])

        else:
            self.algorithm.debug("Data Type not panel or portfolio")
            raise ValueError("Unsupported data_type: %s" % data_type)

        return pred

    def predict_panel(self, X, indices, T, mean_factor=False):
        #Predicts values using the previously fitted regressor and panel data
        mean_Factors = np.mean(self.Factors, axis=1).reshape((-1, 1))

        if mean_factor:
            ypred = np.squeeze(X.dot(self.Gamma).dot(mean_Factors))
        elif T != self.Factors.shape[1]:
            self.algorithm.debug("Input data time periods must match number of time periods of factors estimated")
            raise ValueError("If mean_factor isn't used date shape must align\
                            with Factors shape")
        else:
            ypred = np.full((X.shape[0]), np.nan)
            for t in range(T):
                ix = (indices[:, 1] == t)
                ypred[ix] = np.squeeze(X[ix, :].dot(self.Gamma)\
                    .dot(self.Factors[:, t]))
        return ypred

    def predict_portfolio(self, W, L, T, mean_factor=False):

        # Compute goodness of fit measures, portfolio level
        Num_tot, Denom_tot, Num_pred, Denom_pred = 0, 0, 0, 0

        if not mean_factor and T != self.Factors.shape[1]:
            raise ValueError("If mean_factor isn't used date shape must align\
                            with Factors shape")

        if mean_factor:
            mean_Factors = np.mean(self.Factors, axis=1).reshape((-1, 1))

        Qpred = np.full((L, T), np.nan)
        for t in range(T):
            if mean_factor:
                qpred = W[:, :, t].dot(self.Gamma).dot(mean_Factors)
                qpred = np.squeeze(qpred)
                Qpred[:,t] = qpred
            else:
                qpred = W[:, :, t].dot(self.Gamma)\
                    .dot(self.Factors[:, t])
                Qpred[:,t] = qpred

        return Qpred

def _Ft_fit_panel(Gamma_Old, X_t, y_t):
    #fits the factors using panel data

    estimated_factor_loadings = X_t.dot(Gamma_Old)
    Ft = _numba_lstsq(estimated_factor_loadings, y_t)[0]

    return Ft

def _Ft_fit_PSF_panel(Gamma_Old, X_t, y_t, PSF_t, K, Ktilde):
    #fits the factors using panel data given PSF
    #subtracts PSF predicted values from target value and regresses that against the estimated latent factor loadings
    #residuals are y values that cannot be explained by prespecified factors

    estimated_factor_loadings = X_t.dot(Gamma_Old)
    y_t_resid = y_t - estimated_factor_loadings[:,K:Ktilde].dot(PSF_t)
    Ft = _numba_lstsq(estimated_factor_loadings[:,:K], y_t_resid)[0]

    return Ft

def _Ft_fit_portfolio(Gamma_Old, W_t, Q_t):
    #fits factor loadings to aggregated data given first order condition:
    #m1 x Ft = m2

    m1 = Gamma_Old.T.dot(W_t).dot(Gamma_Old)
    m2 = Gamma_Old.T.dot(Q_t)

    return np.squeeze(_numba_solve(m1, m2.reshape((-1, 1))))

def _Ft_fit_PSF_portfolio(Gamma_Old, W_t, Q_t, PSF_t, K, Ktilde):
    #fits factor loadings to aggregated data with prespecified factors given first order condition:
    #Gamma_latent.T.dot(W_t).dot(Gamma_latent) x F_latent = Gamma_latent.T.dot(Q_t) - Gamma_latent.T.dot(W_t).dot(Gamma_PSF).dot(F_PSF)

    m1 = Gamma_Old[:,:K].T.dot(W_t).dot(Gamma_Old[:,:K])
    m2 = Gamma_Old[:,:K].T.dot(Q_t)
    m2 -= Gamma_Old[:,:K].T.dot(W_t).dot(Gamma_Old[:,K:Ktilde]).dot(PSF_t)

    return np.squeeze(_numba_solve(m1, m2.reshape((-1, 1))))

def _Gamma_fit_panel(F_New, X, y, indices, PSF, L, Ktilde, alpha, l1_ratio,
                     **kwargs):
    #fits Gamma using panel data

    # join observed factors with latent factors and map to panel
    if PSF is None:
        F = F_New
    else:
        if F_New is None:
            F = PSF
        else:
            F = np.vstack((F_New, PSF))
    F = F[:,indices[:,1]]

    # interact factors and characteristics
    ZkF = np.hstack([F[k,:,None] * X for k in range(Ktilde)])

    # elastic net fit
    if alpha:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)
        model.fit(ZkF, y)
        gamma = model.coef_

    # OLS fit
    else:
        gamma = _numba_lstsq(ZkF, y)[0]

    gamma = gamma.reshape((Ktilde, L)).T

    return gamma

def _Gamma_fit_portfolio(F_New, Q, W, val_obs, PSF, L, K, Ktilde, T):
    #fits portoflio data to Gamma matrix

    Numer = _numba_full((L*Ktilde, 1), 0.0)
    Denom = _numba_full((L*Ktilde, L*Ktilde), 0.0)

    #no prespecified factors
    if PSF is None:
        for t in range(T):

            Numer += _numba_kron(Q[:, t].reshape((-1, 1)),
                                 F_New[:, t].reshape((-1, 1)))\
                                 * val_obs[t]
            Denom += _numba_kron(W[:, :, t],
                                 F_New[:, t].reshape((-1, 1))
                                 .dot(F_New[:, t].reshape((1, -1)))) \
                                 * val_obs[t]

    # observed+latent factors
    elif K > 0:
        for t in range(T):
            Numer += _numba_kron(Q[:, t].reshape((-1, 1)),
                                 np.vstack(
                                 (F_New[:, t].reshape((-1, 1)),
                                 PSF[:, t].reshape((-1, 1)))))\
                                 * val_obs[t]
            Denom_temp = np.vstack((F_New[:, t].reshape((-1, 1)),
                                   PSF[:, t].reshape((-1, 1))))
            Denom += _numba_kron(W[:, :, t], Denom_temp.dot(Denom_temp.T)
                                 * val_obs[t])

    # only observed factors
    else:
        for t in range(T):
            Numer += _numba_kron(Q[:, t].reshape((-1, 1)),
                                 PSF[:, t].reshape((-1, 1)))\
                                 * val_obs[t]
            Denom += _numba_kron(W[:, :, t],
                                 PSF[:, t].reshape((-1, 1))
                                 .dot(PSF[:, t].reshape((-1, 1)).T))\
                                 * val_obs[t]

    Gamma_New = _numba_solve(Denom, Numer).reshape((L, Ktilde))

    return Gamma_New

@jit(nopython=True)
def _numba_lstsq(m1, m2):
    #efficiently calculates factors using linear regression
    return np.linalg.lstsq(np.ascontiguousarray(m1), np.ascontiguousarray(m2))

@jit(nopython=True)
def _numba_solve(m1, m2):
    return np.linalg.solve(np.ascontiguousarray(m1), np.ascontiguousarray(m2))

@jit(nopython=True)
def _numba_full(m1, m2):
    return np.full(m1, m2)

@jit(nopython=True)
def _numba_chol(m1):
    return np.linalg.cholesky(np.ascontiguousarray(m1))

@jit(nopython=True)
def _numba_svd(m1):
    return np.linalg.svd(np.ascontiguousarray(m1))

@jit(nopython=True)
def _numba_kron(m1, m2):
    return np.kron(np.ascontiguousarray(m1), np.ascontiguousarray(m2))
