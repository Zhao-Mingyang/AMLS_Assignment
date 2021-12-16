"""
High-level Extreme Learning Machine modules
"""

import numpy as np
import warnings
import scipy as sp

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone,  TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels, type_of_target

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_random_state
from enum import Enum
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from sklearn.utils.extmath import safe_sparse_dot

class BatchCholeskySolver(BaseEstimator, RegressorMixin):
    
    def __init__(self, alpha=1e-7):
        self.alpha = alpha


    def _init_XY(self, X, y):
        """Initialize covariance matrices, including a separate bias term.
        """
        d_in = X.shape[1]
        self._XtX = np.eye(d_in + 1) * self.alpha
        self._XtX[0, 0] = 0
        if len(y.shape) == 1:
            self._XtY = np.zeros((d_in + 1,)) 
        else:
            self._XtY = np.zeros((d_in + 1, y.shape[1]))

    @property
    def XtY_(self):
        return self._XtY

    @property
    def XtX_(self):
        return self._XtX

    @XtY_.setter
    def XtY_(self, value):
        self._XtY = value

    @XtX_.setter
    def XtX_(self, value):
        self._XtX = value

    def _solve(self):
        """Second stage of solution (X'X)B = X'Y using Cholesky decomposition.

        Sets `is_fitted_` to True.
        """
        B = sp.linalg.solve(self._XtX, self._XtY, assume_a='pos', overwrite_a=False, overwrite_b=False)
        self.coef_ = B[1:]
        self.intercept_ = B[0]
        self.is_fitted_ = True

    def _reset(self):
        """Erase solution and data matrices.
        """
        [delattr(self, attr) for attr in ('_XtX', '_XtY', 'coef_', 'intercept_', 'is_fitted_') if hasattr(self, attr)]

    def fit(self, X, y):
        """Solves an L2-regularized linear system like Ridge regression, overwrites any previous solutions.
        """
        self._reset()  # remove old solution
        self.partial_fit(X, y, compute_output_weights=True)
        return self

    
    def partial_fit(self, X, y, compute_output_weights=True):
        """Update model with a new batch of data.
        
        Output weight computation can be temporary turned off for faster processing. This will mark model as
        not fit. Enable `compute_output_weights` in the final call to `partial_fit`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            Training input samples

        y : array-like, shape=[n_samples, n_targets]
            Training targets

        compute_output_weights : boolean, optional, default True
            Whether to compute new output weights (coef_, intercept_). Disable this in intermediate `partial_fit`
            steps to run computations faster, then enable in the last call to compute the new solution.

            .. Note::
                Solution can be updated without extra data by setting `X=None` and `y=None`.
        """
        if self.alpha < 0:
            raise ValueError("Regularization parameter alpha must be non-negative.")

        # solution only
        if X is None and y is None and compute_output_weights:
            self._solve()
            return self

        # validate parameters
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True, y_numeric=True, ensure_2d=True)
        if len(y.shape) > 1 and y.shape[1] == 1:
            msg = "A column-vector y was passed when a 1d array was expected.\
                   Please change the shape of y to (n_samples, ), for example using ravel()."
            warnings.warn(msg, DataConversionWarning)
        
        # init temporary data storage
        if not hasattr(self, '_XtX'):
            self._init_XY(X, y)
        else:
            if X.shape[1] + 1 != self._XtX.shape[0]:
                n_new, n_old = X.shape[1], self._XtX.shape[0] - 1
                raise ValueError("Number of features %d does not match previous data %d." % (n_new, n_old))
                
        # compute temporary data
        X_sum = safe_sparse_dot(X.T, np.ones((X.shape[0],)))
        y_sum = safe_sparse_dot(y.T, np.ones((y.shape[0],)))
        self._XtX[0, 0] += X.shape[0]
        self._XtX[1:, 0] += X_sum
        self._XtX[0, 1:] += X_sum
        self._XtX[1:, 1:] += X.T @ X

        self._XtY[0] += y_sum
        self._XtY[1:] += X.T @ y
        
        # solve
        if not compute_output_weights:
            # mark as not fitted
            [delattr(self, attr) for attr in ('coef_', 'intercept_', 'is_fitted_') if hasattr(self, attr)]
        else:
            self._solve()
        return self

    
    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True)
        return safe_sparse_dot(X, self.coef_, dense_output=True) + self.intercept_

def auto_neuron_count(n, d):
    # computes default number of neurons for `n` data samples with `d` features
    return min(int(250 * np.log(1 + d/10) - 15), n//3 + 1)


def dummy(x):
    return x


class HiddenLayerType(Enum):
    RANDOM = 1    # Gaussian random projection
    SPARSE = 2    # Sparse Random Projection
    PAIRWISE = 3  # Pairwise kernel with a number of centroids
    

ufuncs = {"tanh": np.tanh,
          "sigm": sp.special.expit,
          "relu": lambda x: np.maximum(x, 0),
          "lin": dummy,
          None: dummy}


class HiddenLayer(BaseEstimator, TransformerMixin):

    def __init__(self, n_neurons=None, density=None, ufunc="tanh", pairwise_metric=None, random_state=None):
        self.n_neurons = n_neurons
        self.density = density
        self.ufunc = ufunc
        self.pairwise_metric = pairwise_metric
        self.random_state = random_state

        
    def _fit_random_projection(self, X):
        self.hidden_layer_ = HiddenLayerType.RANDOM
        self.projection_ = GaussianRandomProjection(n_components=self.n_neurons_, random_state=self.random_state_)
        self.projection_.fit(X)
            
    def _fit_sparse_projection(self, X):
        self.hidden_layer_ = HiddenLayerType.SPARSE
        self.projection_ = SparseRandomProjection(n_components=self.n_neurons_, density=self.density,
                                                  dense_output=True, random_state=self.random_state_)
        self.projection_.fit(X)

    def _fit_pairwise_projection(self, X):
        self.hidden_layer_ = HiddenLayerType.PAIRWISE
        self.projection_ = PairwiseRandomProjection(n_components=self.n_neurons_,
                                                    pairwise_metric=self.pairwise_metric,
                                                    random_state=self.random_state_)
        self.projection_.fit(X)
    
    def fit(self, X, y=None):
        # basic checks
        X = check_array(X, accept_sparse=True)

        # handle random state
        self.random_state_ = check_random_state(self.random_state)
        
        # get number of neurons
        n, d = X.shape
        self.n_neurons_ = self.n_neurons if self.n_neurons is not None else auto_neuron_count(n, d)
        
        # fit a projection
        if self.pairwise_metric is not None:
            self._fit_pairwise_projection(X)
        elif self.density is not None:
            self._fit_sparse_projection(X)
        else:
            self._fit_random_projection(X)
        
        if self.ufunc in ufuncs.keys():
            self.ufunc_ = ufuncs[self.ufunc]
        elif callable(self.ufunc):
            self.ufunc_ = self.ufunc
        else:
            raise ValueError("Ufunc transformation function not understood: ", self.ufunc)

        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        check_is_fitted(self, "is_fitted_")

        X = check_array(X, accept_sparse=True)
        n_features = self.projection_.components_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d" % (X.shape[1], n_features))

        if self.hidden_layer_ == HiddenLayerType.PAIRWISE:
            return self.projection_.transform(X)  # pairwise projection ignores ufunc

        return self.ufunc_(self.projection_.transform(X))



def flatten(items):
    """Yield items from any nested iterable."""
    for x in items:
        # don't break strings into characters
        if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x



def _dense(X):
    if sp.sparse.issparse(X):
        return X.todense()
    else:
        return X

class PairwiseRandomProjection(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=100, pairwise_metric='l2', n_jobs=None, random_state=None):
        """Pairwise distances projection with random centroids.

        Parameters
        ----------
        n_components : int
            Number of components (centroids) in the projection. Creates the same number of output features.

        pairwise_metric : str
            A valid pairwise distance metric, see pairwise-distances_.
            .. _pairwise-distances: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances

        n_jobs : int or None, optional, default=None
            Number of jobs to use in distance computations, or `None` for no parallelism.
            Passed to _pairwise-distances function.

        random_state
            Used for random generation of centroids.
        """
        self.n_components = n_components
        self.pairwise_metric = pairwise_metric
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y=None):
        """Generate artificial centroids.

        Fit a QuantileTransformer to project from data distribution onto a normal one,
        then sample centroids from normal distribution and inverse-project into the data space.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            Input data
        """
        X = check_array(X, accept_sparse=True)
        self.random_state_ = check_random_state(self.random_state)

        if self.n_components <= 0:
            raise ValueError("n_components must be greater than 0, got %s" % self.n_components)

        transformer = QuantileTransformer(n_quantiles=min(100, X.shape[0]), ignore_implicit_zeros=True,
                                          random_state=self.random_state_)
        transformer.fit(X)
        random_centroids = self.random_state_.rand(self.n_components, X.shape[1])
        self.components_ = transformer.inverse_transform(random_centroids)

        self.n_jobs_ = 1 if self.n_jobs is None else self.n_jobs
        return self


    def transform(self, X):
        """Compute distance matrix between input data and the centroids.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            Input data samples.

        Returns
        -------
        X_dist : numpy array
            Distance matrix between input data samples and centroids.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'components_')

        if X.shape[1] != self.components_.shape[1]:
            raise ValueError(
                'Impossible to perform projection: X at fit stage had a different number of features. '
                '(%s != %s)' % (X.shape[1], self.components_.shape[1]))

        try:
            X_dist = pairwise_distances(X, self.components_, n_jobs=self.n_jobs_, metric=self.pairwise_metric)
        except TypeError:
            # scipy distances that don't support sparse matrices
            X_dist = pairwise_distances(_dense(X), _dense(self.components_), n_jobs=self.n_jobs_, metric=self.pairwise_metric)

        return X_dist
    

class _BaseELM(BaseEstimator):

    def __init__(self, alpha=1e-7, batch_size=None, include_original_features=False, n_neurons=None, ufunc="tanh",
                 density=None, pairwise_metric=None, random_state=None):
        self.alpha = alpha
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.ufunc = ufunc
        self.include_original_features = include_original_features
        self.density = density
        self.pairwise_metric = pairwise_metric
        self.random_state = random_state

    def _init_model(self, X):
        """Init an empty model, creating objects for hidden layers and solver.

        Also validates inputs for several hidden layers.
        """
        self.n_features_ = X.shape[1]
        self.solver_ = BatchCholeskySolver(alpha=self.alpha)

        # only one type of neurons
        if not hasattr(self.n_neurons, '__iter__'):
            hl = HiddenLayer(n_neurons=self.n_neurons, density=self.density, ufunc=self.ufunc,
                             pairwise_metric=self.pairwise_metric, random_state=self.random_state)
            hl.fit(X)
            self.hidden_layers_ = (hl, )

        # several different types of neurons
        else:
            k = len(self.n_neurons)

            # fix default values
            ufuncs = self.ufunc
            if isinstance(ufuncs, str) or not hasattr(ufuncs, "__iter__"):
                ufuncs = [ufuncs] * k

            densities = self.density
            if densities is None or not hasattr(densities, "__iter__"):
                densities = [densities] * k

            pw_metrics = self.pairwise_metric
            if pw_metrics is None or isinstance(pw_metrics, str):
                pw_metrics = [pw_metrics] * k

            if not k == len(ufuncs) == len(densities) == len(pw_metrics):
                raise ValueError("Inconsistent parameter lengths for model with {} different types of neurons.\n"
                                 "Set 'ufunc', 'density' and 'pairwise_distances' by lists "
                                 "with {} elements, or leave the default values.".format(k, k))

            self.hidden_layers_ = []
            for n_neurons, ufunc, density, metric in zip(self.n_neurons, ufuncs, densities, pw_metrics):
                hl = HiddenLayer(n_neurons=n_neurons, density=density, ufunc=ufunc,
                                 pairwise_metric=metric, random_state=self.random_state)
                hl.fit(X)
                self.hidden_layers_.append(hl)

    def _reset(self):
        [delattr(self, attr) for attr in ('n_features_', 'solver_', 'hidden_layers_', 'is_fitted_') if hasattr(self, attr)]

    @property
    def coef_(self):
        return self.solver_.coef_

    @property
    def intercept_(self):
        return self.solver_.intercept_

    def partial_fit(self, X, y=None, compute_output_weights=True):
        """Update model with a new batch of data.

        Output weight computation can be temporary turned off for faster processing. This will mark model as
        not fit. Enable `compute_output_weights` in the final call to `partial_fit`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            Training input samples

        y : array-like, shape=[n_samples, n_targets]
            Training targets

        compute_output_weights : boolean, optional, default True
            Whether to compute new output weights (coef_, intercept_). Disable this in intermediate `partial_fit`
            steps to run computations faster, then enable in the last call to compute the new solution.

            .. Note::
                Solution can be updated without extra data by setting `X=None` and `y=None`.

            Example:
                >>> model.partial_fit(X_1, y_1)
                ... model.partial_fit(X_2, y_2)
                ... model.partial_fit(X_3, y_3)    # doctest: +SKIP

            Faster, option 1:
                >>> model.partial_fit(X_1, y_1, compute_output_weights=False)
                ... model.partial_fit(X_2, y_2, compute_output_weights=False)
                ... model.partial_fit(X_3, y_3)    # doctest: +SKIP

            Faster, option 2:
                >>> model.partial_fit(X_1, y_1, compute_output_weights=False)
                ... model.partial_fit(X_2, y_2, compute_output_weights=False)
                ... model.partial_fit(X_3, y_3, compute_output_weights=False)
                ... model.partial_fit(X=None, y=None)    # doctest: +SKIP
        """
        # compute output weights only
        if X is None and y is None and compute_output_weights:
            self.solver_.partial_fit(None, None, compute_output_weights=True)
            self.is_fitted_ = True
            return self

        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)

        if len(y.shape) > 1 and y.shape[1] == 1:
            msg = ("A column-vector y was passed when a 1d array was expected. "
                   "Please change the shape of y to (n_samples, ), for example using ravel().")
            warnings.warn(msg, DataConversionWarning)

        n_samples, n_features = X.shape
        if hasattr(self, 'n_features_') and self.n_features_ != n_features:
            raise ValueError('Shape of input is different from what was seen in `fit`')

        # init model if not fit yet
        if not hasattr(self, 'hidden_layers_'):
            self._init_model(X)

        # set batch size, default is bsize=2000 or all-at-once with less than 10_000 samples
        self.bsize_ = self.batch_size
        if self.bsize_ is None:
            self.bsize_ = n_samples if n_samples < 10 * 1000 else 2000

        # special case of one-shot processing
        if self.bsize_ >= n_samples:
            H = [hl.transform(X) for hl in self.hidden_layers_]
            H = np.hstack(H if not self.include_original_features else [_dense(X)] + H)
            self.solver_.partial_fit(H, y, compute_output_weights=False)

        else:  # batch processing
            for b_start in range(0, n_samples, self.bsize_):
                b_end = min(b_start + self.bsize_, n_samples)
                b_X = X[b_start:b_end]
                b_y = y[b_start:b_end]

                b_H = [hl.transform(b_X) for hl in self.hidden_layers_]
                b_H = np.hstack(b_H if not self.include_original_features else [_dense(b_X)] + b_H)
                self.solver_.partial_fit(b_H, b_y, compute_output_weights=False)

        # output weights if needed
        if compute_output_weights:
            self.solver_.partial_fit(None, None, compute_output_weights=True)
            self.is_fitted_ = True

        # mark as needing a solution
        elif hasattr(self, 'is_fitted_'):
            del self.is_fitted_

        return self

    def fit(self, X, y=None):
        """Reset model and fit on the given data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data samples.
        
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values used as real numbers.

        Returns
        -------
        self : object
            Returns self.         
        """
        
        #todo: add X as bunch of files support
        
        self._reset()
        self.partial_fit(X, y)
        return self

    def predict(self, X):
        """Predict real valued outputs for new inputs X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data samples.
            
        Returns
        -------
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            Predicted outputs for inputs X. 
            
            .. attention::
            
                :mod:`predict` always returns a dense matrix of predicted outputs -- unlike 
                in :meth:`fit`, this may cause memory issues at high number of outputs 
                and very high number of samples. Feed data by smaller batches in such case.
        """
        
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        H = [hl.transform(X) for hl in self.hidden_layers_]
        if self.include_original_features:
            H = [_dense(X)] + H
        H = np.hstack(H)

        return self.solver_.predict(H)


class ELMRegressor(_BaseELM, RegressorMixin):
    """Extreme Learning Machine for regression problems.

    This model solves a regression problem, that is a problem of predicting continuous outputs.
    It supports multi-variate regression (when ``y`` is a 2d array of shape [n_samples, n_targets].)
    ELM uses ``L2`` regularization, and optionally includes the original data features to
    capture linear dependencies in the data natively.

    Parameters
    ----------
    alpha : float
        Regularization strength; must be a positive float. Larger values specify stronger effect.
        Regularization improves model stability and reduces over-fitting at the cost of some learning
        capacity. The same value is used for all targets in multi-variate regression.

        The optimal regularization strength is suggested to select from a large range of logarithmically
        distributed values, e.g. :math:`[10^{-5}, 10^{-4}, 10^{-3}, ..., 10^4, 10^5]`. A small default
        regularization value of :math:`10^{-7}` should always be present to counter numerical instabilities
        in the solution; it does not affect overall model performance.

        .. attention::
            The model may automatically increase the regularization value if the solution
            becomes unfeasible otherwise. The actual used value contains in ``alpha_`` attribute.

    batch_size : int, optional
        Actual computations will proceed in batches of this size, except the last batch that may be smaller.
        Default behavior is to process all data at once with <10,000 samples, otherwise use batches
        of size 2000.

    include_original_features : boolean, default=False
        Adds extra hidden layer neurons that simpy copy the input data features, adding a linear part
        to the final model solution that can directly capture linear relations between data and
        outputs. Effectively increases `n_neurons` by `n_inputs` leading to a larger model.
        Including original features is generally a good thing if the number of data features is low.

    n_neurons : int or [int], optional
        Number of hidden layer neurons in ELM model, controls model size and learning capacity.
        Generally number of neurons should be less than the number of training data samples, as
        otherwise the model will learn the training set perfectly resulting in overfitting.

        Several different kinds of neurons can be used in the same model by specifying a list of
        neuron counts. ELM will create a separate neuron type for each element in the list.
        In that case, the following attributes ``ufunc``, ``density`` and ``pairwise_metric``
        should be lists of the same length; default values will be automatically expanded into a list.

        .. note::
            Models with <1,000 neurons are very fast to compute, while GPU acceleration is efficient
            starting from 1,000-2,000 neurons. A standard computer should handle up to 10,000 neurons.
            Very large models will not fit in memory but can still be trained by an out-of-core solver.

    ufunc : {'tanh', 'sigm', 'relu', 'lin' or callable}, or a list of those (see n_neurons)
        Transformation function of hidden layer neurons. Includes the following options:
            - 'tanh' for hyperbolic tangent
            - 'sigm' for sigmoid
            - 'relu' for rectified linear unit (clamps negative values to zero)
            - 'lin' for linear neurons, transformation function does nothing
            - any custom callable function like members of ``Numpu.ufunc``

    density : float in range (0, 1], or a list of those (see n_neurons), optional
        Specifying density replaces dense projection layer by a sparse one with the specified
        density of the connections. For instance, ``density=0.1`` means each hidden neuron will
        be connected to a random 10% of input features. Useful for working on very high-dimensional
        data, or for large numbers of neurons.

    pairwise_metric : {'euclidean', 'cityblock', 'cosine' or other}, or a list of those (see n_neurons), optional
        Specifying pairwise metric replaces multiplicative hidden neurons by distance-based hidden
        neurons. This ELM model is known as Radial Basis Function ELM (RBF-ELM).

        .. note::
            Pairwise function neurons ignore ufunc and density.

        Typical metrics are `euclidean`, `cityblock` and `cosine`. For a full list of metrics check
        the `webpage <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html>`_
        of :mod:`sklearn.metrics.pairwise_distances`.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when generating random numbers e.g.
        for hidden neuron parameters. Random state instance is passed to lower level objects and routines.
        Use it for repeatable experiments.

    Attributes
    ----------
    n_neurons_ : int
        Number of automatically generated neurons.

    ufunc_ : function
        Tranformation function of hidden neurons.

    projection_ : object
        Hidden layer projection function.

    solver_ : object
        Solver instance, read solution from there.


    Examples
    --------

    Combining ten sigmoid and twenty RBF neurons in one model:

    >>> model = ELMRegressor(n_neurons=(10, 20),
    ...                      ufunc=('sigm', None),
    ...                      density=(None, None),
    ...                      pairwise_metric=(None, 'euclidean'))   # doctest: +SKIP

    Default values in multi-neuron ELM are automatically expanded to a list

    >>>  model = ELMRegressor(n_neurons=(10, 20),
    ...                       ufunc=('sigm', None),
    ...                       pairwise_metric=(None, 'euclidean'))   # doctest: +SKIP

    >>>  model = ELMRegressor(n_neurons=(30, 30),
    ...                       pairwise_metric=('cityblock', 'cosine'))   # doctest: +SKIP
    """
    pass


class ELMClassifier(_BaseELM, ClassifierMixin):
    """ELM classifier, modified for multi-label classification support.

    :param classes: Set of classes to consider in the model; can be expanded at runtime.
                    Samples of other classes will have their output set to zero.
    :param solver: Solver to use, "default" for build-in Least Squares or "ridge" for Ridge regression
    

    Example descr...

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    
    def __init__(self, classes=None, alpha=1e-7, batch_size=None, include_original_features=False, n_neurons=None,
                 ufunc="tanh", density=None, pairwise_metric=None, random_state=None):
        super().__init__(alpha, batch_size, include_original_features, n_neurons, ufunc, density, pairwise_metric,
                         random_state)
        self.classes = classes


    @property
    def classes_(self):
        return self.label_binarizer_.classes_

    def _get_tags(self):
        return {"multioutput": True, "multilabel": True}

    def _update_classes(self, y):
        if not isinstance(self.solver_, BatchCholeskySolver):
            raise ValueError("Only iterative solver supports dynamic class update")

        old_classes = self.label_binarizer_.classes_
        partial_classes = clone(self.label_binarizer_).fit(y).classes_

        # no new classes detected
        if set(partial_classes) <= set(old_classes):
            return

        if len(old_classes) < 3:
            raise ValueError("Dynamic class update has to start with at least 3 classes to function correctly; "
                             "provide 3 or more 'classes=[...]' during initialization.")

        # get new classes sorted by LabelBinarizer
        self.label_binarizer_.fit(np.hstack((old_classes, partial_classes)))
        new_classes = self.label_binarizer_.classes_

        # convert existing XtY matrix to new classes
        if hasattr(self.solver_, 'XtY_'):
            XtY_old = self.solver_.XtY_
            XtY_new = np.zeros((XtY_old.shape[0], new_classes.shape[0]))
            for i, c in enumerate(old_classes):
                j = np.where(new_classes == c)[0][0]
                XtY_new[:, j] = XtY_old[:, i]
            self.solver_.XtY_ = XtY_new

        # reset the solution
        if hasattr(self.solver_, 'is_fitted_'):
            del self.solver_.is_fitted_

    def partial_fit(self, X, y=None, classes=None, update_classes=False, compute_output_weights=True):
        """Update classifier with new data.

        :param classes: ignored
        :param update_classes: Includes new classes from 'y' into the model;
                               assumes they are set to 0 in all previous targets.
        """

        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)

        # init label binarizer if needed
        if not hasattr(self, 'label_binarizer_'):
            self.label_binarizer_ = LabelBinarizer()
            if type_of_target(y).endswith("-multioutput"):
                self.label_binarizer_ = MultiLabelBinarizer()
            self.label_binarizer_.fit(self.classes if self.classes is not None else y)

        if update_classes:
            self._update_classes(y)

        y_numeric = self.label_binarizer_.transform(y)
        super().partial_fit(X, y_numeric, compute_output_weights=compute_output_weights)
        return self


    def fit(self, X, y=None):
        """Fit a classifier erasing any previously trained model.

        Returns
        -------
        self : object
            Returns self.
        """
        if hasattr(self, "label_binarizer_"):
            del self.label_binarizer_
        self.partial_fit(X, y, compute_output_weights=True)
        return self


    def predict(self, X):
        """Predict classes of new inputs X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            Returns one most probable class for multi-class problem, or
            a binary vector of all relevant classes for multi-label problem.
        """    
            
        check_is_fitted(self, "is_fitted_")
        scores = super().predict(X)
        return self.label_binarizer_.inverse_transform(scores)