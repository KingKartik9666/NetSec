"""Memory-bounded train-only neural scaling and randomized PCA."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,MinMaxScaler

from framework.training.neural import signed_log1p


class StreamedNeuralPreprocessor:
    def __init__(self,prefix,pca=None):
        self.prefix=prefix;self.pca=pca

    def transform(self,X):
        values=self.prefix.transform(X)
        return self.pca.transform(values) if self.pca is not None else values


def fit_streamed_preprocessing(family,X,V,config,seed,directory,batch_size=4096):
    """Exact all-training medians/minima/maxima; bounded float32 PCA workspace.

    Randomized PCA fits every training row. Its algorithm is an explicitly
    separate capacity/preprocessing choice, not the original author's PCA.
    """
    if config.get('scaling')!='log_minmax':raise ValueError('Expected signed-log MinMax recipe')
    imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(X)
    scaler=MinMaxScaler(feature_range=(-1,1) if family=='kan' else (0,1),clip=True)
    for start in range(0,len(X),batch_size):
        scaler.partial_fit(signed_log1p(imputer.transform(X[start:start+batch_size])))
    prefix=Pipeline([('imputer',imputer),('log',FunctionTransformer(signed_log1p)),('scaler',scaler)])
    cache=np.lib.format.open_memmap(directory/'scaled_training_cache.npy',mode='w+',dtype='float32',shape=X.shape)
    for start in range(0,len(X),batch_size):cache[start:start+batch_size]=prefix.transform(X[start:start+batch_size])
    cache.flush()
    if family=='hem':
        components=min(config.get('pca_components',64),X.shape[1],len(X)-1)
        # copy=False may modify the private cache during centering/variance work.
        # Always project fresh scaled chunks; original prepared X is read-only.
        pca=PCA(n_components=components,svd_solver='randomized',iterated_power=2,random_state=seed,copy=False).fit(cache)
        cache._mmap.close()
        train=np.empty((len(X),components),dtype=np.float32)
        for start in range(0,len(X),batch_size):
            train[start:start+batch_size]=pca.transform(prefix.transform(X[start:start+batch_size]))
        preprocessing=StreamedNeuralPreprocessor(prefix,pca)
    else:
        train=cache;preprocessing=StreamedNeuralPreprocessor(prefix)
    val=np.concatenate([preprocessing.transform(V[start:start+batch_size]).astype(np.float32)
                        for start in range(0,len(V),batch_size)])
    # Sample proof checks the centered-cache training calculation against the saved pipeline.
    sample=np.linspace(0,len(X)-1,min(128,len(X)),dtype=np.int64)
    np.testing.assert_allclose(preprocessing.transform(X[sample]),train[sample],atol=2e-5,rtol=1e-5)
    return preprocessing,train,val
