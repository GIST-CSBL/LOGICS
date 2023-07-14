"""
    These codes are modification of some of DeepChem package's modules:
        https://github.com/deepchem/deepchem
    
    We used this code for scaffold split for the datasets.
        
    @book{Ramsundar-et-al-2019,
        title={Deep Learning for the Life Sciences},
        author={Bharath Ramsundar and Peter Eastman and Patrick Walters and Vijay Pande and Karl Leswing and Zhenqin Wu},
        publisher={O'Reilly Media},
        note={\url{https://www.amazon.com/Deep-Learning-Life-Sciences-Microscopy/dp/1492039837}},
        year={2019}
    }
}
"""

import csv
import numpy as np
import pandas as pd
import logging
from typing import Iterator, List, Optional, Tuple
from numpy.typing import ArrayLike

Batch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

logger = logging.getLogger(__name__)

### https://github.com/deepchem/deepchem/blob/master/deepchem/data/datasets.py
class Dataset(object):
    """Abstract base class for datasets defined by X, y, w elements.

    `Dataset` objects are used to store representations of a dataset as
    used in a machine learning task. Datasets contain features `X`,
    labels `y`, weights `w` and identifiers `ids`. Different subclasses
    of `Dataset` may choose to hold `X, y, w, ids` in memory or on disk.

    The `Dataset` class attempts to provide for strong interoperability
    with other machine learning representations for datasets.
    Interconversion methods allow for `Dataset` objects to be converted
    to and from numpy arrays, pandas dataframes, tensorflow datasets,
    and pytorch datasets (only to and not from for pytorch at present).

    Note that you can never instantiate a `Dataset` object directly.
    Instead you will need to instantiate one of the concrete subclasses.
    """

    def __init__(self) -> None:
        raise NotImplementedError()

    def __len__(self) -> int:
        """Get the number of elements in the dataset.

        Returns
        -------
        int
            The number of elements in the dataset.
        """
        raise NotImplementedError()

    def get_shape(self):
        """Get the shape of the dataset.

        Returns four tuples, giving the shape of the X, y, w, and ids
        arrays.

        Returns
        -------
        Tuple
            The tuple contains four elements, which are the shapes of
            the X, y, w, and ids arrays.
        """
        raise NotImplementedError()

    def get_task_names(self) -> np.ndarray:
        """Get the names of the tasks associated with this dataset."""
        raise NotImplementedError()

    @property
    def X(self) -> np.ndarray:
        """Get the X vector for this dataset as a single numpy array.

        Returns
        -------
        np.ndarray
            A numpy array of identifiers `X`.

        Note
        ----
        If data is stored on disk, accessing this field may involve loading
        data from disk and could potentially be slow. Using
        `iterbatches()` or `itersamples()` may be more efficient for
        larger datasets.
        """
        raise NotImplementedError()

    @property
    def y(self) -> np.ndarray:
        """Get the y vector for this dataset as a single numpy array.

        Returns
        -------
        np.ndarray
            A numpy array of identifiers `y`.

        Note
        ----
        If data is stored on disk, accessing this field may involve loading
        data from disk and could potentially be slow. Using
        `iterbatches()` or `itersamples()` may be more efficient for
        larger datasets.
        """
        raise NotImplementedError()

    @property
    def ids(self) -> np.ndarray:
        """Get the ids vector for this dataset as a single numpy array.

        Returns
        -------
        np.ndarray
            A numpy array of identifiers `ids`.

        Note
        ----
        If data is stored on disk, accessing this field may involve loading
        data from disk and could potentially be slow. Using
        `iterbatches()` or `itersamples()` may be more efficient for
        larger datasets.
        """
        raise NotImplementedError()

    def iterbatches(self,
                    batch_size: Optional[int] = None,
                    epochs: int = 1,
                    deterministic: bool = False,
                    pad_batches: bool = False) -> Iterator[Batch]:
        """Get an object that iterates over minibatches from the dataset.

        Each minibatch is returned as a tuple of four numpy arrays:
        `(X, y, w, ids)`.

        Parameters
        ----------
        batch_size: int, optional (default None)
            Number of elements in each batch.
        epochs: int, optional (default 1)
            Number of epochs to walk over dataset.
        deterministic: bool, optional (default False)
            If True, follow deterministic order.
        pad_batches: bool, optional (default False)
            If True, pad each batch to `batch_size`.

        Returns
        -------
        Iterator[Batch]
            Generator which yields tuples of four numpy arrays `(X, y, w, ids)`.
        """
        raise NotImplementedError()

    def itersamples(self) -> Iterator[Batch]:
        """Get an object that iterates over the samples in the dataset.

        Examples
        --------
        >>> dataset = NumpyDataset(np.ones((2,2)))
        >>> for x, y, w, id in dataset.itersamples():
        ...   print(x.tolist(), y.tolist(), w.tolist(), id)
        [1.0, 1.0] [0.0] [0.0] 0
        [1.0, 1.0] [0.0] [0.0] 1
        """
        raise NotImplementedError()

    def to_dataframe(self) -> pd.DataFrame:
        """Construct a pandas DataFrame containing the data from this Dataset.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe. If there is only a single feature per datapoint,
            will have column "X" else will have columns "X1,X2,..." for
            features.  If there is only a single label per datapoint, will
            have column "y" else will have columns "y1,y2,..." for labels. If
            there is only a single weight per datapoint will have column "w"
            else will have columns "w1,w2,...". Will have column "ids" for
            identifiers.
        """
        X = self.X
        y = self.y
        w = self.w
        ids = self.ids
        if len(X.shape) == 1 or X.shape[1] == 1:
            columns = ['X']
        else:
            columns = [f'X{i+1}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=columns)
        if len(y.shape) == 1 or y.shape[1] == 1:
            columns = ['y']
        else:
            columns = [f'y{i+1}' for i in range(y.shape[1])]
        y_df = pd.DataFrame(y, columns=columns)
        if len(w.shape) == 1 or w.shape[1] == 1:
            columns = ['w']
        else:
            columns = [f'w{i+1}' for i in range(w.shape[1])]
        w_df = pd.DataFrame(w, columns=columns)
        ids_df = pd.DataFrame(ids, columns=['ids'])
        return pd.concat([X_df, y_df, w_df, ids_df], axis=1, sort=False)

    def to_csv(self, path: str) -> None:
        """Write object to a comma-seperated values (CSV) file

        Example
        -------
        >>> import numpy as np
        >>> X = np.random.rand(10, 10)
        >>> dataset = dc.data.DiskDataset.from_numpy(X)
        >>> dataset.to_csv('out.csv')  # doctest: +SKIP

        Parameters
        ----------
        path: str
            File path or object

        Returns
        -------
        None
        """
        columns = []
        X_shape, y_shape, w_shape, id_shape = self.get_shape()
        assert len(
            X_shape) == 2, "dataset's X values should be scalar or 1-D arrays"
        assert len(
            y_shape) == 2, "dataset's y values should be scalar or 1-D arrays"
        if X_shape[1] == 1:
            columns.append('X')
        else:
            columns.extend([f'X{i+1}' for i in range(X_shape[1])])
        if y_shape[1] == 1:
            columns.append('y')
        else:
            columns.extend([f'y{i+1}' for i in range(y_shape[1])])
        if w_shape[1] == 1:
            columns.append('w')
        else:
            columns.extend([f'w{i+1}' for i in range(w_shape[1])])
        columns.append('ids')
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            for (x, y, w, ids) in self.itersamples():
                writer.writerow(list(x) + list(y) + list(w) + [ids])
        return None

class NumpyDataset(Dataset):
    """A Dataset defined by in-memory numpy arrays.

    This subclass of `Dataset` stores arrays `X,y,w,ids` in memory as
    numpy arrays. This makes it very easy to construct `NumpyDataset`
    objects.

    Examples
    --------
    >>> import numpy as np
    >>> dataset = NumpyDataset(X=np.random.rand(5, 3), y=np.random.rand(5,), ids=np.arange(5))
    """

    def __init__(self,
                 X: ArrayLike,
                 y: Optional[ArrayLike] = None,
                 w: Optional[ArrayLike] = None,
                 ids: Optional[ArrayLike] = None,
                 n_tasks: int = 1) -> None:
        """Initialize this object.

        Parameters
        ----------
        X: np.ndarray
            Input features. A numpy array of shape `(n_samples,...)`.
        y: np.ndarray, optional (default None)
            Labels. A numpy array of shape `(n_samples, ...)`. Note that each label can
            have an arbitrary shape.
        w: np.ndarray, optional (default None)
            Weights. Should either be 1D array of shape `(n_samples,)` or if
            there's more than one task, of shape `(n_samples, n_tasks)`.
        ids: np.ndarray, optional (default None)
            Identifiers. A numpy array of shape `(n_samples,)`
        n_tasks: int, default 1
            Number of learning tasks.
        """
        n_samples = np.shape(X)[0]
        if n_samples > 0:
            if y is None:
                # Set labels to be zero, with zero weights
                y = np.zeros((n_samples, n_tasks), np.float32)
                w = np.zeros((n_samples, 1), np.float32)
        if ids is None:
            ids = np.arange(n_samples)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if w is None:
            if len(y.shape) == 1:
                w = np.ones(y.shape[0], np.float32)
            else:
                w = np.ones((y.shape[0], 1), np.float32)
        if not isinstance(w, np.ndarray):
            w = np.array(w)
        self._X = X
        self._y = y
        self._w = w
        self._ids = np.array(ids, dtype=object)

    def __len__(self) -> int:
        """Get the number of elements in the dataset."""
        return len(self._y)

    def get_shape(self):
        """Get the shape of the dataset.

        Returns four tuples, giving the shape of the X, y, w, and ids arrays.
        """
        return self._X.shape, self._y.shape, self._w.shape, self._ids.shape

    def get_task_names(self) -> np.ndarray:
        """Get the names of the tasks associated with this dataset."""
        if len(self._y.shape) < 2:
            return np.array([0])
        return np.arange(self._y.shape[1])

    @property
    def X(self) -> np.ndarray:
        """Get the X vector for this dataset as a single numpy array."""
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Get the y vector for this dataset as a single numpy array."""
        return self._y

    @property
    def ids(self) -> np.ndarray:
        """Get the ids vector for this dataset as a single numpy array."""
        return self._ids

    @property
    def w(self) -> np.ndarray:
        """Get the weight vector for this dataset as a single numpy array."""
        return self._w

    def itersamples(self) -> Iterator[Batch]:
        """Get an object that iterates over the samples in the dataset.

        Returns
        -------
        Iterator[Batch]
            Iterator which yields tuples of four numpy arrays `(X, y, w, ids)`.

        Examples
        --------
        >>> dataset = NumpyDataset(np.ones((2,2)))
        >>> for x, y, w, id in dataset.itersamples():
        ...   print(x.tolist(), y.tolist(), w.tolist(), id)
        [1.0, 1.0] [0.0] [0.0] 0
        [1.0, 1.0] [0.0] [0.0] 1
        """
        n_samples = self._X.shape[0]
        return ((self._X[i], self._y[i], self._w[i], self._ids[i])
                for i in range(n_samples))    
    
def _generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Compute the Bemis-Murcko scaffold for a SMILES string.

    Bemis-Murcko scaffolds are described in DOI: 10.1021/jm9602928.
    They are essentially that part of the molecule consisting of
    rings and the linker atoms between them.

    Paramters
    ---------
    smiles: str
        SMILES
    include_chirality: bool, default False
        Whether to include chirality in scaffolds or not.

    Returns
    -------
    str
        The MurckScaffold SMILES from the original SMILES

    References
    ----------
    .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
        1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.

    Note
    ----
    This function requires RDKit to be installed.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")

    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

### https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
# removed Splitter inheritance
class ScaffoldSplitter():
    """Class for doing data splits based on the scaffold of small molecules.

    Group  molecules  based on  the Bemis-Murcko scaffold representation, which identifies rings,
    linkers, frameworks (combinations between linkers and rings) and atomic properties  such as
    atom type, hibridization and bond order in a dataset of molecules. Then split the groups by
    the number of molecules in each group in decreasing order.

    It is necessary to add the smiles representation in the ids field during the
    DiskDataset creation.

    Examples
    ---------
    >>> import deepchem as dc
    >>> # creation of demo data set with some smiles strings
    ... data_test= ["CC(C)Cl" , "CCC(C)CO" ,  "CCCCCCCO" , "CCCCCCCC(=O)OC" , "c3ccc2nc1ccccc1cc2c3" , "Nc2cccc3nc1ccccc1cc23" , "C1CCCCCC1" ]
    >>> Xs = np.zeros(len(data_test))
    >>> Ys = np.ones(len(data_test))
    >>> # creation of a deepchem dataset with the smile codes in the ids field
    ... dataset = dc.data.DiskDataset.from_numpy(X=Xs,y=Ys,w=np.zeros(len(data_test)),ids=data_test)
    >>> scaffoldsplitter = dc.splits.ScaffoldSplitter()
    >>> train,test = scaffoldsplitter.train_test_split(dataset)
    >>> train
    <DiskDataset X.shape: (5,), y.shape: (5,), w.shape: (5,), ids: ['CC(C)Cl' 'CCC(C)CO' 'CCCCCCCO' 'CCCCCCCC(=O)OC' 'C1CCCCCC1'], task_names: [0]>

    References
    ----------
    .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
        1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.

    Note
    ----
    This class requires RDKit to be installed.
    """

    def split(
        self,
        dataset: Dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = None,
        log_every_n: Optional[int] = 1000
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits internal compounds into train/validation/test by scaffold.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        frac_train: float, optional (default 0.8)
            The fraction of data to be used for the training split.
        frac_valid: float, optional (default 0.1)
            The fraction of data to be used for the validation split.
        frac_test: float, optional (default 0.1)
            The fraction of data to be used for the test split.
        seed: int, optional (default None)
            Random seed to use.
        log_every_n: int, optional (default 1000)
            Controls the logger by dictating how often logger outputs
            will be produced.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            A tuple of train indices, valid indices, and test indices.
            Each indices is a list of integers.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
        scaffold_sets = self.generate_scaffolds(dataset)

        train_cutoff = frac_train * len(dataset)
        valid_cutoff = (frac_train + frac_valid) * len(dataset)
        train_inds: List[int] = []
        valid_inds: List[int] = []
        test_inds: List[int] = []

        logger.info("About to sort in scaffold sets")
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(
                        scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set
        return train_inds, valid_inds, test_inds

    def generate_scaffolds(self,
                           dataset: Dataset,
                           log_every_n: int = 1000) -> List[List[int]]:
        """Returns all scaffolds from the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        log_every_n: int, optional (default 1000)
            Controls the logger by dictating how often logger outputs
            will be produced.

        Returns
        -------
        scaffold_sets: List[List[int]]
            List of indices of each scaffold in the dataset.
        """
        scaffolds = {}
        data_len = len(dataset)

        logger.info("About to generate scaffolds")
        for ind, smiles in enumerate(dataset.ids):
            if ind % log_every_n == 0:
                logger.info("Generating scaffold %d/%d" % (ind, data_len))
            scaffold = _generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (scaffold,
                 scaffold_set) in sorted(scaffolds.items(),
                                         key=lambda x: (len(x[1]), x[1][0]),
                                         reverse=True)
        ]
        return scaffold_sets
