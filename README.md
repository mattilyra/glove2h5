# GloVe2H5
A small utility for converting Stanford GloVe vectors to HDF5 / NumPy

# Usage

Assuming you've downloaded the GloVe vectors from https://nlp.stanford.edu/projects/glove/
into `./GloVe`

Convert all GloVe vectors in `glove.6B.zip` to NumPy and store in an HDF5 file.
The call below creates a vocabulary stored in a `sqlitedict.SqliteDict` on
extracts the GloVe vectors into an `.h5` file. The results are stored in the
current directory under `$SOURCEFILE.glove2h5` where `$SOURCEFILE` is the name
of the original source file, `glove.6B.zip` below without the extension.

```python
glove2h5 = GloVe2H5.create_from('./GloVe/glove.6B.zip', compression='lzf')

# get the 50 dimensional vector for 'sample'
glove2h5['glove.50d/sample']

# get the 100 dimensional vector for 'sample'
glove2h5['glove.100d/sample']

```

## Extract only certain dimensional vectors

The `glove.6B.txt.zip` file contains vectors in 50, 100, 200 and 300. Extracting
all of them into `HDF5` is unnecessary if you only need some of them.

```python
# extract only the 100 dimensional vectors
glove2h5_100d = GloVe2H5.create_from('./GloVe/glove.6B.zip', collections['glove.6B.100d.txt'], compression='lzf')`
```

# Requirements

Python 3.6
`h5py`
`sqlitedict`
