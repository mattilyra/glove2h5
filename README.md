# GloVe2H5
A small utility for converting Stanford GloVe vectors to HDF5 / NumPy. The pretrained
Stanford vectors are distributed as zipped text files with one line per vector. This is
not the most convenient way of interacting with the vectors, this utility converts the
zip files into NumPy arrays contained in HDF5 (using `h5py`) files with a separate sqllite
dictionary for the vocabulry.

The GloVe code (in `C`) is available on github https://github.com/stanfordnlp/GloVe and you
can download the pretrained Stanford GloVe vectors from https://nlp.stanford.edu/projects/glove/.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.
https://nlp.stanford.edu/pubs/glove.pdf

# Install & Usage

### CLI

Extract the 50-dimensional word vectors using LZF compression for HDF5

```
$ git clone https://github.com/mattilyra/glove2h5.git
$ cd glove2h5
$ python -m glove2h5 ~/Downloads/glove.6B.txt.zip --collection glove.6B.50d.txt --compression lzf
```

---

Assuming you've downloaded the GloVe vectors from https://nlp.stanford.edu/projects/glove/
into `./GloVe`

Convert all GloVe vectors in `glove.6B.zip` to NumPy and store in an HDF5 file.
The call below creates a vocabulary stored in a `sqlitedict.SqliteDict` and
extracts the GloVe vectors into an `vectors.h5` file. The results are stored in the
current directory under `$SOURCEFILE.glove2h5` where `$SOURCEFILE` is the name
of the original source file, `glove.6B.zip` without the file extension. A separate
vocabulary is created to allow indexing the vectors from HDF5.

```python
glove2h5 = GloVe2H5.create_from('./GloVe/glove.6B.zip', compression='lzf')

# get the 50 dimensional vector for 'sample'
glove2h5['glove.50d/sample']

# get the 100 dimensional vector for 'sample'
glove2h5['glove.100d/sample']

```

## Extract only certain dimensional vectors

The `glove.6B.txt.zip` file contains vectors in 50, 100, 200 and 300 dimensions. Each
of these is stored in a separate file in the zip archive.

```
- glove.6B.zip
-- glove.6B.50d.txt    # 50 dimensional vectors
-- glove.6B.100d.txt   # 100 dimensional vectors
-- glove.6B.200d.txt   # 200 dimensional vectors
-- glove.6B.300d.txt   # 300 dimensional vectors
```

Extracting all of them into `HDF5` is unnecessary (and obivously slow) if you only need
some of them. You can provide a keyword to `create_from` to only extract certain files
contained in the zip archive.

```python
# extract only the 100 dimensional vectors
glove2h5_100d = GloVe2H5.create_from('./GloVe/glove.6B.zip', collections=['glove.6B.100d.txt'], compression='lzf')`

# the collection is defined automatically as 'glove.6B.100D'
glove2h5['sample']
```

# Load already extracted vectors

You can load an earlier extracted set of vectors by just calling the constructor

```python
glove2h5_100d = GloVe2H5('./GloVe/glove.6B.zip', collection='glove.6B.100d.txt')`

# the collection was defined to be 'glove.6B.100D' so we don't need it for __getitem__ anymore
glove2h5['sample']
```


# Requirements

- Python 3.6
- `h5py`
- `sqlitedict`
