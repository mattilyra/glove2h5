from pathlib import Path
import zipfile
import tarfile
import tempfile
import shutil

import numpy as np
import h5py
import sqlitedict

class GloVe2H5:
    def __init__(self, path, collection=''):
        self.path = Path(path)
        self.id2token = {}
        self.collection = collection

    @staticmethod
    def _extract_vocab_from_stanford_zip(zipfh, zipinfo, vocab):
        with zipfh.open(zipinfo) as inputfile:
            num_entries = 0
            for i_row, row in enumerate(inputfile):
                row = row.decode('utf-8').strip()
                parts = row.split()
                token = parts[0]
                vocab[token] = num_entries
                num_entries += 1
            vocab.commit()
        return num_entries

    @staticmethod
    def _extract_vectors_from_stanford_zip(zipfh, zipinfo, vocab, h5_dataset):
        with zipfh.open(zipinfo) as inputfile:
            for i_row, row in enumerate(inputfile):
                row = row.decode('utf-8').strip()
                parts = row.split()
                token = parts[0]
                vec = np.asarray(parts[1:], dtype=np.float64)
                h5_dataset[vocab[token]] = vec

    @staticmethod
    def create_from(datafile, compression='lzf'):
        """Initialise the H5 container and vocabulary from the original Stanford ZIP files"""

        output_path = Path(datafile).expanduser().parent

        if not output_path.exists():
            output_path.parent.mkdir()
        vector_dimensions = 0

        output_file = Path(datafile).with_suffix('.glove2h5')
        try:
            output_file.mkdir()
        except FileExistsError: pass

        h5_path = Path(output_file / 'vectors.h5')
        vocab_path = Path(output_file / 'vocab.sqlite')
        vocab_rev_path = Path(output_file / 'vocab_rev.sqlite')

        with zipfile.ZipFile(datafile, 'r') as zipfh,\
             h5py.File(h5_path, 'w', ) as h5fh:
            zipfiles = zipfh.filelist
            for zipinfo in zipfiles[:1]:
                try:
                    vocab = sqlitedict.SqliteDict(str(vocab_path), autocommit=False, flag='w')
                    num_entries = GloVe2H5._extract_vocab_from_stanford_zip(zipfh, zipinfo, vocab)
                    vocab.commit()
                finally:
                    vocab.close()

            vocab = sqlitedict.SqliteDict(str(vocab_path), autocommit=False, flag='r')

            for zipinfo in zipfiles:
                with zipfh.open(zipinfo) as inputfile:
                    parts = inputfile.readline().decode('utf-8').strip().split()
                    D = len(parts) - 1

                dataset_name = Path(zipinfo.filename).stem
                h5_dataset = h5fh.create_dataset(dataset_name, (num_entries, D), dtype=np.float64, compression=compression)
                GloVe2H5._extract_vectors_from_stanford_zip(zipfh, zipinfo, vocab, h5_dataset)

        vocab.close()

        vector_dimensions = len(parts) - 1
        return GloVe2H5(output_file, collection=zipfiles[0])

    def __getitem__(self, entry):
        vocab = sqlitedict.SqliteDict(str(self.path / 'vocab.sqlite'), autocommit=False, flag='r')
        entry_ = Path(entry)
        with h5py.File(self.path / 'vectors.h5', mode='r') as h5:
            if entry_.name in vocab:
                token_idx = vocab[entry_.name]
                parent = self.collection if entry_.parent == Path('.') else entry_.parent
                if parent == '.':
                    raise RuntimeError('HDF5 dataset name not defined, either set a default \'collection=\' in constructor or define the access key as \'d[\'collection/entry\']\'')
                v = h5[parent][token_idx]
            else:
                raise KeyError(f'Entry {entry} not found in vocabulary.')
        vocab.close()
        return v