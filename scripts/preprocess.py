import numpy as np
import pandas as pd

def sample_negative_examples(neg_seq_arr, pos_seq_arr, num_samples=500, seq_length=17, seed=14):
    '''
    This function samples negative sequence examples when a class imbalances
    exists with more negative examples than postive examples. This function
    ensures that none of the negative examples are in the positive examples.

    Args:
        neg_seq_arr (array-like): Numpy array of negative sequences.
        pos_seq_arr (array-like): Numpy array of positive sequences.
        num_samples (int, default=500): Number of negative samples to return.
        seq_length (int, default=17): Length of sequence to return.

    Returns:
        negative_samples (array-like): Numpy array of negative sequence examples.
    '''
    # defining positive sequence set to ensure negative sequences do not
    # overlap with the postive sequences
    np.random.seed(seed)
    pos_seq_set = set(list(pos_seq_arr))
    neg_seq_set = set()
    negative_samples = []
    if num_samples < neg_seq_arr.shape[0]:
        samples_per_seq = 1
    else:
        samples_per_seq = len(neg_seq_arr) // num_samples + 1
    for neg_full_seq in neg_seq_arr:
        k = 0
        while k <= samples_per_seq:
            full_seq_len = len(neg_full_seq)
            assert full_seq_len > seq_length
            rand_idx = np.random.choice(np.arange(full_seq_len - seq_length))
            neg_sample = neg_full_seq[rand_idx:(rand_idx+seq_length)]
            if neg_sample not in pos_seq_set and neg_sample not in neg_seq_set:
                negative_samples.append(neg_sample)
                neg_seq_set.add(neg_sample)
                k += 1

    if len(negative_samples) > num_samples:
        negative_samples = np.random.choice(np.array(negative_samples), num_samples, replace=False)
    else:
        negative_samples = np.array(negative_samples)
    return negative_samples


def encode_seqs(seq_arr):
    '''
    '''
    dna_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    all_encodings = []
    for seq in seq_arr:
        seq = np.array(list(seq))
        values = np.vectorize(dna_dict.get)(seq)
        encoding = np.zeros((values.shape[0], 4))
        encoding[np.arange(values.shape[0]), values] = 1
        encoding = np.ravel(encoding)
        all_encodings.append(encoding)
    all_encodings = np.array(all_encodings)
    return all_encodings


class KFoldCV:
    '''
    '''
    def __init__(self, X, y, num_folds=5, shuffle=True):
        self.num_folds = num_folds
        full_ds = np.hstack([X, y])
        colnames = list(range(X.shape[1]))
        colnames = [str(x) for x in colnames]
        colnames.append('label')
        df = pd.DataFrame(full_ds, columns=colnames)
        split_X = []
        split_y = []
        for grp_name, grp in df.groupby('label'):
            if shuffle == True:
                grp = grp.sample(frac=1)
                grp = grp.reset_index(drop=True)
                grp_values = grp.values[:, :-1]
            assert num_folds < len(grp)
            split_grp = np.array_split(grp_values, num_folds, axis=0)
            split_y.append([np.ones(X.shape[0])*int(grp_name) for X in split_grp])
            split_X.append(split_grp)
        folds_X = []
        folds_y = []
        for fold in range(len(split_X[0])):
            fold_X = np.vstack([split_X[i][fold] for i in range(len(split_X))])
            fold_y = np.hstack([split_y[i][fold] for i in range(len(split_y))])
            fold_y = np.expand_dims(fold_y, axis=1)
            full_fold = np.hstack([fold_X, fold_y])
            np.random.shuffle(full_fold)
            fold_X = full_fold[:, :-1]
            fold_y = full_fold[:, -1]
            folds_X.append(fold_X)
            folds_y.append(fold_y)
        self._folds_X = folds_X
        self._folds_y = folds_y

    def get_fold(self, fold=0):
        '''
        '''
        assert fold <= self.num_folds
        split_X = self._folds_X.copy()
        split_y = self._folds_y.copy()
        X_test = split_X.pop(fold)
        y_test = split_y.pop(fold)
        y_test = np.expand_dims(y_test, axis=1)
        X_train = np.vstack(split_X)
        y_train = np.hstack(split_y)
        y_train = np.expand_dims(y_train, axis=1)
        return X_train, X_test, y_train, y_test

def split_basic_binary(X, y, split=[0.8, 0.2], shuffle=True):
    '''
    '''
    neg_ds = X[np.squeeze(y == 0), :]
    split_idx_neg = int(np.round(split[0]*neg_ds.shape[0]))
    X_train = neg_ds[:split_idx_neg, :]
    y_train = np.expand_dims(np.array([0]*X_train.shape[0]), axis=1)
    X_test = neg_ds[split_idx_neg:, :]
    y_test = np.expand_dims(np.array([0]*X_test.shape[0]), axis=1)

    pos_ds = X[np.squeeze(y == 1), :]
    split_idx_pos = int(np.round(split[0]*pos_ds.shape[0]))
    X_train = np.vstack([X_train, pos_ds[:split_idx_pos, :]])
    y_train = np.vstack([y_train, np.expand_dims(np.array([1]*pos_ds[:split_idx_pos, :].shape[0]), axis=1)])
    X_test = np.vstack([X_test, pos_ds[split_idx_pos+1:, :]])
    y_test = np.vstack([y_test, np.expand_dims(np.array([1]*pos_ds[split_idx_pos+1:, :].shape[0]), axis=1)])

    full_train = np.hstack([X_train, y_train])
    np.random.shuffle(full_train)
    X_train = full_train[:, :-1]
    y_train = full_train[:, -1]
    full_test = np.hstack([X_test, y_test])
    np.random.shuffle(full_test)
    X_test = full_test[:, :-1]
    y_test = full_test[:, -1]

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    return X_train, X_test, y_train, y_test
