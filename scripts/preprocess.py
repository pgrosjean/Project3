import numpy as np

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
            if neg_sample not in pos_seq_set:
                negative_samples.append(neg_sample)
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


def split_data(X, y, num_folds=1):
    '''
    '''
    pass
