import numpy as np
import pandas as pd

def read_text_file(filename):
    '''
    This function reads in a text file into a numpy array of str dtype.

    Args:
        filename (str): File path and name of file, filename should end in .txt.

    Returns:
        arr (array-like): Numpy array of sequences.
    '''
    with open(filename, 'r') as f:
        arr = np.array([l.strip() for l in f.readlines()])
    return arr

def read_fasta_file(filename):
    '''
    This function reads in a fasta file into a numpy array of sequence strings.

    Args:
        filename (str): File path and name of file, filename should end
            in .fa or .fasta.

    Returns:
        seqs (array-like): Numpy array of sequences.
    '''
    with open(filename, 'r') as f:
        seqs = []
        seq = ''
        for line in f:
            if line.startswith('>'):
                seqs.append(seq)
                seq = ''
            else:
                seq += line.strip()
        seqs = np.array(seqs)[1:]
        return seqs
