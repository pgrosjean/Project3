from scripts import NN
from scripts import optim
from scripts import io
from scripts import preprocess
import numpy as np


# Testing Sequence Encoding
def test_encoder():
    '''
    This function tests the correct encoding of sequences using my
    flattened one hot encoding method.
    '''
    positive_seqs = np.array(['AA', 'AT', 'GC', 'AG'])
    encoded_seq = preprocess.encode_seqs(positive_seqs)
    ### testing that encoding is correct
    assert np.equal(encoded_seq[0], np.array([1,0,0,0,1,0,0,0]).astype('float')).sum() == 8
    assert np.equal(encoded_seq[1], np.array([1,0,0,0,0,1,0,0]).astype('float')).sum() == 8
    assert np.equal(encoded_seq[2], np.array([0,0,0,1,0,0,1,0]).astype('float')).sum() == 8
    assert np.equal(encoded_seq[3], np.array([1,0,0,0,0,0,0,1]).astype('float')).sum() == 8
    ### testing that encodings are returned in correct array shape
    assert encoded_seq.shape[0] == positive_seqs.shape[0]
    assert encoded_seq.shape[1] == 8

def test_io():
    '''
    This function tests reading in both a text file and a fasta file.
    '''
    ### testing reading in text file
    text_io = io.read_text_file('./data/rap1-lieb-positives.txt')
    assert text_io[0] == 'ACATCCGTGCACCTCCG'
    ### testing reading in fasta file
    fasta_io = io.read_fasta_file('./data/yeast-upstream-1k-negative.fa')
    assert fasta_io[0] == 'CTTCATGTCAGCCTGCACTTCTGGGTCGTTGAAGTTTCTACCGATCAAACGCTTAGCGTCGAAAACGGTATTCGAAGGATTCATAGCAGCTTGATTCTTAGCAGCATCACCAATCAATCTTTCAGTGTCAGTGAAAGCGACAAAAGATGGAGTGGTTCTGTTACCTTGATCGTTGGCAATAATGTCCACACGATCATTAGCAAAGTGAGCAACACACGAGTATGTTGTACCTAAATCAATACCGACAGCTTTTGACATATTATCTGTTATTTACTTGAATTTTTGTTTCTTGTAATACTTGATTACTTTTCTTTTGATGTGCTTATCTTACAAATAGAGAAAATAAAACAACTTAAGTAAGAATTGGGAAACGAAACTACAACTCAATCCCTTCTCGAAGATACATCAATCCACCCCTTATATAACCTTGAAGTCCTCGAAACGATCAGCTAATCTAAATGGCCCCCCTTCTTTTTGGGTTCTTTCTCTCCCTTTTGCCGCCGATGGAACGTTCTGGAAAAAGAAGAATAATTTAATTACTTTCTCAACTAAAATCTGGAGAAAAAACGCAAATGACAGCTTCTAAACGTTCCGTGTGCTTTCTTTCTAGAATGTTCTGGAAAGTTTACAACAATCCACAAGAACGAAAATGCCGTTGACAATGATGAAACCATCATCCACACACCGCGCACACGTGCTTTATTTCTTTTTCTGAATTTTTTTTTTCCGCCATTTTCAACCAAGGAAATTTTTTTTCTTAGGGCTCAGAACCTGCAGGTGAAGAAGCGCTTTAGAAATCAAAGCACAACGTAACAATTTGTCGACAACCGAGCCTTTGAAGAAAAAATTTTTCACATTGTCGCCTCTAAATAAATAGTTTAAGGTTATCTACCCACTATATTTAGTTGGTTCTTTTTTTTTTCCTTCTACTCTTTATCTTTTTACCTCATGCTTTCTACCTTTCAGCACTGAAGAGTCCAACCGAATATATACACACATA'


# Testing Auto Encoder
def test_autoencoder():
    '''
    This funciton tests the AutoEncoder by generating and training the
    autoencoder and ensuring that the output dimensions are correct and
    that the loss is always decreasing. This also tests that the NN
    architecture is correct and weight matrices are the right size.
    '''
    # Defining neccessary hyperparameters for the autoencoder
    nn_architecture = [{'input_dim': 8, 'output_dim': 3, 'activation': 'sigmoid'},
                       {'input_dim': 3, 'output_dim': 8, 'activation': 'sigmoid'}]
    loss_function = 'mse'
    learning_rate = 10
    seed = 20
    epochs = 10000
    # Generating autoencoder NN class instance
    identity_ae = NN.NeuralNetwork(nn_architecture, lr=learning_rate, seed=seed, epochs=epochs, loss_function=loss_function)
    # Defining data for use in the autoencoder
    X = np.eye(8,8)
    y = X
    # Training the Auto Encoder
    per_epoch_loss_train, per_epoch_loss_val, _, _ = identity_ae.fit(X, y, X, y)
    recon = identity_ae.predict(X[:2, :])
    ### ensuring that training loss always decreases
    assert np.sum(np.diff(per_epoch_loss_train) > 0) == 0
    ### ensuring that the output predicted reconstruction has the right shape
    assert recon.shape == (2, 8)
    ### testing nn architecture is correct
    assert identity_ae.arch == nn_architecture
    ### testing weight matrices are the correct shape
    assert identity_ae.param_dict['W1'].shape == (3, 8)
    assert identity_ae.param_dict['W2'].shape == (8, 3)
    ### testing bias matrices are the correct shape
    assert identity_ae.param_dict['b1'].shape == (3, 1)
    assert identity_ae.param_dict['b2'].shape == (8, 1)


# Testing the Binary Classifier Rap1 Binding
def test_binary_classifier():
    '''
    This function tests the Rap1 Binary Classifier by ensuring that
    '''
    # Reading in postive sequences
    positive_seqs = io.read_text_file('./data/rap1-lieb-positives.txt')
    # Reading in upstream yeast sequences
    pot_neg_seqs = io.read_fasta_file('./data/yeast-upstream-1k-negative.fa')
    # Randomly sampling negative sequences
    negative_seqs = preprocess.sample_negative_examples(pot_neg_seqs, positive_seqs, num_samples=1000, seq_length=17)
    # Generating labels
    pos_labels = np.expand_dims(np.array([1]*len(positive_seqs)), axis=0)
    neg_labels = np.expand_dims(np.array([0]*len(negative_seqs)), axis=0)
    labels = np.swapaxes(np.hstack([pos_labels, neg_labels]), 0, 1)
    # Generating Unraveled One Hot Encoded Input Feauture --> 4*seq_length
    pos_input = preprocess.encode_seqs(positive_seqs)
    neg_input = preprocess.encode_seqs(negative_seqs)
    # Generating full inputs to network
    inputs = np.vstack([pos_input, neg_input])
    # Defining neccessary hyperparameters for the NN model
    nn_architecture = [{'input_dim': 68, 'output_dim': 2, 'activation': 'relu'},
                       {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    loss_function = 'binary_crossentropy'
    learning_rate = 2
    seed = 14
    epochs = 1000
    early_stop = [20, -1e-12]
    # Generating NN class instance for binary classifier
    bc_nn = NN.NeuralNetwork(nn_architecture, lr=learning_rate, seed=seed, epochs=epochs, loss_function=loss_function)
    # saving initial weights to make sure they are changing
    init_w1 = bc_nn.param_dict['W1'].copy()
    init_w2 = bc_nn.param_dict['W2'].copy()
    # Doing sinlge fold training split
    X_train, X_val, y_train, y_val = preprocess.split_basic_binary(inputs, labels, split=[0.8, 0.2], shuffle=True)
    # Training Model
    per_epoch_loss_train, per_epoch_loss_val, per_epoch_acc_train, per_epoch_acc_val = bc_nn.fit(X_train, y_train, X_val, y_val, early_stop=early_stop)
    ### ensuring that training loss always decreases
    assert np.sum(np.diff(per_epoch_loss_train) > 0) == 0
    ### ensuring that weight matrices change while training
    final_w1 = bc_nn.param_dict['W1']
    final_w2 = bc_nn.param_dict['W2']
    assert np.sum(init_w1 - final_w1) != 0
    assert np.sum(init_w2 - final_w2) != 0


# Testing Splits
def test_splits():
    '''
    Testing the splits to make sure there are not sharing between training and test sets.
    This makes sure that no data leakage is occuring.
    '''
    # Reading in postive sequences
    positive_seqs = io.read_text_file('./data/rap1-lieb-positives.txt')
    # Reading in upstream yeast sequences
    pot_neg_seqs = io.read_fasta_file('./data/yeast-upstream-1k-negative.fa')
    # Randomly sampling negative sequences
    negative_seqs = preprocess.sample_negative_examples(pot_neg_seqs, positive_seqs, num_samples=1000, seq_length=17)
    # Generating labels
    pos_labels = np.expand_dims(np.array([1]*len(positive_seqs)), axis=0)
    neg_labels = np.expand_dims(np.array([0]*len(negative_seqs)), axis=0)
    labels = np.swapaxes(np.hstack([pos_labels, neg_labels]), 0, 1)
    # Generating Unraveled One Hot Encoded Input Feauture --> 4*seq_length
    pos_input = preprocess.encode_seqs(positive_seqs)
    neg_input = preprocess.encode_seqs(negative_seqs)
    # Generating full inputs to network
    inputs = np.vstack([pos_input, neg_input])
    # Doing sinlge fold training split
    X_train, X_val, y_train, y_val = preprocess.split_basic_binary(inputs, labels, split=[0.8, 0.2], shuffle=True)
    # Making sure that there is no overlap between training and validation set
    X_train_mod = [','.join(X.astype(int).astype(str)) for X in X_train]
    train_set = set(X_train_mod)
    X_val_mod = [','.join(X.astype(int).astype(str)) for X in X_val]
    val_set = set(X_val_mod)
    assert len(train_set & val_set) == 0
