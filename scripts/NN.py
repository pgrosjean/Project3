## Tuesday
# implement the L2 loss function
# train the autoencoder
# write function for loading in sequence data
# describe and implement training regimine

## Wednesday
# debug/train the neural network for classifying the Rap1 binding
# implement k-fold cross-validation
# implement genetic algorithm or something for hyperparmater tuning

## Thursday
# evaluate the model on the final test set
# write all unit tests
# get tests working
# sphinx autodoc
# write up

## Friday
# finish write-up
# turn in assignment


import numpy as np
from tqdm import tqdm

class NeuralNetwork:
    '''
    This is a nerual network class that generates a nerual network
    and allows for its training and use for prediction base on user
    defined input parameters.

    Paramters:
        nn_architechture (list of dicts): This list of dictionaries describes
            the fully connected layers of the artificial neural network.
        lr (float): Learning Rate (alpha).
        seed (int): Random seed for assuring reproducibility.
        lambda (float): L2 regularization paramter.
        batch_size (int): Size of mini-batchs used for training.
        epochs (int): Number of epochs during training.
        loss_function (str): Name of loss function can be one of multiple.

    Attributes:
        arch (list of dicts): This list of dictionaries describes
            the fully connected layers of the artificial neural network.
        param_dict (dict): Dictionary of parameters in neural network.
        lr (float): Learning rate.

    '''
    def __init__(self, nn_architechture, lr=0.05, seed=1, epochs=100, loss_function='binary_crossentropy'):
        #Note - these paramaters are examples, not the required init function parameters
        self.arch = nn_architechture
        self.lr = lr
        self._seed = seed
        self.epochs = epochs
        self._depth = len(nn_architechture)
        self.loss_fnc = loss_function
        self.param_dict = self._init_params()

    def _init_params(self):
        '''
        This method generates generates the parameter matrices for all layers of
        the neural network. This function does not return anything, but instead
        saves an attribute

        Args:
            None

        Returns:
            param_dict (dict): Dictionary of parameters in neural network.

        '''
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def forward(self, X):
        '''
        This method is responsible for one forward pass of the
        entire neural network.

        Args:
            X (array-like): Input matrix with size [batch_size, features].

        Returns:
            output (array-like): Output of forward pass.
            cache (dictionary of arry-like): Memory store of Z and A matrices
                for use in backprop.

        '''
        param_dict = self.param_dict
        # defining dictionary for use in backprop
        cache = {}
        # Setting previous activation matrix as input
        A_curr = X
        # iterating through layers
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            A_prev = A_curr
            activation_curr = layer['activation']
            W_curr = param_dict['W' + str(layer_idx)]
            b_curr = param_dict['b' + str(layer_idx)]
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation_curr)
            # saving previous A and current Z for backprop
            cache['A' + str(idx)] = A_prev
            cache['Z' + str(layer_idx)] = Z_curr
        output = A_curr
        return output, cache

    def _single_forward(self, W_curr, b_curr, A_prev, activation):
        '''
        This method is used for a single feedfoward pass on
        a single layer.

        Args:
            W_curr (array-like): Current layer weight matrix.
            b_curr (array-like): Current layer bias matrix.
            A_prev (array-like): Previous layer activation matrix.
            activation (str): Name of activation function for current layer.

        Returns:
            A_curr (array-like): Current layer activation matrix.
            Z_curr (array-like): Current layer linear transformed matrix.

        '''
        # Current Linear Transformation and Bias addition
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        # Activation Functions
        activation = activation.lower()
        assert activation in ['relu', 'sigmoid', 'linear'], 'unsupported activation function'
        if activation == 'relu':
            A_curr = self._relu(Z_curr)
        elif activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        elif activation == 'linear':
            A_curr = Z_curr
        return A_curr, Z_curr

    def backprop(self, y, y_hat, cache):
        '''
        This method is responsible for the entire backprop for the whole
        neural network.

        Args:
            y (array-like): Ground truth labels.
            y_hat (array-like): Predicted output values.
            cache (dict): Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict (dict): Dictionary containing the graident information
                from this round of backprop.

        '''
        param_dict = self.param_dict
        # setting up dict for saving partial derivative values
        grad_dict = {}
        # getting partial derivate of loss with respect to final activation
        dA_prev = self._binary_crossentropy_backprop(y, y_hat)
        # iterating through NN backwards to calculate gradients
        for idx_prev, layer in list(enumerate(self.arch))[::-1]:
            idx_curr = idx_prev + 1
            activation_curr = layer['activation']
            dA_curr = dA_prev
            A_prev = cache['A' + str(idx_prev)]
            Z_curr = cache['Z' + str(idx_curr)]
            W_curr = param_dict['W' + str(idx_curr)]
            b_curr = param_dict['b' + str(idx_curr)]
            # single layer backprop
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)
            # saving gradients
            grad_dict['dW' + str(idx_curr)] = dW_curr
            grad_dict['db' + str(idx_curr)] = db_curr
        return grad_dict

    def _single_backprop(self, W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr):
        '''
        This method is used for a single backprop pass on
        a single layer.

        Args:
            W_curr (array-like): Current layer weight matirx.
            b_curr (array-like): Current layer bias matrix.
            Z_curr (array-like): Current layer linear transform matrix.
            A_prev (array-like): Previous layer activation matrix.
            dA_curr (array-like): Partial derivative of loss function
                with respect to current layer activation matrix.
            activation_curr (str): Name of activation function of layer.

        Returns:
            dA_prev (array-like): Partial derivative of loss function
                with respect to previous layer activation matrix.
            dW_curr (array-like): Partial derivative of loss function
                with respect to current layer weight matrix.
            db_curr (array-like): Partial derivative of loss function
                with respect to current layer bias matrix.

        '''
        mb_size = A_prev.shape[1] # mini-batch size
        if activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr == 'linear':
            dZ_curr = dA_curr
        dW_curr = np.dot(dZ_curr, A_prev.T) / mb_size
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / mb_size
        dA_prev = np.dot(W_curr.T, dZ_curr)
        return dA_prev, dW_curr, db_curr

    def _update_params(self, grad_dict):
        '''
        This function updates the parameters in the neural network after
        backprop.

        Args:
            grad_dict (dict): Dictionary containing the graident information
                from most recent round of backprop.
        Returns:
            None
        '''
        lr = self.lr
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            # updating weights
            self.param_dict['W' + str(layer_idx)] -= lr * grad_dict['dW' + str(layer_idx)]
            # updating biases
            self.param_dict['b' + str(layer_idx)] -= lr * grad_dict['db' + str(layer_idx)]
        return None

    def fit_batch_wise(self, data_loader):
        '''
        This function trains the model.

        Args:
            data_loader
        '''
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        for _ in range(self.epochs):
            curr_loss_train = []
            curr_loss_val = []
            for batch in data_loader.train_queue:
                X, y = batch
                y_hat, cache = self.forward(X)
                curr_loss_train.append(self._loss_function(y, y_hat))
                grad_dict = self.backprop(y, y_hat, cache)
                self._update_params(grad_dict)
            per_epoch_loss_train.append(np.mean(np.array(curr_loss_train)))
            for batch in data_loader.val_queue:
                X, y = batch
                y_hat, _ = self.forward(X)
                curr_loss_val.append(self._loss_function(y, y_hat))
            per_epoch_loss_val.append(np.mean(np.array(curr_loss_val)))

        per_epoch_loss = {'train': per_epoch_loss_train, 'val': per_epoch_loss_val}
        return per_epoch_loss

    def fit(self, X_train, y_train, X_val, y_val):
        '''
        This function trains the nerual network via training for
        the number of epochs defined at the initialization of this
        class instance.

        Args:
            X_train (array-like): Input features of training set.
            y_train (array-like): Labels for training set.
            X_val (array-like): Input features of validation set.
            y_val (array-like): Labels for validation set.

        Returns:
            per_epoch_loss_train (list): List of per epoch loss for training set.
            per_epoch_loss_val (list): List of per epoch loss for validation set.
        '''
        # Swaping from batch first to features firs for NN implementaton
        X_train = np.swapaxes(X_train, 1, 0) # [features, batch]
        y_train = np.swapaxes(y_train, 1, 0) # [1, batch]
        X_val = np.swapaxes(X_val, 1, 0) # [features, batch]
        y_val = np.swapaxes(y_val, 1, 0) # [1, batch]
        # Training Loop
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        for _ in range(self.epochs):
            # training pass
            y_hat, cache = self.forward(X_train)
            per_epoch_loss_train.append(self._loss_function(y_train, y_hat))
            grad_dict = self.backprop(y_train, y_hat, cache)
            self._update_params(grad_dict)
            # validation pass
            y_hat, _ = self.forward(X_val)
            per_epoch_loss_val.append(self._loss_function(y_val, y_hat))
        return [per_epoch_loss_train, per_epoch_loss_val]

    def predict(self, X):
        '''

        '''
        y_hat, _ = self.forward(X)
        return y_hat

    def _sigmoid(self, Z):
        '''
        Sigmoid activation function.

        Args:
            Z (array-like): Output of layer linear tranform.

        Returns:
            nl_transform (array-like): Activation function output.
        '''
        nl_transform = 1/(1+np.exp(-Z))
        return nl_transform

    def _relu(self, Z):
        '''
        ReLU activation function.

        Args:
            Z (array-like): Output of layer linear tranform.

        Returns:
            nl_transform (array-like): Activation function output.
        '''
        nl_transform = np.maximum(0, Z)
        return nl_transform

    def _sigmoid_backprop(self, dA, Z):
        '''
        Sigmoid derivative for backprop.

        Args:
            dA (array-like): Partial derivative of previous layer activation
                matrix.
            Z (array-like): Output of layer linear tranform.

        Returns:
            dZ (array-like): Partial derivative of current layer Z matrix.
        '''
        sigmoid = self._sigmoid(Z)
        dZ = dA*sigmoid*(1-sigmoid)
        return dZ

    def _relu_backprop(self, dA, Z):
        '''
        ReLU derivative for backprop.

        Args:
            dA (array-like): Partial derivative of previous layer activation
                matrix.
            Z (array-like): Output of layer linear tranform.

        Returns:
            dZ (array-like): Partial derivative of current layer Z matrix.
        '''
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def _mean_squared_error(self, y, y_hat):
        '''

        '''
        pass

    def _mean_squared_error_backprop(self, y, y_hat):
        '''

        '''
        pass

    def _binary_crossentropy(self, y, y_hat):
        '''
        Binary crossentropy loss function.

        Args:
            y_hat (array-like): Predicted output.
            y (array-like): Ground truth output.

        Returns:
            loss (array-like): Average loss of mini-batch.

        '''
        # assuming y and y_hat have dimensions [1, batch_size]
        m = y_hat.shape[1]
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1-eps)
        batch_loss = -1 / m * (np.dot(y, np.log(y_hat).T) + np.dot(1-y, np.log(1-y_hat).T)) # dims: [1, 1]
        loss = np.squeeze(batch_loss) # dims: [1]
        return loss

    def _binary_crossentropy_backprop(self, y, y_hat):
        '''
        Binary crossentropy loss function derivative.

        Args:
            y_hat (array-like): Predicted output.
            y (array-like): Ground truth output.

        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.

        '''
        # assuming y and y_hat have dimensions [1, batch_size]
        # calculating partial derivative of loss with respect to A matrix
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1-eps)
        dA = -(np.divide(y, y_hat) - np.divide(1-y, 1-y_hat))
        return dA

    def _loss_function(self, y, y_hat):
        '''
        Loss function, computes loss given y_hat and y. This function is
        here for the case that someone where to want to write more loss
        functions than just binary crossentropy.

        Args:
            y_hat (array-like): Predicted output.
            y (array-like): Ground truth output.

        Returns:
            loss (array-like): Average loss of mini-batch.

        '''
        assert y.shape == y_hat.shape, 'Predicted value dimensions need to match labels'
        assert self.loss_fnc in ['binary_crossentropy', 'mean_squared_error'], 'Unsupported loss function'
        if self.loss_fnc == 'binary_crossentropy':
            # if y has dimesnions greater than 1 e.g.
            ## PROB CHANGE THIS
            if y.shape[0] > 1:
                losses = []
                for y_temp, y_hat_temp in zip(y.T, y_hat.T):
                    y_temp = np.expand_dims(y_temp.T, axis=0)
                    y_hat_temp = np.expand_dims(y_hat_temp.T, axis=0)
                    losses.append(self._binary_crossentropy(y_temp, y_hat_temp))
                loss = np.sum(np.array(losses))
            # if y has 1 dimension e.g. a class label
            elif y.shape[0] == 1:
                loss = self._binary_crossentropy(y, y_hat)
            else:
                print('LOSS FUNCTION ERROR: y must have at least dimensionality of 1.')
        return loss


class DataLoader:
    '''
    This class is used to load data while training a model. The DataLoader
    class generates a queue for training, validation, and testing.

    Parameters:
        data (Pandas DataFrame): Dataset including features and
        X_cols (list of strings): Column names corresponding to features
        y_col (string): Column name corresponding to labels
        split (list of floats): Floats describing the splitting ratio for the
            training, validation, and testing sets.
        seed (int): Seed for random processes involved in sampling the data.
        batch_size (int): Mini-batch size.

    Attributes:
        train_queue (list of array-like):
        val_queue (list of array-like):
        test_queue ()

    '''
    def __init__(self, data, X_cols, y_col, split=[0.8, 0.1, 0.1], seed=14, batch_size=16):
        self.df = data
        self.X_cols = X_cols
        self.y_col = y_col
        self._split = split

    def _split_data(self):
        df = self.df
        df = df.shuffle(frac=1)
        df = df.reset_index(drop=True)
        for grp_idx, grp in df.groupby(y_col):
            pass
