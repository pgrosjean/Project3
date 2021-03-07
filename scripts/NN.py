import numpy as np
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
    def __init__(self, setup=[[68,25,"sigmoid",0],[25,1,"sigmoid",0]],lr=.05,seed=1,error_rate=0,bias=1,iter=500,lambda=.00001,simple=0):
        #Note - these paramaters are examples, not the required init function parameters
        self.arch = nn_architechture
        self.lr = lr
        self._seed = seed
        self._depth = len(nn_architechture)
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
            param_dict['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1)
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1)
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
            W_curr = param_dict['W' + layer_idx]
            b_curr = param_dict['b' + layer_idx]
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

    def backprop(self, y_hat, y, cache):
        '''
        This method is responsible for the entire backprop for the whole
        neural network.

        Args:
            y_hat (array-like): Predicted output values.
            y (array-like): Ground truth labels.
            cache (dict): Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict (dict): Dictionary containing the graident information
                from this round of backprop.

        '''
        param_dict = self.param_dict
        # setting up dict for saving partial derivative values
        grad_dict = {}
        # expanding the dimension on the labels to match y_hat dims
        y = np.expand_dims(y, dim=0) # dims: [1, batch]
        # getting partial derivate of loss with respect to final activation
        dA_prev = self._binary_crossentropy_backprop(y, y_hat)
        # iterating through NN backwards to calculate gradients
        for idx_prev, layer in reverse(list(enumerate(self.arch))):
            idx_curr = idx_prev + 1
            activation_curr = layer['activation']
            dA_curr = dA_prev
            A_prev = cache['A' + str(idx_prev)]
            Z_curr = cache['Z' + str(idx_curr)]
            W_curr = param_dict['W' + str(idx_curr)]
            b_curr = param_duct['b' + str(idx_curr)]
            # single layer backprop
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr)
            # saving gradients
            grad_dict['dW' + str(idx_curr)] = dW_curr
            grad_dict['db' + str(idx_curr)] = db_curr
        return grad_dict

    def _single_backprop(self, W_curr, b_curr, Z_curr, A_prev, dA_curr):
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

        Returns:
            dA_prev (array-like): Partial derivative of loss function
                with respect to previous layer activation matrix.
            dW_curr (array-like): Partial derivative of loss function
                with respect to current layer weight matrix.
            db_curr (array-like): Partial derivative of loss function
                with respect to current layer bias matrix.

        '''
        mb_size = A_prev.shape[1] # mini-batch size
        if activation == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        if activation == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / mb_size
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / mb_size
        dA_prev = np.dot(W_curr.T, dZ_curr)
        return dA_prev, dW_curr, db_curr

    def update_params(self, grad_dict):
        '''

        '''
        lr = self.lr
        

    def fit(self, X_train, y_train):
        pass

    def predict(self, X):
        pass

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
        loss = -1 / m * (np.dot(Y, np.log(y_hat).T) + np.dot(1-y, np.log(1-y_hat).T)) # dims: [1, 1]
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
        assert self.loss_function in ['binary_crossentropy'], 'Unsupported loss function'
        if self.loss_function == 'binary_crossentropy':
            loss = self._binary_crossentropy(y, y_hat)
        return loss



def activation(x):
    pass
