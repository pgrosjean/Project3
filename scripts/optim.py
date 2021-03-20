from __future__ import division
import random
import math
import numpy as np


### CLASS DEFINITONS ###

class Particle:
    '''
    This class is for use in the ParticleSwarm class and is an individual
    particle in a swarm used for optimization.

    Parameters:
        x0 (list of float): This is the inital position of the particle.

    Attributes:
        p_i (list): List of positions.
        v_i (list): List of velocities.
        p_best_i (list): List of best position.
        err_best_i (float): Best error value.
        err_i (list): List of errors over iterations.
        num_dims (int): Number of dimension in optimization problem.
    '''
    def __init__(self, x0):
        self.p_i = []   # position
        self.v_i = []   # velocity
        self.p_best_i = []   # best position individual
        self.err_best_i = -1   # best error individual
        self.err_i =- 1  # error individual
        self.num_dims = len(x0) # number of dimensions
        # Generating the number of dimesnions
        for i in range(0, self.num_dims):
            self.v_i.append(random.uniform(-1, 1))
            self.p_i.append(x0[i])

    # evaluate current loss function of the particle
    def evaluate_particle(self, loss_func):
        '''
        This function evaluates the loss function for this particle instance
        amd updates if the particle position is the best opserved so far and
        what the associated error is.

        Args:
            loss_func (Python function): Loss or cost function for evalutating
                the particles fitness.
        '''
        self.err_i=loss_func(self.p_i)
        # check tcurrent position to see if it is best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.p_best_i=self.p_i
            self.err_best_i=self.err_i

    # update particle velocity
    def velocity_update(self, p_best_g):
        '''
        This function updates the velocity of the particle based on the intertia
        cognative constant and the social constant.

        Args:
            p_best_g (list of float): Best position for the entire particle
                swarm group.
        '''
        wc=8 # inertia
        cc=12 # cognative constant
        sc=15 # social constant
        for i in range(0, self.num_dims):
            r1=random.random()
            r2=random.random()
            v_cognitive=cc*r1*(self.p_best_i[i]-self.p_i[i])
            v_social=sc*r2*(p_best_g[i]-self.p_i[i])
            self.v_i[i]=wc*self.v_i[i]+v_cognitive+v_social

    def position_update(self, bounds):
        '''
        This function updates the position of particle for a given itertion of
        the optimization procedure.

        Args:
            bounds (list of tuples): This is a list of the defined bounds for
                each feature in the optimization problem.
        '''
        for i in range(0,self.num_dims):
            self.p_i[i]=self.p_i[i]+self.v_i[i]
            # adjust maximum position if necessary
            if self.p_i[i]>bounds[i][1]:
                self.p_i[i]=bounds[i][1]
            # adjust minimum position if neseccary
            if self.p_i[i] < bounds[i][0]:
                self.p_i[i]=bounds[i][0]

class ParticleSwarm():
    def __init__(self, loss_func, x0, bounds, num_particles, max_iter):
        '''
        This class is used to conduct a particle swarm optimization. This class
        was written for hyperparamter tuning.

        Parameters:
            loss_func (Python Function): Function to minimize in optimization
                task.
            x0 (list): Initial values of each dimension for optimization. Each
                particle is initialized as a perturbation from this inital x0
                value and allowed to search the optimization landscape.
            bounds (list of tuples): Bounds for each hyperparameter or dimension
                in the optimization problem.
            num_particles (int): The number of particles in the swarm.
            max_iter (int): Number of iterations when optimization stops.

        Attributes:
            swarm (list of Particle class instances): List of particles instances
                in the swarm used for optimization.
        '''
        self.num_dims=len(x0)
        swarm=[]
        for i in range(0, num_particles):
            # randomly choosing the starting points of the particles
            x0_temp = x0.copy()
            for ind in range(len(x0)):
                pert = np.random.randint(0, (bounds[ind][1] - bounds[ind][0])*0.1)
                rand_val = np.random.normal(0, 1)
                if rand_val >= 0:
                    x0_temp[ind] = np.clip(x0_temp[ind] + pert, bounds[ind][0], bounds[ind][1])
                else:
                    x0_temp[ind] = np.clip(x0_temp[ind] - pert, bounds[ind][0], bounds[ind][1])
            swarm.append(Particle(x0_temp))
        self.swarm = swarm
        self._num_particles = num_particles
        self._max_iter = max_iter
        self._bounds = bounds
        self._loss_func = loss_func

    def optimize(self):
        '''
        This method runs the optimization procedure defined in the initialization
        of the class instance. This function takes no arguments, as it is just
        running based on the paramters given in the initalization of the class
        instance.

        Returns:
            p_best_g (list of floats): Best particle position or the best values
                of the hyperparamters being tuned.
            err_best_g (float): Loss function value at the best particle position.
        '''
        err_best_g=-1 # best error for whole group
        p_best_g=[] # best position for whole group
        # initialize swarm of particles
        loss_func = self._loss_func
        max_iter = self._max_iter
        swarm = self.swarm
        num_particles = self._num_particles
        bounds = self._bounds
        # Optimization process
        i = 0
        while i < max_iter:
            # iterating through particles for loss function calc
            for j in range(0, num_particles):
                swarm[j].evaluate_particle(loss_func)
                # determine if current particle is the best
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    p_best_g = list(swarm[j].p_i)
                    err_best_g = float(swarm[j].err_i)
            # iterate through swarm and update parameters
            for j in range(0, num_particles):
                swarm[j].velocity_update(p_best_g)
                swarm[j].position_update(bounds)
            i+=1
        return [p_best_g, err_best_g]
