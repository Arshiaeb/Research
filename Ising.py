import numpy as np
import pandas as pd
import os
from skimage.measure import block_reduce
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from torch.nn.functional import pad
from morans import morans
np.seterr(all="raise")


# Create a square 2D Ising Lattice 
# Initial magnetization state of the elements in the lattice is uniform (u) or randomized (r)

class IsingLattice:

    def __init__(self, initial_state, size, J):
        self.size = size # size of each dimension
        
        if J is np.ndarray:
            self.J = J
        else:
            self.J = J*np.ones((size,size))

        self.system = self._build_system(initial_state)

    @property
    def sqr_size(self):
        return (self.size, self.size)

    def _build_system(self, initial_state):
        """Build the system

        Build either a randomly distributed system or a homogeneous system (for
        watching the deterioration of magnetization

        Parameters
        ----------
        initial_state : str: "r" for random or "u" for uniform (all +1)
            Initial state of the lattice. 
        """

        if initial_state == 'r':
            system = np.random.choice([-1, 1], self.sqr_size)
        elif initial_state == 'u':
            system = np.ones(self.sqr_size)*np.random.choice([-1, 1])
        else:
            raise ValueError(
                "Initial State must be 'r', random, or 'u', uniform"
            )

        return system
    
    def _bc(self, i):
        """Apply periodic boundary condition

        Check if a lattice site coordinate falls out of bounds. If it does,
        apply periodic boundary condition

        Assumes lattice is square

        Parameters
        ----------
        i : int
            lattice site coordinate

        Return
        ------
        int
            corrected lattice site coordinate
        """
        if i >= self.size:
            return 0
        if i < 0:
            return self.size - 1
        else:
            return i

    def energy(self, N, M):
        """Calculate the energy of spin interaction at a given lattice site
        i.e. the interaction of a Spin at lattice site n,m with its 4 neighbors

        - S_n,m*(S_n+1,m + Sn-1,m + S_n,m-1, + S_n,m+1)

        Parameters
        ----------
        N : int
            lattice site coordinate
        M : int
            lattice site coordinate

        Return
        ------
        float
            energy of the site
        """
        interactions = -self.system[N, M]*self.J[N,M]*(self.system[self._bc(N - 1), M] + self.system[self._bc(N + 1), M]+ self.system[N, self._bc(M - 1)] + self.system[N, self._bc(M + 1)])
        #external = -self.system[N, M]*self.h[N,M]
        #energy = interactions + external
        energy = interactions
        return energy
    
    @property
    def internal_energy(self):
        e = 0
        E = 0
        E_2 = 0

        for i in range(self.size):
            for j in range(self.size):
                e = self.energy(i, j)
                E += e
                E_2 += e**2

        U = (1./self.size**2)*E
        U_2 = (1./self.size**2)*E_2

        return U, U_2

    @property
    def heat_capacity(self,temp):
        U, U_2 = self.internal_energy
        return np.mean((U_2 - U**2)/np.power(temp,2))

    @property
    def magnetization(self):
        """Find the overall magnetization of the system
        """
        return np.sum(self.system)/self.size**2   #Maybe get rid of abs or add it
    
def hc(lattice,temp):
    U, U_2 = lattice.internal_energy
    return np.mean((U_2 - U**2)/np.power(temp,2))

def create_params():
    try:
        rng = np.random.default_rng()
    except AttributeError:
        rng = np.random.RandomState()


    J_mean = 2**(5*rng.uniform())  # why 5? can choose something else
    J_std = 0.3*rng.uniform()*J_mean # 0 - 30% of mean

   

    Tc = 2*J_mean/(np.log(1+np.sqrt(2))) # critical temperature

    null = rng.choice([0,1])
    #null = 0

    offset_dir =rng.choice([-1,1])
    spatial_coarse_graining = rng.choice(np.arange(3,8))
    temporal_coarse_graining = rng.choice(np.arange(3,8))
    epoch_len = 5000

    if null: # No Transition run (does not go through Tc)
        Tb1 = Tc * (0.2 + 0.2*rng.uniform())
        Tb2 = Tc * (0.4 + 0.2*rng.uniform())
            
        Tbounds = Tc + offset_dir*np.array([Tb1,Tb2])
        Tbounds = rng.permutation(Tbounds)

    else: # Transition Run (goes through Tc)
        Tb1 = Tc * (0.7 - 0.4*rng.uniform())
        Tb2 = Tc * (1.2 + 0.4*rng.uniform())
        #Tbounds = np.sort(np.array([Tb1,Tb2]))[::-1] # descending order, why ?? *******************
        Tbounds = rng.choice(([[Tb1,Tc],[Tc,Tb2]]))
        Tbounds = rng.permutation(Tbounds)
    
    target_duration = rng.choice(list(range(575,675)))


    run_params = {'J_mean':J_mean, 'J_std':J_std, 'Tc':Tc, 
                    'spatial_coarse_graining':spatial_coarse_graining,
                    'temporal_coarse_graining':temporal_coarse_graining,
                    'epoch_len':epoch_len, 'null':null,'Tbounds':Tbounds,'target_duration':target_duration}
    

    return run_params



def run(lattice, temps, epoch_len):
    """Run the simulation
    """

    epochs = temps.shape[0]

    System = np.zeros((epochs,lattice.system.shape[0],lattice.system.shape[1]))
    Magnetization = np.zeros(epochs)
    Heat_capacity = np.zeros(epochs)

    for epoch,temp in enumerate(temps):    

        step_avg = np.zeros((lattice.size,lattice.size)) # why are we taking step average

        for step in range(epoch_len):
            # Randomly select a site on the lattice
            N, M = np.random.randint(0, lattice.size, 2)

            # Calculate energy of a flipped spin
            dE = -2*lattice.energy(N, M)

            # "Roll the dice" to see if the spin is flipped
            if dE <= 0.:
                lattice.system[N, M] *= -1
            elif np.exp(-dE/(temp)) > np.random.rand():
                lattice.system[N, M] *= -1

            step_avg += lattice.system

        step_avg = step_avg/epoch_len

        # check and account for burn time (write an if statement)
        System[epoch,:,:] = step_avg
        #Magnetization[epoch] = lattice.magnetization
        #Heat_capacity[epoch] = hc(lattice,temp)

    return System



def ising_run(temps, size, J, epoch_len, initial_state):
    
    
    lattice = IsingLattice(initial_state=initial_state, size=size, J=J)
    out_vars = run(lattice, temps, epoch_len)

    return out_vars