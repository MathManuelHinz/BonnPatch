import numpy as np
from functools import reduce
import logging
from typing import Callable
from scipy import linalg
import  numpy.typing as npt
from typing import Callable, Tuple, Union, Optional
from scipy.special import poch
import math
from abc import ABC, abstractmethod 


from .hohd import *

matrix=npt.NDArray[np.float64]

logger = logging.getLogger(__name__)

L5TOPOLOGIES=[np.array([[0,0,0,0,1],[0,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[1,0,0,1,0]]),
            np.array([[0,0,0,1,1],[0,0,1,0,0],[0,1,0,1,0],[1,0,1,0,0],[1,0,0,0,0]]),
            np.array([[0,0,1,1,0],[0,0,1,0,0],[1,1,0,0,0],[1,0,0,0,1],[0,0,0,1,0]]),
            np.array([[0,1,0,0,1],[1,0,0,0,0],[0,0,0,1,0],[0,0,1,0,1],[1,0,0,1,0]]),
            np.array([[0,0,0,1,1],[0,0,0,0,1],[0,0,0,1,0],[1,0,1,0,0],[1,1,0,0,0]]),
            np.array([[0,0,1,1,0],[0,0,0,0,1],[1,0,0,0,0],[1,0,0,0,1],[0,1,0,1,0]]),
            np.array([[0,0,1,0,0],[0,0,0,0,1],[1,0,0,1,0],[0,0,1,0,1],[0,1,0,1,0]]),
            np.array([[0,1,0,1,0],[1,0,0,0,1],[0,0,0,1,0],[1,0,1,0,0],[0,1,0,0,0]]),
            np.array([[0,0,1,1,0],[0,0,0,1,1],[1,0,0,0,0],[1,1,0,0,0],[0,1,0,0,0]]),
            np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[0,0,0,1,0]]),
            np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,0,1],[0,0,0,0,1],[0,0,1,1,0]]),
            np.array([[0,1,0,0,0],[1,0,0,0,1],[0,0,0,1,1],[0,0,1,0,0],[0,1,1,0,0]]),
            np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[0,0,0,1,0]]),
            np.array([[0,1,0,0,0],[1,0,0,1,0],[0,0,0,1,1],[0,1,1,0,0],[0,0,1,0,0]]),
            np.array([[0,0,0,1,0],[0,0,1,1,0],[0,1,0,0,1],[1,1,0,0,0],[0,0,1,0,0]]),
            np.array([[0,1,0,1,0],[1,0,1,0,0],[0,1,0,0,1],[1,0,0,0,0],[0,0,1,0,0]]),
            np.array([[0,1,0,0,0],[1,0,0,1,0],[0,0,0,0,1],[0,1,0,0,1],[0,0,1,1,0]]),
            np.array([[0,0,0,1,0],[0,0,0,1,1],[0,0,0,0,1],[1,1,0,0,0],[0,1,1,0,0]])]

L5NAMES=["CCCCO","CCCOC","CCOCC","CCCOO","CCOCO","COCCO","OCCCO","CCOOC","COCOC",
       "OOOOC","OOOCO","OOCOO","OOOCC","OOCOC","OCOOC","COOOC","OOCCO","OCOCO"]

L5INTERESTING_MODELS={
    "TwoHillsFast": {"topology":L5TOPOLOGIES[5],"name":L5NAMES[5],"generator":10_000*np.array([[-0.80294985,0.0,0.40123272,0.40171713,0.0],[0.0,-0.01381559,0.0,0.0,0.01381559],[0.10448446,0.0,-0.10448446, 0.0, 0.0],[0.86417804, 0.0, 0.0, -0.95476492, 0.09058688],[0.0, 0.9441882, 0.0, 0.24489628, -1.18908448]])}
}

def lu_inverse(A):
    """Use LU decomposition to calculate the inverse of a psd matrix"""
    lu_decomp, piv = linalg.lu_factor(A)
    identity = np.eye(A.shape[0])
    A_inv = np.zeros_like(A)
    
    for i in range(A.shape[0]):
        A_inv[:,i] = linalg.lu_solve((lu_decomp, piv), identity[:,i])
    return A_inv

class Topology:
    """
    Topology defined by a adjacency matrix and the number of open states n_o. Optional arguments are 
    the topology name in the case of linear topologies and 
    the top_index in the case that the topology is one of the linear five state topologies.
    """
    def __init__(self,adjacency_matrix:matrix,n_o,name:str="",top_index:Optional[int]=None):
        self.update_topology(adjacency_matrix,n_o,name,top_index)

    def update_topology(self,adjacency_matrix:matrix,n_o:int,name:str,top_index:Optional[int]):
        self._topology=adjacency_matrix
        self.n:int=adjacency_matrix.shape[0]
        self._n_o=n_o
        self._n_c=self.n-self._n_o
        self._name=name 
        self.top_index=top_index

class TopologyFamily(ABC):
    """
        A class describing a set of topologies and a way to sample from them. 
        Can be used together with the AMP sampler to sample processes from the specified family.
    """

    @abstractmethod
    def sample_index(self)->int:
        pass 

    @abstractmethod
    def get_topology_by_index(self,index:int)->Topology:
        pass 

class LinearFiveStateToplogies(TopologyFamily):

    def __init__(self, rng:Optional[np.random.Generator]=None):
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng

    def sample_index(self)->int:
        return self._rng.integers(18)
    
    def get_topology_by_index(self,index:int)->Topology:
        top=L5TOPOLOGIES[index]
        top_name=L5NAMES[index]
        n_o=top_name.count("O")
        return Topology(top,n_o,top_name,top_index=index)

class HigherOrderHinkleyDetector:
    """
    Class to filter a time series using the higher order Hinkley detector. Upon initialization, this 
    class calculates the cutoff parameter lambda which is then used upon calling the filter method. 
    Furthermore the order can be set to an arbitrary positive integer, while 1 (normal Hinkley detector)
    and 8 (Higher order Hinkley detector) are the common choices. Due to multiple popular ways of choosing the
    cutoff parameter, the cutoff method can be specified, see the documentation of the calculate_cutoff method.
    """
    def __init__(self,order:int=8, cutoff_method:str="auto"):
        self.set_order(order)
        self.set_cutoff(self.calculate_cutoff(cutoff_method=cutoff_method))

    def set_order(self,order)->None:
        """Updates the order of the detector, default is 8, order 1 is equivalent to a Hinkley detector"""
        self.order=order

    def calculate_cutoff(self,cutoff_method="auto", t_res:int=3, snr:float=5.0,p:float=0.5)->float:
        """
        Calculates the cutoff value lambda used in the algorithm. Depending on the method other inputs (such as t_res,snr) are needed
        """
        if cutoff_method=="auto":
            if t_res>8: return self.calculate_cutoff(cutoff_method="simulate",t_res=t_res,snr=snr,p=p) 
            elif self.order>=t_res:
                return p*poch(self.order+1,t_res-1)/math.factorial(t_res-1)
            else: return p*poch(t_res,self.order)/math.factorial(self.order)
        elif cutoff_method=="simulate":
            gs=[np.arange(t_res+1)]
            for _ in range(1,self.order):
                cg=[0]
                for t in range(1,t_res+1):
                    cg.append(cg[-1]+gs[-1][t])
                gs.append(np.array(cg))
            return p*gs[-1][-1]
        elif cutoff_method=="schultze-draber":
            return 22*p/(snr**2)
        else: raise NotImplementedError

    def set_cutoff(self, cutoff=2)->None:
        """Setter for the cutoff value"""
        self.cutoff=cutoff

    def filter(self,z:matrix,mu_0=0.0,mu_1=1.0,)->matrix:
        """Applies the HOHD to the signal z with assumed levels mu_0 and mu_1. If not provided the values 0 and 1 are chosen."""
        assert mu_0<mu_1
        return np.array(higher_order_hinkley_detector(z,mu_0,mu_1,self.cutoff,self.order))
    

class AggregatedMarkovProcess:
    """
    Describes an aggregated Markov process by its rate matrix Q, topology, aggregation map f and stationary distribution pi.
    This class can be used to generate Paths using the gillepsie algorithm or to generate theoretical densities.
    """
    def __init__(self,topology:Topology,Q:matrix,f:Callable[[int],int],pi:matrix=None):
        self.update_process(topology,Q,pi,f)
    
    def get_stationary_distribution(self)->matrix:
        """
        Calculates and return the stationary distribution of the underlying process.
        """
        # Compute the stationary distribution by computing the left eigenvector of the intensity matrix with eigenvalue 0
        w, v = np.linalg.eig(
            self._Q.T
        )  # We are working with the transpose of the intensity matrix here because we want the left eigenvector
        idx = np.where(np.isclose(w, 0, rtol=1e-03, atol=1e-05))[0][0]
        stationary_distribution = np.abs(v[:, idx] / np.sum(v[:, idx]))
        # Check if the stationary distribution is valid
        if not np.allclose(np.sum(stationary_distribution), 1, rtol=1e-03, atol=1e-05):
            logger.warning("The stationary distribution does not sum to 1: %s", stationary_distribution)
        return stationary_distribution

    def update_process(self,topology:Topology,Q:matrix,pi:matrix,f:Callable[[int],int])->None:
        """
        Updates the topology, the generator, the initial distribution and f. Calculates the stationary distribution if no initial distribution is given.
        """
        self._topology=topology
        self._Q=Q
        if pi is None:
            self._pi=self.get_stationary_distribution()
        else: 
            self._pi=pi
        self._f=f
        self.theoretical_dists_ready=False 

    def get_theoretical_densities(self,mode:str="matrix")->Tuple[Callable[[np.float64],np.float64],Callable[[np.float64],np.float64],Callable[[np.float64],np.float64],Callable[[np.float64],np.float64]]:
        """
        Calculate theoretical densities of the one and two dimensional dwell time distributions. Admissible modes: matrixexpo, expo
        Returns the densities f_o,f_c,f_oc,f_co, since any higher dimensional dwell time distributions follow from these cases.
        """
        n_o=self._topology._n_o
        q_oo=self._Q[:n_o,:n_o]
        q_cc=self._Q[n_o:,n_o:]
        q_oc=self._Q[:n_o,n_o:]
        q_co=self._Q[n_o:,:n_o]

        w,v=linalg.eig((linalg.inv(q_oo)@q_oc@linalg.inv(q_cc)@q_co).T)
        idx = np.where(np.isclose(w, 1, rtol=1e-03, atol=1e-05))[0][0]
        pi_o = v[:, idx] / np.sum(v[:, idx])[np.newaxis]
        w2,v2=linalg.eig((linalg.inv(q_cc)@q_co@linalg.inv(q_oo)@q_oc).T)
        idx2 = np.where(np.isclose(w2, 1, rtol=1e-03, atol=1e-05))[0][0]
        pi_c=v2[:, idx2] / np.sum(v2[:, idx2])[np.newaxis]

        if mode=="matrixexpo":
            f_o=lambda t: (pi_o@linalg.expm(q_oo*t)@q_oc).sum()
            f_c=lambda s: (pi_c@linalg.expm(q_cc*s)@q_co).sum()
            f_oc=lambda t,s: (pi_o@linalg.expm(q_oo*t)@q_oc@linalg.expm(q_cc*s)@q_co).sum()
            f_co=lambda s,t: (pi_c@linalg.expm(q_cc*s)@q_co@linalg.expm(q_oo*t)@q_oc).sum()

        elif mode=="expo":
            lambdas, x, y = linalg.eig(q_oo, left=True)
            gamma=pi_o@x
            delta=(y.T@q_oc).sum(axis=1)
            alpha=gamma*delta

            omega, z, w = linalg.eig(q_cc, left=True)
            rho=pi_c@z
            mu=(w.T@q_co).sum(axis=1)
            beta=rho*mu

            xi=y.T@q_oc@z
            eta=w.T@q_co@x 

            alpha2d=gamma[:,np.newaxis]*xi*mu[np.newaxis,:]
            beta2d=rho[:,np.newaxis]*eta*delta[np.newaxis,:]
            f_o=lambda t: (alpha*np.exp(lambdas*t)).sum().real
            f_c=lambda t: (beta*np.exp(omega*t)).sum().real
            f_oc=lambda t,s:(alpha2d*np.exp(np.add.outer(t*lambdas,s*omega))).sum().real
            f_co=lambda s,t:(beta2d*np.exp(np.add.outer(s*omega,t*lambdas))).sum().real
        else:
            logger.warning(f"Mode {mode} is not implemented")
            raise NotImplementedError
        self.theoretical_dists_ready=True 
        self.f_o=f_o 
        self.f_c=f_c 
        self.f_oc=f_oc 
        self.f_co=f_co
        return f_o, f_c, f_oc, f_co 
    
    def get_generator_initial(self)->Tuple[matrix,matrix]:
        """
        Returns the generator of the underlying processes as well as the stationary distribution, which is assumed to be the initial distribution.
        """
        return self._Q, self._pi

class AMPSampler:
    """
    Class for sampling both aggregated markov Processes and noise for paths generated by the gillepsie algorithm.
    """
    def __init__(self):
        self.update_sampler()
    
    def update_sampler(self,rng=None):
        """Method used to update the rng. If no rng is given, the default rng of numpy is used. """
        if rng is None:
            rng = np.random.default_rng()
        self._rng=rng

    def sample_amp(self, topology:Topology, min_rate:float=10.0**2,max_rate:float=10.0**5):
        """Samples a AMP, which satisfies detailed balance, with the given topology. """
        n_o=topology._n_o
        n=topology.n
        top=topology._topology
        f=lambda x: x<n_o
        pi = self._rng.dirichlet(np.ones(n)) # Sample stationary distribution
        Q = np.zeros((n, n))
        
        for i in range(n): #Set the lower diagonal to Q_ji=Q_ij pi[i]/pi[j]
            for j in range(i+1, n):
                Q[i, j] = self._rng.uniform(min_rate,max_rate)*top[i,j] # make sure the generator respects the topology
                Q[j, i] = (pi[i] / pi[j]) * Q[i, j]

        for i in range(n):
            Q[i, i] = -np.sum(Q[i, :])
        
        # Since by construction diag(pi)@Q=Q.T@diag(pi) we know that pi is still a valid stationary distribution

        return AggregatedMarkovProcess(topology,Q,f,pi)
        
    def add_noise(self,y:matrix,SNR:float=5.0):
        """
        Adds white noise with sd 1/SNR to the time series y of shape n.
        """
        return y + np.random.normal(loc=0,scale=1/SNR,size=y.shape)

    def sample_from_family(self,top_fam:TopologyFamily,min_rate:float=10.0**2,max_rate:float=10.0**5)->AggregatedMarkovProcess:
        return self.sample_amp(top_fam.get_topology_by_index(top_fam.sample_index()),min_rate,max_rate)