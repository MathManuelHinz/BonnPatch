import numpy as np
from functools import reduce
import logging
from typing import Callable
from scipy import linalg
import tqdm
import  numpy.typing as npt
from typing import Callable, Tuple, Union, Optional
from scipy.signal import bessel, filtfilt
from scipy.special import poch
import math
import matplotlib.pyplot as plt

import hohd

matrix=npt.NDArray[np.float64]

logger = logging.getLogger(__name__)

TOPOLOGIES=[np.array([[0,0,0,0,1],[0,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[1,0,0,1,0]]),
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

NAMES=["CCCCO","CCCOC","CCOCC","CCCOO","CCOCO","COCCO","OCCCO","CCOOC","COCOC",
       "OOOOC","OOOCO","OOCOO","OOOCC","OOCOC","OCOOC","COOOC","OOCCO","OCOCO"]

INTERESTING_MODELS={
    "TwoHillsFast": {"topology":TOPOLOGIES[5],"name":NAMES[5],"generator":10_000*np.array([[-0.80294985,0.0,0.40123272,0.40171713,0.0],[0.0,-0.01381559,0.0,0.0,0.01381559],[0.10448446,0.0,-0.10448446, 0.0, 0.0],[0.86417804, 0.0, 0.0, -0.95476492, 0.09058688],[0.0, 0.9441882, 0.0, 0.24489628, -1.18908448]])}
}

def lu_inverse(A):
    lu_decomp, piv = linalg.lu_factor(A)
    identity = np.eye(A.shape[0])
    A_inv = np.zeros_like(A)
    
    for i in range(A.shape[0]):
        A_inv[:,i] = linalg.lu_solve((lu_decomp, piv), identity[:,i])
    return A_inv
    
def add_mirrored_topologies():
    for i in range(9):
        n_o=NAMES[i].count("O")
        n_c=NAMES[i].count("C")
        tmp=np.eye(5)
        tmp[:n_c,:n_c]=TOPOLOGIES[i][n_o:,n_o:]
        tmp[n_c:,n_c:]=TOPOLOGIES[i][:n_o,:n_o]
        tmp[:n_c,n_c:]=TOPOLOGIES[i][n_o:,:n_o,]
        tmp[n_c:,:n_c]=TOPOLOGIES[i][:n_o,n_o:]
        TOPOLOGIES.append(tmp)
        NAMES.append(reduce(lambda a,b:a+b,["C"*(a=="O")+"O"*(a=="C") for a in NAMES[i]],""))

class Topology:
    def __init__(self,adjacency_matrix:matrix,n_o,name:str="",top_index:Optional[int]=None):
        self.update_topology(adjacency_matrix,n_o,name,top_index)

    def update_topology(self,adjacency_matrix:matrix,n_o:int,name:str,top_index:Optional[int]):
        self._topology=adjacency_matrix
        self.n:int=adjacency_matrix.shape[0]
        self._n_o=n_o
        self._n_c=self.n-self._n_o
        self._name=name 
        self.top_index=top_index

class HigherOrderHinkleyDetector:

    def __init__(self,order:int=8, cutoff_method:str="auto"):
        self.set_order(order)
        self.set_cutoff(self.calculate_cutoff(cutoff_method=cutoff_method))

    def set_order(self,order):
        """Updates the order of the detector, default is 8, order 1 is equivalent to a Hinkley detector"""
        self.order=order

    def calculate_cutoff(self,cutoff_method="auto", t_res:int=3, snr:float=5.0,p:float=0.5):
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
            for i in range(1,self.order):
                cg=[0]
                for t in range(1,t_res+1):
                    cg.append(cg[-1]+gs[-1][t])
                gs.append(np.array(cg))
            return p*gs[-1][-1]
        elif cutoff_method=="schultze-draber":
            return 22*p/(snr**2)
        else: raise NotImplementedError

    def set_cutoff(self, cutoff=2):
        """Setter for the cutoff value"""
        self.cutoff=cutoff

    def filter(self,z:matrix,mu_0=0.0,mu_1=1.0,):
        assert mu_0<mu_1
        return np.array(hohd.higher_order_hinkley_detector(z,mu_0,mu_1,self.cutoff,self.order))
    
    def filter_and_plot(self,measured_times:matrix,values_at_measured_times:matrix,measured_values:matrix,mu_0=0.0,mu_1=1.0,show_plot=False,path="images/experiments/hohd.png"):
        assert mu_0<mu_1
        p=(mu_1-mu_0)/2
        gs,jump_idxs,jump_values,filtered_z=hohd.higher_order_hinkley_detector_display(measured_values,mu_0,mu_1,self.cutoff,self.order)
        gs=[np.array(g) for g in gs]
        filtered_z=np.array(filtered_z)
        if not show_plot: plt.clf()
        fig, axs = plt.subplots(2,1,figsize=(13, 9))
        fig.suptitle(f"Accuracy: {(values_at_measured_times==filtered_z).sum()/measured_times.shape[0]:.3f}")
        axs[0].plot(measured_times,measured_values,c="black",alpha=0.3,label="$z_t$")
        axs[0].step(measured_times,values_at_measured_times,where="post",c="red",alpha=0.3,label="$x_t$")
        axs[0].plot(measured_times[:gs[0].shape[0]],p+gs[0],c="purple",alpha=0.6,label=r"$\eta \frac{g_t^"+str(self.order)+r"}{\lambda}$")
        for g in gs[1:]:
            axs[0].plot(measured_times[:g.shape[0]],p+g,c="purple",alpha=0.6)
        axs[0].scatter(measured_times[jump_idxs],jump_values,c="purple",marker="*",alpha=0.6)
        axs[0].legend()

        axs[1].plot(measured_times,measured_values,c="black",alpha=0.3,label="$z_t$")
        axs[1].step(measured_times,values_at_measured_times,where="post",c="red",alpha=0.3,label="$x_t$")
        axs[1].plot(measured_times,filtered_z,c="blue",alpha=0.6,label=r"$\mathcal{F}(z_t)$")
        axs[1].legend()
        if show_plot: plt.show()
        else: plt.savefig(path,dpi=300)

        return filtered_z

class AggregatedMarkovProcess:
    def __init__(self,topology:Topology,Q:matrix,f:Callable[[int],int],pi:matrix=None):
        self.update_process(topology,Q,pi,f)
        
    
    def get_stationary_distribution(self)->matrix:
        """
        Return the stationary distribution and relaxation time of the Markov jump process.
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
        Calculate theoretical densities of the one and two dimensional dwell time distributions. Admissible modes: matrix, exponential
        Returns the densities f_o,f_c,f_oc,f_co
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

        if mode=="matrix":
            f_o=lambda t: (pi_o@linalg.expm(q_oo*t)@q_oc).sum()
            f_c=lambda s: (pi_c@linalg.expm(q_cc*s)@q_co).sum()
            f_oc=lambda t,s: (pi_o@linalg.expm(q_oo*t)@q_oc@linalg.expm(q_cc*s)@q_co).sum()
            f_co=lambda s,t: (pi_c@linalg.expm(q_cc*s)@q_co@linalg.expm(q_oo*t)@q_oc).sum()

        elif mode=="exponential":
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
        return self._Q, self._pi

class AMPSampler:
    def __init__(self):
        self.update_sampler()
    
    def update_sampler(self,rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self._rng=rng

    def sample_amp(self, topology:Union[Topology,None]=None, min_rate:float=10.0**2,max_rate:float=10.0**5):
        """Samples a AMP, which satisfies detailed balance, from top. If top is None, a random topology is generated."""
        if topology==None:
            top_index=self._rng.integers(18)
            top=TOPOLOGIES[top_index]
            top_name=NAMES[top_index]
            n_o=top_name.count("O")
            n=top.shape[0]
            topology=Topology(top,n_o,top_name,top_index=top_index)
        else:
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
    
    def sample_path_progressbar(self,X:AggregatedMarkovProcess,T:float=100.0,t_0:float=0.0):
        """
        Given Y=(f,X), samples a path of X and returns ts, xs which are are times and values at which events happen.
        This version updates a progress bar in the terminal.
        """
        q,pi=X.get_generator_initial()
        t=t_0 
        s=self._rng.choice(q.shape[0],p=pi)
        ts=[t_0]
        xs=[s]
        pbar = tqdm.tqdm(total=T)
        while t<T:
            pbar.n=np.floor(t) 
            pbar.refresh()
            scale=-1/q[s,s]
            tau=self._rng.exponential(scale)
            t+=tau
            if t>=T: break 
            p=q[s,:].copy()*scale 
            p[s]=0
            s=self._rng.choice(q.shape[0],p=p)
            ts.append(t)
            xs.append(s)
        pbar.n=np.floor(T) 
        pbar.refresh()
        ts.append(T)
        xs.append(xs[-1])
        return np.array(ts),np.array(xs)
    
    def sample_path(self,Y:AggregatedMarkovProcess,T:float=100.0,t_0:float=0.0)->Tuple[matrix,matrix]:
        """Given Y=(f,X), samples a path of X and returns ts, xs which are are times and values at which events happen."""
        q,pi=Y.get_generator_initial()
        t=t_0 
        s=self._rng.choice(q.shape[0],p=pi)
        ts=[t_0]
        xs=[s]
        while t<T:
            scale=-1/q[s,s]
            tau=self._rng.exponential(scale)
            t+=tau
            if t>=T: break 
            p=q[s,:].copy()*scale 
            p[s]=0
            s=self._rng.choice(q.shape[0],p=p)
            ts.append(t)
            xs.append(s)
        ts.append(T)
        xs.append(xs[-1])
        return np.array(ts),np.array(xs)
    
    def add_noise(self,y:matrix,SNR:float=5.0):
        """
        Adds white noise with sd 1/SNR to the time series y of shape n.
        """
        return y + np.random.normal(loc=0,scale=1/SNR,size=y.shape)