import numpy as np
import logging
from scipy.signal import bessel, filtfilt
from helper_generation import  INTERESTING_MODELS, TOPOLOGIES, NAMES, AggregatedMarkovProcess, AMPSampler, Topology, HigherOrderHinkleyDetector

import gillepsie
from helper_histograms import get_histogram

sampler=AMPSampler()
hohd=HigherOrderHinkleyDetector()
hohd.set_cutoff(hohd.calculate_cutoff(t_res=2))
fs = 100_000 # Sampling freq
order = 4
Wn = 10_000
b, a = bessel(order, Wn, btype='low',fs=fs) 
def generate_data_block(index,num=10000):
    global sampler, hohd, topology, b, a
    T=10
    measured_times=np.arange(start=0,stop=T,step=1/(10**6))
    data={"oc":[],"co":[],"Q":[],"ns":[],"index":[]}
    logging.info("Data generation of process "+str(index)+" started")
    for _ in range(num):
            Z=sampler.sample_amp()
            ts,xs=gillepsie.sample_path(Z,T=T)
            ts=np.asarray(ts)
            xs=np.asarray(xs)
            idxs = (np.searchsorted(ts, measured_times, side="right") - 1)
            values_at_measured_times = np.where(idxs >= 0, Z._f(xs[idxs]), np.nan)
            measured_values=sampler.add_noise(values_at_measured_times)
            ys=np.array(hohd.filter(filtfilt(b,a,measured_values)),np.bool)
            _,_,h_oc,h_co,_=get_histogram(measured_times,ys,res=60,min_exp=-5,max_exp=1)
            data["oc"].append(h_oc)
            data["co"].append(h_co)
            data["Q"].append(Z._Q)
            data["ns"].append([Z._topology._n_o,Z._topology._n_c])
            data["index"].append(Z._topology.top_index)
    np.save(f"data/{index}_hoc",np.asarray(data["oc"]))
    np.save(f"data/{index}_hco",np.asarray(data["co"]))
    np.save(f"data/{index}_Q",np.asarray(data["Q"]))
    np.save(f"data/{index}_ns",np.asarray(data["ns"]))
    np.save(f"data/{index}_index",np.asarray(data["index"]))
    logging.info("Data generation of process "+str(index)+" finished")
if __name__ == "__main__":
    offset=0
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",filename="Datageneration.log", encoding="utf-8", level=logging.INFO,force=True)
    logging.info(f"Started data generation script")
    for i in range(10):
        generate_data_block(offset+i)
