import numpy as np
from scipy.signal import bessel, filtfilt
import matplotlib.pyplot as plt
import networkx as nx
import logging

from helper_generation import  INTERESTING_MODELS, TOPOLOGIES, NAMES, add_mirrored_topologies, AggregatedMarkovProcess, AMPSampler, Topology, HigherOrderHinkleyDetector
from helper_histograms import get_histogram, normalized_volume_deviation, get_histogram_and_deviation, get_histogram_and_hellinger, get_histogram_and_l1

import gillepsie

import time

TEST_TOPOLOGIES=0
COMPARE_THEORETICAL_DENSITIES=1
PLOT_CONVERGENCE=2
COMPARE_HISTOGRAMS_1D=3
COMPARE_HISTOGRAMS_2D=4
PLOT_PATH=5
DRAW_TOPOLOGIES=6

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

logger = logging.getLogger(__name__)

def test_linear_topology(top,name):
    assert top.shape==(5,5)
    assert top.sum()==8
    assert top.sum(axis=0).max()==2 and top.sum(axis=1).max()==2
    assert top.sum(axis=0).min()==1 and top.sum(axis=1).min()==1
    logger.info(f"{name} passed")

def compare_theoretical_densities(Z:AggregatedMarkovProcess,bin_edges,show_plot):
    f_o_matrix,f_c_matrix,f_oc_matrix,f_co_matrix=Z.get_theoretical_densities(mode="matrix")
    f_o_exponential,f_c_exponential,f_oc_exponential,f_co_exponential=Z.get_theoretical_densities(mode="exponential")
    if not show_plot: plt.clf()
    fig, axs = plt.subplots(1,2,figsize=(13, 9))
    fig.suptitle(f"Comparison: {Z._topology._name}")
    density=np.diff(bin_edges)
    ts = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    axs[0].set_title(r"$V_D(f_o^\text{matrix},f_o^\text{exponential})$"+f"={normalized_volume_deviation(np.array([f_o_matrix(t) for t in ts]),np.array([f_o_exponential(t) for t in ts])):.5f}")
    axs[0].plot(ts,[f_o_matrix(t) for t in ts],label="Matrix based")
    axs[0].scatter(ts,[f_o_exponential(t) for t in ts],label="Exponential mixture",c="red",s=10,alpha=0.6)
    axs[0].set_xscale("log")
    axs[0].legend()
    axs[1].set_title(r"$V_D(f_c^\text{matrix},f_c^\text{exponential})$"+f"={normalized_volume_deviation(np.array([f_c_matrix(t) for t in ts]),np.array([f_c_exponential(t) for t in ts])):.5f}")
    axs[1].plot(ts,[f_c_matrix(t) for t in ts],label="Matrix based")
    axs[1].scatter(ts,[f_c_exponential(t) for t in ts],label="Exponential mixture",c="red",s=10,alpha=0.6)
    axs[1].set_xscale("log")
    axs[1].legend()
    plt.legend()
    if show_plot: plt.show()
    else: plt.savefig("images/experiments/theoretical_densities.png",dpi=300)
    if not show_plot: plt.clf()
    density=np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    theoretical_histogram_oc_matrix=np.array([[f_oc_matrix(t,s)*density[i]*density[j] for  i,t in enumerate(bin_centers)] for j,s in enumerate(bin_centers)]).astype(float)
    theoretical_histogram_oc_exponential=np.array([[f_oc_exponential(t,s)*density[i]*density[j] for  i,t in enumerate(bin_centers)] for j,s in enumerate(bin_centers)]).astype(float)
    theoretical_histogram_co_matrix=np.array([[f_co_matrix(s,t)*density[i]*density[j] for  j,s in enumerate(bin_centers)] for i,t in enumerate(bin_centers)]).astype(float)
    theoretical_histogram_co_exponential=np.array([[f_co_exponential(s,t)*density[i]*density[j] for  j,s in enumerate(bin_centers)] for i,t in enumerate(bin_centers)]).astype(float)
    fig, axs = plt.subplots(2,2,figsize=(13, 9))
    title=r"$\substack{V_D(f_{oc}^\text{matrix},f_{oc}^\text{exponential})"+f"={normalized_volume_deviation(theoretical_histogram_oc_matrix,theoretical_histogram_oc_exponential):.5f}"+r"\\V_D(f_{co}^\text{matrix},f_{co}^\text{exponential})"+f"={normalized_volume_deviation(theoretical_histogram_co_matrix,theoretical_histogram_co_exponential):.5f}"+r"}$"
    fig.suptitle(title)
    axs[0,0].set_title(r"$f_{oc}^\text{matrix}$")
    axs[0,0].set_xscale("log")
    axs[0,0].set_yscale("log")
    pc1=axs[0,0].pcolormesh(bin_centers,bin_centers,theoretical_histogram_oc_matrix)
    fig.colorbar(pc1)
    axs[1,0].set_title(r"$f_{oc}^\text{exponential}$")
    axs[1,0].set_xscale("log")
    axs[1,0].set_yscale("log")
    pc2=axs[1,0].pcolormesh(bin_centers,bin_centers,theoretical_histogram_oc_exponential)
    fig.colorbar(pc2)
    axs[0,1].set_title(r"$f_{co}^\text{matrix}$")
    axs[0,1].set_xscale("log")
    axs[0,1].set_yscale("log")
    pc3=axs[0,1].pcolormesh(bin_centers,bin_centers,theoretical_histogram_co_matrix)
    fig.colorbar(pc3)
    axs[1,1].set_title(r"$f_{co}^\text{exponential}$")
    axs[1,1].set_xscale("log")
    axs[1,1].set_yscale("log")
    pc4=axs[1,1].pcolormesh(bin_centers,bin_centers,theoretical_histogram_co_exponential)
    fig.colorbar(pc4)
    if show_plot: plt.show()
    else: plt.savefig("images/experiments/theoretical_densities2d.png",dpi=300)

def plot_convergence(deviations,top_name,show_plot,mode="Deviations",distance="V_D"):
    if not show_plot: plt.clf()
    fig, axs = plt.subplots(1,2,figsize=(13, 9))
    fig.suptitle(f"Convergence({mode}): {top_name}")
    axs[0].loglog(deviations["o"],label=f"${distance}(h_o,H(f_o))$")
    axs[0].loglog(deviations["c"],label=f"${distance}(h_c,H(f_c))$")
    axs[0].legend()
    axs[1].loglog(deviations["oc"],label="$"+distance+"(h_{oc},H(f_{oc}))$")
    axs[1].loglog(deviations["co"],label="$"+distance+"(h_{co},H(f_{co}))$")
    axs[1].legend()
    plt.legend()
    if show_plot: plt.show()
    else: plt.savefig(f"images/experiments/convergence_matrix_{mode.replace("$","").replace("^","")}.png",dpi=300)

def plot_convergence2(deviations_matrix,deviations_expo,top_name,show_plot,mode="deviations",distance="V_D"):
    if not show_plot: plt.clf()
    fig, axs = plt.subplots(1,2,figsize=(13, 9))
    fig.suptitle(f"Convergence({mode}): {top_name}") 
    axs[0].loglog(deviations_matrix["o"],label="$"+distance+r"(h_o,H(f_o)^\text{matrix})$",alpha=0.75)
    axs[0].loglog(deviations_matrix["c"],label="$"+distance+r"(h_c,H(f_c)^\text{matrix})$",alpha=0.75)
    axs[0].loglog(deviations_expo["o"],label="$"+distance+r"(h_o,H(f_o))^\text{exponential}$",alpha=0.5)
    axs[0].loglog(deviations_expo["c"],label="$"+distance+r"(h_c,H(f_c))^\text{exponential}$",alpha=0.5)
    axs[0].legend()
    axs[1].loglog(deviations_matrix["oc"],label="$"+distance+r"(h_{oc},H(f_{oc}^\text{matrix}))$",alpha=0.75)
    axs[1].loglog(deviations_matrix["co"],label="$"+distance+r"(h_{co},H(f_{co}^\text{matrix}))$",alpha=0.75)
    axs[1].loglog(deviations_expo["oc"],label="$"+distance+r"(h_{oc},H(f_{oc}^\text{exponential}))$",alpha=0.5)
    axs[1].loglog(deviations_expo["co"],label="$"+distance+r"(h_{co},H(f_{co}^\text{exponential}))$",alpha=0.5)
    axs[1].legend()
    plt.legend()
    if show_plot: plt.show()
    else: plt.savefig(f"images/experiments/convergence_{mode.replace("$","").replace("^","")}.png",dpi=300)

def compare_convergence(plots,top_name,show_plot):
    if not show_plot: plt.clf()
    fig, axs = plt.subplots(1,2,figsize=(13, 9))
    fig.suptitle(f"Convergence: {top_name}") 
    for plot in plots:
        axs[0].loglog(plot["matrix"]["o"],label="$"+plot["distance"]+r"(h_o,H(f_o)^\text{matrix})$",alpha=0.75)
        axs[0].loglog(plot["matrix"]["c"],label="$"+plot["distance"]+r"(h_c,H(f_c)^\text{matrix})$",alpha=0.75)
        axs[0].loglog(plot["expo"]["o"],label="$"+plot["distance"]+r"(h_o,H(f_o))^\text{exponential}$",alpha=0.5)
        axs[0].loglog(plot["expo"]["c"],label="$"+plot["distance"]+r"(h_c,H(f_c))^\text{exponential}$",alpha=0.5)
        axs[0].legend()
        axs[1].loglog(plot["matrix"]["oc"],label="$"+plot["distance"]+r"(h_{oc},H(f_{oc}^\text{matrix}))$",alpha=0.75)
        axs[1].loglog(plot["matrix"]["co"],label="$"+plot["distance"]+r"(h_{co},H(f_{co}^\text{matrix}))$",alpha=0.75)
        axs[1].loglog(plot["expo"]["oc"],label="$"+plot["distance"]+r"(h_{oc},H(f_{oc}^\text{exponential}))$",alpha=0.5)
        axs[1].loglog(plot["expo"]["co"],label="$"+plot["distance"]+r"(h_{co},H(f_{co}^\text{exponential}))$",alpha=0.5)
        axs[1].legend()
    plt.legend()
    if show_plot: plt.show()
    else: plt.savefig(f"images/experiments/convergence_comparison.png",dpi=300)

def compare_densities_histograms1d(Z:AggregatedMarkovProcess,h_o,h_c,bin_edges,show_plot):
    density=np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    theoretical_histogram_o=np.array([Z.f_o(t)*density[i] for  i,t in enumerate(bin_centers)])
    theoretical_histogram_c=np.array([Z.f_c(t)*density[i] for  i,t in enumerate(bin_centers)])
    if not show_plot: plt.clf()
    fig, axs = plt.subplots(1,2,figsize=(13, 9))
    fig.suptitle(Z._topology._name)
    #h_o
    axs[0].set_title("$h_o$")
    axs[0].step(bin_centers, h_o/h_o.sum(), where='mid', label=f"Histogram: $V_D=${normalized_volume_deviation(h_o,h_o.sum()*theoretical_histogram_o):.5f}")
    axs[0].set_xscale("log")
    axs[0].plot(bin_centers,theoretical_histogram_o,label=r"$H\left(f_o^{\text{matrix}}\right)$")
    axs[0].legend()
    #h_c
    axs[1].set_title("$h_c$")
    axs[1].step(bin_centers, h_c/h_c.sum(), where='mid', label=f"Histogram: $V_D=${normalized_volume_deviation(h_c,h_c.sum()*theoretical_histogram_c):.5f}")
    axs[1].set_xscale("log")
    axs[1].plot(bin_centers,theoretical_histogram_c,label=r"$H\left(f_c^{\text{matrix}}\right)$")
    axs[1].legend()
    if show_plot: plt.show()
    else: plt.savefig("images/experiments/histograms1d.png",dpi=300)

def compare_densities_histograms2d(Z:AggregatedMarkovProcess,h_oc,h_co,bin_edges,show_plot): 
    density=np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    theoretical_histogram_oc=np.array([[Z.f_oc(t,s)*density[i]*density[j] for  i,t in enumerate(bin_centers)] for j,s in enumerate(bin_centers)])
    theoretical_histogram_co=np.array([[Z.f_co(s,t)*density[i]*density[j] for  j,s in enumerate(bin_centers)] for  i,t in enumerate(bin_centers)])
    if not show_plot: plt.clf()
    fig, axs = plt.subplots(2,2,figsize=(13, 9))
    fig.suptitle(Z._topology._name)
    
    # h_oc
    axs[0,0].set_title("$h_{oc}$")
    axs[0,0].set_xscale("log")
    axs[0,0].set_yscale("log")
    pc3=axs[0,0].pcolormesh(bin_edges,bin_edges,h_oc/h_oc.sum())
    fig.colorbar(pc3)
    # h_co
    axs[0,1].set_title("$h_{co}$")
    axs[0,1].set_xscale("log")
    axs[0,1].set_yscale("log")
    pc4=axs[0,1].pcolormesh(bin_edges,bin_edges,h_co/h_co.sum())
    fig.colorbar(pc4)


    # H(f_oc)
    axs[1,0].set_title("$H(f_{oc})$"+f", $V_D$={normalized_volume_deviation(h_oc.sum()*theoretical_histogram_oc,h_oc):.5f}")
    axs[1,0].set_xscale("log")
    axs[1,0].set_yscale("log")
    pc1= axs[1,0].pcolormesh(bin_edges,bin_edges,theoretical_histogram_oc)
    fig.colorbar(pc1)
    # H(f_co)
    axs[1,1].set_title("$H(f_{co})$"+f", $V_D$={normalized_volume_deviation(h_co.sum()*theoretical_histogram_co,h_co):.5f}")
    axs[1,1].set_xscale("log")
    axs[1,1].set_yscale("log")
    pc2=axs[1,1].pcolormesh(bin_edges,bin_edges,theoretical_histogram_co)
    fig.colorbar(pc2)
    
    if show_plot: plt.show()
    else: plt.savefig("images/experiments/histograms2d.png",dpi=300)

def plot_paths(ts,xs,f,idxs,measured_times,measured_values,top_name,sampler:AMPSampler,show_plot):
    if not show_plot: plt.clf()
    #Plot paths 
    fig, axs = plt.subplots(3,1,figsize=(13, 9))
    axs[0].set_title(f"{top_name}: Paths")
    axs[0].step(ts[:idxs[-1]],xs[:idxs[-1]],where="post")
    axs[0].step(measured_times,measured_values,where="post")
    axs[0].scatter(measured_times,measured_values)

    # Plot aggregated paths
    axs[1].set_title(f"{top_name}: Aggregated paths")
    axs[1].step(ts[:idxs[-1]],f(xs[:idxs[-1]]),where="post")
    axs[1].step(measured_times,measured_values,where="post")
    
    # Plot aggregated paths with noise
    noised_values=sampler.add_noise(measured_values)
    axs[2].set_title(f"{top_name}: Aggregated paths with noise, SNR={5}")
    axs[2].step(measured_times,measured_values,where="post")
    axs[2].plot(measured_times,noised_values,c="orange")
    if show_plot: plt.show()
    else: plt.savefig("images/experiments/paths.png",dpi=300)
    return noised_values


def main():
    experiments=[TEST_TOPOLOGIES,
                 COMPARE_THEORETICAL_DENSITIES,
                 PLOT_CONVERGENCE,
                 COMPARE_HISTOGRAMS_1D,
                 COMPARE_HISTOGRAMS_2D,
                 PLOT_PATH,
                 DRAW_TOPOLOGIES]
    
    time_shown=500
    T=100
    res=60
    show_plots=False
    topology="TwoHillsFast"
    logging.basicConfig(filename="linear_five_state_example.log", encoding="utf-8", level=logging.INFO,force=True)
    logging.info(f"Started example script")
    # Create mirrored topologies
    add_mirrored_topologies()

    #Test topologies
    if TEST_TOPOLOGIES in experiments:
        for i in range(18):
            test_linear_topology(TOPOLOGIES[i],NAMES[i])

    # Draw topologies
    if DRAW_TOPOLOGIES in experiments:
        fig, axs=plt.subplots(2,9,figsize=(13,9))
        for i in range(9):
            n_o=NAMES[i].count("O")
            n_c=NAMES[i].count("C")
            colors=[]
            colors2=[]
            for _ in range(n_o):
                colors.append("green")
            for _ in range(n_c):
                colors.append("red")
                colors2.append("green")
            for _ in range(n_o):
                colors2.append("red")
            axs[0,i].set_title(NAMES[i])
            axs[1,i].set_title(NAMES[i+9])
            nx.draw_networkx(nx.from_numpy_array(TOPOLOGIES[i]),node_color=colors,with_labels=False,ax=axs[0,i])
            nx.draw_networkx(nx.from_numpy_array(TOPOLOGIES[i+9]),node_color=colors2,with_labels=False,ax=axs[1,i])
        if show_plots: plt.show()
        else: plt.savefig("images/experiments/topologies.png",dpi=300)

    sampler=AMPSampler()
    hohd=HigherOrderHinkleyDetector()
    #Sample Q-Matrix
    if topology==None:
        Z=sampler.sample_amp()
    else:
        topology_dict=INTERESTING_MODELS[topology]
        Z=AggregatedMarkovProcess(Topology(topology_dict["topology"],n_o=topology_dict["name"].count("O"),name=topology_dict["name"]),Q=topology_dict["generator"],f=lambda x: x<n_o)
    #Sample path
    if COMPARE_HISTOGRAMS_1D in experiments or COMPARE_HISTOGRAMS_2D in experiments or PLOT_CONVERGENCE in experiments or PLOT_CONVERGENCE or PLOT_PATH in experiments:
        print("Sampling a path")
        start = time.perf_counter()
        ts,xs=gillepsie.sample_path(Z,T=T)
        ts=np.array(ts)
        xs=np.array(xs)
        end=time.perf_counter()
        print(f"Elapsed: {end - start} seconds")

    # Plot path
    if PLOT_PATH in experiments:
        measured_times=np.arange(start=0,stop=ts[-1],step=1/(10**5))[:time_shown] 
        idxs = (np.searchsorted(ts, measured_times, side="right") - 1)[:time_shown]
        values_at_measured_times = np.where(idxs >= 0, Z._f(xs[idxs]), np.nan)
        measured_values=plot_paths(ts,xs,Z._f,idxs,measured_times,values_at_measured_times,Z._topology._name,sampler,show_plots)
        hohd.filter_and_plot(measured_times,values_at_measured_times,measured_values)

    if COMPARE_HISTOGRAMS_1D in experiments or COMPARE_HISTOGRAMS_2D in experiments or PLOT_CONVERGENCE in experiments:
        # Inefficient, but makes the times comparable
        print("Generating histograms and recording deviations")
        start=time.perf_counter()
        Z.get_theoretical_densities()
        h_o,h_c,h_oc,h_co,bin_edges,deviations=get_histogram_and_deviation(ts,Z._f(xs),Z,res=res)
        end=time.perf_counter()
        print(f"Elapsed: {end - start} seconds")
        print("Generating histograms and recording deviations (expo)")
        start=time.perf_counter()
        Z.get_theoretical_densities(mode="exponential")
        h_o,h_c,h_oc,h_co,bin_edges,deviations_expo=get_histogram_and_deviation(ts,Z._f(xs),Z,res=res)
        end=time.perf_counter()
        print(f"Elapsed: {end - start} seconds")
        print("Generating histograms and recording hellinger")
        start=time.perf_counter()
        Z.get_theoretical_densities()
        h_o,h_c,h_oc,h_co,bin_edges,hellinger=get_histogram_and_hellinger(ts,Z._f(xs),Z,res=res)
        end=time.perf_counter()
        print(f"Elapsed: {end - start} seconds")
        print("Generating histograms and recording hellinger (expo)")
        start=time.perf_counter()
        Z.get_theoretical_densities(mode="exponential")
        h_o,h_c,h_oc,h_co,bin_edges,hellinger_expo=get_histogram_and_hellinger(ts,Z._f(xs),Z,res=res)
        end=time.perf_counter()
        print(f"Elapsed: {end - start} seconds")
        print("Generating histograms and recording l1")
        start=time.perf_counter()
        Z.get_theoretical_densities()
        h_o,h_c,h_oc,h_co,bin_edges,l1_dists=get_histogram_and_l1(ts,Z._f(xs),Z,res=res)
        end=time.perf_counter()
        print(f"Elapsed: {end - start} seconds")
        print("Generating histograms and recording l1 (expo)")
        start=time.perf_counter()
        Z.get_theoretical_densities(mode="exponential")
        h_o,h_c,h_oc,h_co,bin_edges,l1_dists_expo=get_histogram_and_l1(ts,Z._f(xs),Z,res=res)
        end=time.perf_counter()
        print(f"Elapsed: {end - start} seconds")

        
        if COMPARE_HISTOGRAMS_1D in experiments:
            compare_densities_histograms1d(Z,h_o,h_c,bin_edges,show_plots)

        if COMPARE_HISTOGRAMS_2D in experiments:
            compare_densities_histograms2d(Z,h_oc,h_co,bin_edges,show_plots)

    if COMPARE_THEORETICAL_DENSITIES in experiments:
        compare_theoretical_densities(Z,bin_edges,show_plots)

    if PLOT_CONVERGENCE in experiments:
        #plot_convergence(deviations,Z._topology._name,show_plots)
        #plot_convergence(hellinger,Z._topology._name,show_plots,mode="Hellinger",distance="\\text{HD}")
        #plot_convergence(l1_dists,Z._topology._name,show_plots,mode="$L^1$",distance="L^1")

        plot_convergence2(deviations,deviations_expo,Z._topology._name,show_plots)
        plot_convergence2(hellinger,hellinger_expo,Z._topology._name,show_plots,mode="Hellinger",distance="\\text{HD}")
        plot_convergence2(l1_dists,l1_dists_expo,Z._topology._name,show_plots,mode="$L^1$",distance="L^1")

        compare_convergence([
            {"matrix":deviations,
             "expo":deviations_expo,
             "distance":"V_D"},
             {"matrix":hellinger,
             "expo":hellinger_expo,
             "distance":"\\text{HD}"},
             {"matrix":l1_dists,
             "expo":l1_dists_expo,
             "distance":"L^1"},
        ],Z._topology._name,show_plots)

    logging.info(f"Example script finished")

if __name__=="__main__":
    main()