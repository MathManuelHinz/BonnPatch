import numpy as np 
import  numpy.typing as npt
matrix=npt.NDArray[np.float64]
import matplotlib.pyplot as plt
from helper_generation import HigherOrderHinkleyDetector

import hohd

def filter_and_plot(hohd_handler:HigherOrderHinkleyDetector,measured_times:matrix,values_at_measured_times:matrix,measured_values:matrix,mu_0=0.0,mu_1=1.0,show_plot=False,path="images/experiments/hohd.png"):
    assert mu_0<mu_1
    p=(mu_1-mu_0)/2
    gs,jump_idxs,jump_values,filtered_z=hohd.higher_order_hinkley_detector_display(measured_values,mu_0,mu_1,hohd_handler.cutoff,hohd_handler.order)
    gs=[np.array(g) for g in gs]
    filtered_z=np.array(filtered_z)
    if not show_plot: plt.clf()
    fig, axs = plt.subplots(2,1,figsize=(13, 9))
    fig.suptitle(f"Accuracy: {(values_at_measured_times==filtered_z).sum()/measured_times.shape[0]:.3f}")
    axs[0].plot(measured_times,measured_values,c="black",alpha=0.3,label="$z_t$")
    axs[0].step(measured_times,values_at_measured_times,where="post",c="red",alpha=0.3,label="$x_t$")
    axs[0].plot(measured_times[:gs[0].shape[0]],p+gs[0],c="purple",alpha=0.6,label=r"$\eta \frac{g_t^"+str(hohd_handler.order)+r"}{\lambda}$")
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