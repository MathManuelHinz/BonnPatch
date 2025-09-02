# Introduction

BonnPatch is a fast python package contribution lots of utilities for the analysis of ion channels via single channel patch-clamp recording {cite}`qin2004`
It is build with the rich mathematical theory of Fredkin, Rice {cite}`fredkin1986`, Kienker {cite}`kienker1989` and Bruno et al {cite}`bruno2005`, as well as some 
practical approaches such as Higher Order Hinkley Detectors (HOHD) {cite}`schultze1993` in mind.

BonnPatch currently has the following features:
- Sampling aggregated Markov processes from user defined topology classes (e.g. linear five state topologies)
- Sampling a path from a aggregated Markov processes
- Computing theoretical densities of a aggregated Markov processes
- Filtering a noisy observed path using HOHD {cite}`schultze1993`
- Computing histograms of (1-D, 2-D) dwell time distributions of aggregated Markov processes, as well as distances between expected theoretical histograms and observed histograms (both over time and after observing all data)

Planned features include
- Fitting and optimization utilities
- Utilities for dealing with equivalent topologies {cite}`bruno2005`



## Bibliography
```{bibliography}
:style: alpha
:filter: docname in docnames
```
