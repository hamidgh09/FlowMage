# FlowMage

FlowMage is a system that leverages Large Language Models (LLMs) to perform code analysis and extract essential information from stateful network functions (NFs) prior to their deployment on a server. It is designed to find the optimum RSS configuration to deploy a chain of stateful network functions on a commodity server. 

<p align="center">
<br>
<img src="flowmage-design.png" alt="FlowMage working diagram" width="45%"/>
<br>
</p>

FlowMage is framework agnostic by design, however it requires customized functions to (1) find related codes for a network function in a given framework, (2) parse input configuration files, and (3) apply the suggested configurations by writing back the output file with a correct syntax.

For more information check out our paper at EuroMLSys '24.

The current version of FlowMage is customized for [Fastclick][Fastclick]

## Repository Organization

This repository contains information, experiment setups, and some of the results presented in our EuroMLSys'24 paper. More specifically:
- `graphs` contains all raw experiment results, and gnuplot scripts to create graphs that are used in the evaluation section of the paper. Note that raw results are generated by [NPF][NPF] which we used to automate our experiments. We do not add NPF scripts to this repository to reduce the complexities for running Flowmage.
- `FlowMage` contains all source code and requirements for analyzing a Framework. The current version of system is customized for [Fastclick][Fastclick] as mentioned earlier; hence, you will find list of FastClick elements and extracted results for them in this repository. 

## Testbed

**NOTE: Before running the experiments, you need to prepare your testbed according to the following guidelines.**
The current version of FlowMage contains two separate modules for (1) extracting NFs attributes from an LLM and (2) receiving a click configuration file, use the extracted attributes, and finally returning back a file containing optimized click configuration.

### Extracting NFs' Features


### Running Solver


## Citing our paper
If you use FlowMage, please cite our paper:

```bibtex
@inproceedings{FlowMage,
author = {Ghasemirahni, Hamid and Farshin, Alireza},
title = {Deploying Stateful Network Functions Efficiently
using Large Language Models},
year = {2024},
isbn = {79-8-4007-0541-0/24/04},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3642970.3655836},
doi = {10.1145/3642970.3655836},
booktitle = {Proceedings of the 4th Workshop on Machine Learning and Systems},
numpages = {11},
keywords = {Intra-Server Load Balancing, Stateful Network
Functions, LLMs, Static Code Analysis, RSS Configuration},
location = {Athens, Greece},
series = {EuroMLSys '24}
}
```

## Help
If you have any question regarding the code or the paper you can contact me (hamidgr [at] k t h [dot] s e).

[NPF]: https://github.com/tbarbette/npf
[FastClick]: https://github.com/tbarbette/fastclick