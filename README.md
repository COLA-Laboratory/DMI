# Batched Data-Driven Evolutionary Multi-Objective Optimization Based on Manifold Interpolation

![x](https://github.com/COLA-Laboratory/DMI/blob/main/assets/x.gif?raw=true)

![x](https://github.com/COLA-Laboratory/DMI/blob/main/assets/y.gif?raw=true)



**Batched Data-Driven Evolutionary Multi-Objective Optimization Based on Manifold Interpolation**
[Ke Li]()\*, [Renzhi Chen]()\*
[[Paper]]() [[Supplementary]]()



## Overview

This repository contains Python implementation of the algorithm framework for Batched Data-Driven Evolutionary Multi-Objective Optimization Based on Manifold Interpolation.



## Code Structure

algorithms/ --- algorithms definitions
problems/ --- multi-objective problem definitions
revision/ -- patch for Gpy package
scripts/ --- scripts for batch experiments
 ├── build.sh --- complie the c lib for test problems
 ├── run.sh -- run the experiment 
main.py --- main execution file

## Requirements

- Python version: tested in Python 3.7.7
- Operating system: tested in Ubuntu 20.04



## Getting Started

### Basic usage

Run the main file with python with specified arguments:

```bash
python3.7 main.py --problem dtlz7 --n-var 6 --n-obj 2
```

### Parallel experiment

Run the script file with bash, for example:

```bash
./scripts/run.sh
```



## Result

The optimization results are saved in txt format. They are stored under the folder:

```
output/data/{problem}/x{n}y{m}/{algo}-{exp-name}/{seed}/
```

## Citation

If you find our repository helpful to your research, please cite our paper:

```
@article{KeLi2022,
  title={Batched Data-Driven Evolutionary Multi-Objective Optimization Based on Manifold Interpolation},
  author={Li, Ke and Chen, Renzhi},
  journal={IEEE Transactions on Evolutionary Computation},
  year={2022},
  publisher={IEEE}
}
```

