# Cutting Plane Sparsification
This is the companion repository to Herman Appelgren's MSc thesis "Cutting Planes for Mixed Integer Optimization". It contains an implementation of the cutting plane sparsification algorithm described in the thesis, as a C++ plugin to the SCIP optimization suite. It also contains benchmark scripts, SCIP configurations and dataset used for the thesis' experimental results. For the Gomory separator implementation for MiniMIP described in the thesis, see the MiniMIP repository at https://github.com/MiniMIP-Opt/MiniMIP.

This repository was published in May 2025, but all contents were created during the thesis project in Sep 2022 to May 2023.

## Thesis abstract
Mixed Integer Programming is an important paradigm within the field of mathematical optimization. In this paradigm, optimization problems are modeled using a set of variables, some linear constraints, a linear objective, and the requirement that a subset of the variables may only assume integer values. All currently successful Mixed Integer Programming solvers are based on the cut-and-branch algorithm. A crucial step in this algorithm involves introducing additional linear constraints, referred to as cutting planes, to eliminate  solutions where fractional values are assigned to integer variables. These solvers have enabled the application of Mixed Integer Programming to a wide array of real-world problems. A key factor behind their success lies in the fact that most instances arising in practice exhibit sparsity, where each linear constraint involves only a small subset of the available variables.

In this thesis, we explore the topic of sparse cutting planes. Our research present a post-processing sparsification procedure designed to make cutting planes sparser while maintaining their validity. We integrate this procedure into the SCIP solver, as well as a density filtering approach that discards all cuts surpassing a predetermined density threshold. Computational experiments conducted on the MIPLIB2017 dataset demonstrate that both approaches improves solve times. The sparsification procedure reduces solve time by 2-3% for instances where SCIP currently generates very dense cuts, while the density filtering approach yields a solve time reduction of 3-8% on the same instances.

## How to use this repository
The code in this repository has the following dependencies:
* Your environment must have CMake, Python 3, and C++ compilers installed.
* SCIP optimization suite: download the SCIP source code, available at https://scipopt.org/, and place it in a directory named `scipoptsuite` in this repository's root directory.
* The Abseil C++ library (https://abseil.io/). The source code for this library is included in the repository, in accordance with its Apache 2.0 license.

The main way to interact with the code in this directory is via the python scripts at the repository's root level. Run them with the `--help` flag to see usage information.

## Code overview
Here follows a brief overview of the repository code:
* `abseil-cpp/`: The Abseil C++ library. See https://abseil.io/ for more information.
* `configs/`: SCIP configuration files used for different experimental setups. See thesis for more information.
* `data/`: Lists of problems used in the thesis benchmarks. The problem names refers to the MIPLIB 2017 dataset, available at https://miplib.zib.de/.
* `results/`: Raw benchmark results which forms the basis for the thesis results chapter.
* `src/sparcifier.[h,cc]`: The sparsification algorithm implementation.
* `src/testmain.cpp`: Entry point for the benchmark executable, and code for extracting problem information from SCIP.
* `runner.py`: Script used to run SCIP on a problem set. It is primarily written to target a SLURM cluster, but also has modes for running experiments locally.
* `create_dataset.py`: Script used to filter the MIPLIB 2017 dataset, as described in the thesis.
* `analyze.py`: Script for analyzing benchmark results and formatting them as LaTeX tables.
