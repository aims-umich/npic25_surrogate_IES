# Surrogate-driven Variance-based Sensitivity Analysis of Thermal Storage Tanks in Integrated Energy Systems
NPIC&amp;HMIT-2025: Surrogate-driven Variance-based Sensitivity Analysis of Thermal Storage Tanks in Integrated Energy Systems

## Paper

Sene, S., Lin, L., Kim, J., Radaideh, M.I. (2025). “Surrogate-driven Variance-based Sensitivity Analysis of Thermal Storage Tanks in Integrated Energy Systems,” In: 14th Nuclear Plant Instrumentation, Control & Human-Machine Interface Technologies (NPIC & HMIT 2025), Chicago, Illinois, June 15-18, 2025.

## 🛠️ Environment Installation

To set up the environment for this project, follow these steps:

```bash
# 1. Create a new conda environment with Python 3.10
conda create -n teds python=3.10

# 2. Activate the environment
conda activate teds

# 3. Install pyMAISE from GitHub
pip install git+https://github.com/myerspat/pyMAISE.git

# 4. Install SALib
pip install SALib

# 5. Install Jupyter and Papermill for notebook execution
pip install jupyter papermill

```

## How to generate the results

- `data`: contains the input and output data files as numpy data arrays. This data is used by parts 1&2.  

- Step 1: Go to the folder `part1_surrogate_pymaise` and run the following notebook (it takes quite some time to finish)

  ```bash
  papermill inl_teds_pymaise.ipynb inl_teds_pymaise_out.ipynb
  ```
  or
  ```bash
  nohup papermill inl_teds_pymaise.ipynb inl_teds_pymaise_out.ipynb &
  ```
  
  Results will be saved to the directory `./part1_surrogate_pymaise/results`

- Step 2: Go to the folder `part2_sensitivity` and run the following script
  
  ```bash
  python sensitivity.py
  ```
  Results will be saved to the directory `./part2_sensitivity/nn_results` for the NN surrogate, `./part2_sensitivity/sobol_results` for Sobol indices, and `./part2_sensitivity/fast_results` for FAST results.

## Notes 
  - Under `sobol_results` and `fast_results`, sensors #1 and #7 correspond to `TE_4` and `TW_1` sensors which have the results given in Figure 7 and Figure 8 in the paper.
  - Part 1 results from pyMAISE should generate what you see in Figures 4-6 of the paper. The notebook output in Part 1 includes the metrics in Table 1.   
  - Please note that Sobol and Fast results in Part 2 may have the input parameter ranking switched a little bit due to the randomness of the Monte Carlo methods used to estimate the indices. However, the dominant inputs should remain the same and ranking differences should occur in less dominant inputs. 

