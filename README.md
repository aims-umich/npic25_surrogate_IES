# npic25_surrogate_IES
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
