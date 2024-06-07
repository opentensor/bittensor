## 04 Installation on Apple M chip
There are quite a few Python libraries that are not yet compatible with Apple M chipset architecture. The best way to use Bittensor on this hardware is through Conda and Miniforge. The Opentensor team has created a Conda environment that makes installing Bittensor on these systems very easy. 

> NOTE: This tutorial assumes you have installed conda on mac, if you have not done so already you can install it from [here](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html).

1. Create the conda environment from the `apple_m1_environment.yml` file here:
    ```bash
    conda env create -f apple_m1_environment.yml
    ```

2. Activate the new environment: `conda activate bittensor`.
3. Verify that the new environment was installed correctly:
   ```bash
   conda env list
   ```

4. Install bittensor (without dependencies):
    ```bash
    conda activate bittensor        # activate the env 
    pip install --no-deps bittensor # install bittensor
    ```
