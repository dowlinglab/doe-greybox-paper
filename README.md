# Pyomo.DoE Grey Box Paper

This repository will hold the case studies used in the Pyomo.DoE 2.0 paper to expand objective options to include grey box features.

The main additions showcased are utilizing A-optimal (or trace of the covariance matrix), E-optimal (or minimum eigenvalue of the FIM), and Modified-E-optimal (or minimum condition number of the FIM) in optimal experimental design problems. The grey box feature was required to implement these objective options within Pyomo.DoE.

The following section will describe how to set up an environment capable of solving these problems.

## Making an environment with cyipopt

`cyipopt` should be installed first in a conda environment for the best results. This can be done in the command line using a command like:

```
conda create -n <your_environment_name> -c conda-forge cyipopt python=3.12
```

In this install we fix the python version to be 3.12 to avoid bad package interactions.

Next, we install the requirements in `requirements.txt` using the line:

```
pip install -r "requirements.txt"
```

Finally, you need to ensure that the ipopt solver is properly configured. If you wish to use more sophisticated HSL solvers from the coin-or project, we recommend getting them through the `idaes-pse` package using the following instructions in command line:

```
pip install idaes-pse
idaes get-extensions
```

Now, when coding, the statement `import idaes` should include the solvers in your path to make them available. If these are not available still, you can add the directory with the binaries for ipopt and the coin-or HSL solvers to your path. An easy way to find this directory is using the following line:

```
idaes bin-directory
```

Once this environment is generated, you should be able to call the `test_cyipopt_install.py` and this should run without errors.

```
python test_cyipopt_install
```

