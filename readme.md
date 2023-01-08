# Meta-Learning Guided Surrogate Model for Expensive Multi-Objective Optimization

# Required packages

* Python 3.10
* PyTorch*
* Matplotlib
* Pymoo (https://pymoo.org/)

*Note: Runnable on PyTorch 1.11.0,
and for MPS support you should see the official
[PyTorch documentation](https://pytorch.org/get-started/locally/).

Could install with

```shell
# Install PyTorch with GPU support
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

conda install matplotlib
pip install pymoo
```

# Optional packages

* PyNVML (for GPU monitoring in function `main_benchmarking` in [main.py](./main.py))
* IPython (for interactive `sys.excepthook`)

Could install with

```shell
pip install pynvml
pip install ipython
```

# Usage
Just run [main.py](./main.py) with Python 3.10.

Configurable parameters are in `main` function and [problem_config/example.py](./problem_config/example.py).
