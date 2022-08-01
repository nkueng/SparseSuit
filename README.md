# SparseSuit
All code behind my Master's thesis about reducing the number of sensors in an inertial motion capture suit.

<p float="left">
  <img src="/7.gif" width="300" />
  <img src="/7 (1).gif" width="300" /> 
</p>

## Installation
In a directory of choice:

```bash
# install this project
git clone git@github.com:nkueng/SparseSuit.git
cd SparseSuit
virtualenv venv
source venv/bin/activate
pip install -e .
cd ..

# install pymusim (required only for data synthesis)
git clone git@github.com:Rokoko/IMUsimulator.git
cd IMUsimulator
mkdir build && cd build
cmake ..
pip install ninja
pip install -e .
cd ../..

# install procrustes
git clone git@github.com:theochem/procrustes.git
cd procrustes
pip install .

```
