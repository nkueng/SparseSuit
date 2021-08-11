# Sparsesuit
All code behind my MSc Thesis about reducing the number of sensors in an inertial motion capture suit.

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

# install pymusim
git clone git@github.com:Rokoko/IMUsimulator.git
cd IMUsimulator
mkdir build && cd build
cmake ..
pip install ninja
pip install -e .
```