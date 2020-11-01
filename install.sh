#!/usr/bin/bash
# 0. Clone with submodule
git clone https://github.com/ingeechart/DQN-hw.git --recurse-submodules
#git submodule add https://github.com/ntasfi/PyGame-Learning-Environment.git

# 1. Install required modules.
python -m pip install -r requirements.txt

# 2. Install gym-flappy-bird
cd gym-flappy-bird && python -m pip install -e . && cd ..

# 3. Install PyGame-Learning-Environment(ple).
cd PyGame-Learning-Environment && python -m pip install -e . && cd ..