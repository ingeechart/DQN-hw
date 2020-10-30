:: 1. Install required modules.
python -m pip install -r requirements.txt

:: 2. Install gym-flappy-bird
cd gym-flappy-bird && python -m pip install -e . && cd ..

:: 3. Install PyGame-Learning-Environment(ple).
:: 3.1 git clone (PLE를 수정하기 위하여 submodule로 추가하지 않음.)
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
:: 3.3 Install PyGame-Learning-Environment
cd PyGame-Learning-Environment && python -m pip install -e . && cd ..