# DQN-hw
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ingeechart/DQN-hw/blob/gym/main.ipynb)
https://github.com/ntasfi/PyGame-Learning-Environment

## 설치
* Git 저장소를 다운로드 받습니다.
```batch
:: 0. Clone with submodule
git clone https://github.com/ingeechart/DQN-hw.git --recurse-submodules

:: 1. Install required modules.
python -m pip install -r requirements.txt

:: 2. Install gym-flappy-bird
cd gym-flappy-bird && python -m pip install -e . && cd ..

:: 3. Install PyGame-Learning-Environment(ple).
cd PyGame-Learning-Environment && python -m pip install -e . && cd ..
```
or (Windows)
```batch
>> ./install.bat
```
or (MacOS/Linux)
```bash
$ ./install.sh
```

* [Google Colab](https://colab.research.google.com) 사용을 위하여 [Google Drive](https://drive.google.com)에 업로드한다.
- Google Drive `내 드라이브` 디렉토리 안에 `Colab Notebooks` 디렉토리를 생성한다.
- 생성된 `Colab Notebooks` 디렉토리 안에 위에서 설치한 `DQN-hw` 폴더를 업로드한다.
- [Google Colab](https://colab.research.google.com)에 접속한다.
- 상단 메뉴에서 `파일` > `노트 업로드`를 클릭한다.
![Colab Note Upload](https://github.com/ingeechart/DQN-hw/blob/main/res/colab_intro.PNG)
- 업로드 창이 나타나면 `DQN-hw` 폴더 안에 있는 `main.ipynb` 파일을 업로드한다. (드래그 가능)
![Colab Notebook Upload](https://github.com/ingeechart/DQN-hw/blob/main/res/colab_upload_notebook.PNG)
- 짜잔! Colab Notebook이 생성되었다.
![Colab Notebook](https://github.com/ingeechart/DQN-hw/blob/main/res/colab_notebook.PNG)
- 구글 드라이브는 아래와 같은 구조를 갖게 된다. (Colab Notebook의 이름은 FlappyBird가 아닐 수도 있다.)
![Google Drive Colab Notebooks](https://github.com/ingeechart/DQN-hw/blob/main/res/gdrive.PNG)

## GPU 설정
* 위에서 생성된 Colab Notebook으로 들어간다.
![Colab Notebook](https://github.com/ingeechart/DQN-hw/blob/main/res/colab_notebook.PNG)
* 상단 메뉴에서 `수정` > `노트 설정`을 클릭한다.
![Colab Notebook Settings](https://github.com/ingeechart/DQN-hw/blob/main/res/colab_notebook_settings.PNG)
* 하드웨어 가속기 중 `GPU`를 선택하고 저장한다.
![Colab Notebook Settings GPU](https://github.com/ingeechart/DQN-hw/blob/main/res/colab_notebook_settings_gpu.PNG)

## 실행 및 인증
* 상단 메뉴에서 `런타임` > `모두 실행`을 클릭한다.
![Colab Runtime](https://github.com/ingeechart/DQN-hw/blob/main/res/colab_notebook_runtime.PNG)
* **이 프로젝트는 Colab과 Google Drive를 연동하는 과정에서 인증이 필요하다.**
* URL과 함께 `Enter verification code:` 입력창이 나오게 되는데 위의 URL에 접속하여 인증 절차 진행 후 나오는 코드를 입력 후 엔터를 치면 된다. **총 2회 진행한다.**
![Colab Verification](https://github.com/ingeechart/DQN-hw/blob/main/res/colab_verification_code.PNG)
* 코드 블럭이 순서대로 실행되는 것을 기다린다. 마지막 블럭에서 게임이 진행되며 화면이 출력되는 것을 확인한다.
![Colab Run](https://github.com/ingeechart/DQN-hw/blob/main/res/colab_notebook_run.PNG)
* 위의 화면에서 `Device: cuda`가 정상적으로 출력되었는지 확인한다. `Device: cpu` 등의 메시지가 나온다면 [GPU 설정](https://github.com/ingeechart/DQN-hw#GPU-설정) 과정으로 돌아간다.