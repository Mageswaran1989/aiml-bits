# Android Malware Detection System

Dataset: https://www.kaggle.com/datasets/shashwatwork/android-malware-dataset-for-machine-learning


## Setup

```
conda create -n mds python=3.9
conda activate mds
pip install -r requirements.txt
```
## How to run?

```
cd /path/to/aiml-bits/
export PYTHONPATH=src/
python main.py
streamlit run src/mds/ui/home.py
```

Web UI: http://localhost:8501
