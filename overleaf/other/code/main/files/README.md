# View mlflow ui    
    mlflow ui  --backend-store-uri postgresql://artinmajdi:1234@localhost/chest_db_v2 \
                  --default-artifact-root <absolute-path-ro-artifacts> \
                  --host 0.0.0.0
                  --port 5000



# How to run
entry_points:
    main:
        parameters:
            epoch: {type: int, default: 3}
            bsize: {type: int, default: 30}
            max_sample: {type: int, default: 1000}
        command: python main.py --epoch {epoch} --bsize {bsize} --max_sample {max_sample}
        

# Installation

### Using conda 
    conda env create -f requirements.yaml -n env_name

### Using docker

    docker pull artinmajdi/miniconda-cuda-tensorflow:latest
    
### Install on Mac m1
    Source GitHub: https://github.com/conda-forge/miniforge

    http://blog.wafrat.com/installing-tensorflow-2-5-and-jupyter-lab-on-m1/
    
### Installing  packages manually

    conda install -c anaconda keras tensorflow-gpu
    conda install -c anaconda numpy pandas matplotlib 
    conda install -c anaconda scikit-learn scikit-image
    conda install -c anaconda psycopg2 git
    pip install mlflow==1.12.1
    pip install pysftp

### with specific versions
    conda install python=3.9.4
    conda install tensorflow-gpu=2.4
    pip install mlflow==1.12.1 pysftp==0.2.9
    conda install psycopg2==2.8.5
    conda install keras==2.4.3
    pip install scikit-learn scikit-image numpy pandas docker tqdm
    conda install git
