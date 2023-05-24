# # docker build -t datascience -f Docker/Dockerfile_miniconda .
# docker run -v /storage/artinmajdi/:/home/ --network host -itd --name datascience 92f4eee0c52a
# docker exec -it datascience bash
# docker-compose up -d --build
# docker-compose down

#  -------------   Getting Miniconda image   -------------
FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y curl

WORKDIR /home
ENV PATH /opt/conda/envs/base/bin:$PATH
ENV LISTEN_IP 0.0.0.0

EXPOSE 22


# -------------   Adding the conda channgels and metadata to speedup the process   -------------
RUN echo 'starting the installation'
RUN conda config --add channels anaconda
RUN conda config --add channels pytorch
RUN conda config --add channels nvidia
RUN conda config --add channels conda-forge
# RUN conda update --yes --all

#  -------------   Installing the package dependencies   -------------
RUN conda install --yes python=3.10
RUN pip install wget crowd-kit torchxrayvision researchpy
RUN conda install --yes -c conda-forge mamba
RUN mamba install --yes -c conda-forge mlflow psycopg2 pysftp pandas numpy matplotlib wget scikit-learn scikit-image jupyter seaborn tqdm hyperopt umap-learn opencv
RUN mamba install --yes -c conda-forge tensorflow-gpu=2.11 cudatoolkit tensorflow-hub tensorflow-datasets keras
RUN mamba install --yes -c pytorch pytorch torchvision
CMD ["/bin/bash"]
