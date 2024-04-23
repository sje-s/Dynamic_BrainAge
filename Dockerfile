FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

RUN mkdir -p /opt/conda 
RUN apt-get update
RUN apt-get install -y wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O /opt/conda/miniconda.sh 
RUN bash /opt/conda/miniconda.sh -b -p /opt/miniconda
ENV PATH /opt/miniconda/bin:$PATH

RUN mkdir /app
RUN mkdir /out
RUN mkdir /data

RUN ./opt/miniconda/bin/activate

RUN conda update conda -y

RUN conda install -y setuptools
RUN conda create -y --name catalyst3.9 python=3.9.15
SHELL ["/bin/bash", "-i", "-c"]
RUN conda init bash
RUN conda activate catalyst3.9

COPY . /app

RUN pip install -r /app/requirements.txt

