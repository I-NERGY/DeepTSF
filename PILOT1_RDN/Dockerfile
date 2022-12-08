FROM nvidia/cuda:11.6.0-base-ubuntu20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt update

RUN apt install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda --version

WORKDIR /app 

COPY . .

RUN conda env create -f conda.yaml

RUN conda init bash \
    && conda activate darts_38

RUN echo -e "import torch: print(torch.cuda.is_available())" | python

# RUN export MLFLOW_TRACKING_URI="http://localhost:5000" # maybe redundant as I use the .env file in docker-compose

RUN nohup uvicorn api:app --host 0.0.0.0 --port 8080 > uvi_logs.txt

EXPOSE 8080