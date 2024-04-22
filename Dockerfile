FROM nvidia/cuda:11.6.2-base-ubuntu20.04

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

COPY ./conda.yaml ./conda.yaml

RUN conda update conda

RUN conda env create -f conda.yaml

RUN conda init bash

COPY . .

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "DeepTSF_env", "/bin/bash", "-c"]
 
# RUN echo "conda activate DeepTSF_env" > ~/.bashrc

# RUN python -c "import torch: print(torch.cuda.is_available())"

# RUN /root/miniconda3/envs/DeepTSF_env/bin/uvicorn api:app --host 0.0.0.0 --port 8080

# RUN conda list uvicorn

# RUN /root/miniconda3/envs/DeepTSF_env/uvicorn api:app --host 0.0.0.0 --port 8080

# RUN export MLFLOW_TRACKING_URI="http://localhost:5000" # maybe redundant as I use the .env file in docker-compose

# RUN pip install python-multipart

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "DeepTSF_env", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]