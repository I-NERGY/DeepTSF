# miniconda base image
FROM continuumio/miniconda3:23.5.2-0

# Set the working directory inside the container
WORKDIR /app

# Copy environment setup file
COPY ../../conda.yaml .

# Create the conda environment from the env conda.yaml file
RUN conda env create -f conda.yaml

# Activate the conda environment
SHELL ["conda", "run", "-n", "DeepTSF_env", "/bin/bash", "-c"]

# Install additional dependencies
RUN pip install dagster dagster-webserver dagster-shell
RUN pip install protobuf==3.20.*
RUN pip install minio

COPY . .

EXPOSE 8006

WORKDIR /app/dagster-deeptsf
# Set the DAGSTER_HOME environment variable
ENV DAGSTER_HOME /app/dagster-deeptsf

ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "DeepTSF_env", "dagster", "dev", "-h", "0.0.0.0", "-p", "8006"]
