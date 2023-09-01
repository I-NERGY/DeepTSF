# Use the official Miniconda base image
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Copy environment setup file
COPY conda.yaml  .

# Create the conda environment from the env conda.yaml file
RUN conda env create -f conda.yaml

# Activate the conda environment
SHELL ["conda", "run", "-n", "DeepTSF_env", "/bin/bash", "-c"]

# Install additional dependencies
RUN pip install dagster dagster-webserver dagster-shell
RUN pip install protobuf==3.20.*
RUN pip install minio

# Copy your MLflow project files to the container
COPY . .

EXPOSE 3000

WORKDIR /app/dagster
# Set the DAGSTER_HOME environment variable
ENV DAGSTER_HOME /app/dagster

## Set the entrypoint command to execute the desired command
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "DeepTSF_env", "dagster", "dev", "-h", "0.0.0.0", "-p", "3000"]
#"-f", "sample.py" ]