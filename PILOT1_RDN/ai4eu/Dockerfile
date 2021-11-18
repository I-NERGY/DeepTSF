# base image
FROM python:3.9-slim

RUN pip3 install --upgrade pip
# switch to non-root user
RUN adduser app
USER app
WORKDIR /home/app

# copy requirements.txt containing the packages for installation
COPY --chown=app:app requirements.txt requirements.txt
# install packages
RUN pip3 install -r requirements.txt
# Copy all files inside current directory and according to dockerignore
COPY --chown=app:app . .
# compile the prote to generate classes
RUN python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. load_prediction.proto
# requirement for AI4EU, expose the service port 8061
EXPOSE 8061

# port 8062 should be exposed when UI is available for service listening at 8061
# EXPOSE 8062

# start server
ENTRYPOINT python3 ./predict_load_server.py
