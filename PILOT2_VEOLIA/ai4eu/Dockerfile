FROM python:3.9-slim

RUN pip install --upgrade pip

RUN adduser app
USER app
WORKDIR /home/app

COPY --chown=app:app requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY --chown=app:app . .

RUN python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. load_prediction.proto

EXPOSE 8061

# port 8062 should be exposed when UI is available for service listening at 8061
# EXPOSE 8062

ENTRYPOINT python3 ./predict_load_server.py