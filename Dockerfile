FROM nvidia/cuda:10.2-runtime-ubuntu18.04
CMD ["bash"]

RUN mkdir /workspace
WORKDIR /workspace

RUN apt-get update && apt-get install -y python3-dev git wget
RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 80
ENTRYPOINT python ./detectron2_repo/server-gpu.py

LABEL AINIZE_MEMORY_REQUIREMENT=10Gi
