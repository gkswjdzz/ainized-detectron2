FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y gcc libglib2.0-0
RUN python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html

RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 80
ENTRYPOINT streamlit run client_streamlit.py --server.enableXsrfProtection=false --server.enableCORS=true
