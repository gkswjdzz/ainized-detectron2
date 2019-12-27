FROM rackspacedot/python37:latest

CMD ["bash"]

# Install Node.js 8 and npm 5
RUN apt-get update
RUN apt-get -y install curl gnupg
RUN curl -sL https://deb.nodesource.com/setup_11.x  | bash -
RUN apt-get -y install nodejs
RUN mkdir /workspace 
WORKDIR /workspace

RUN rm -rf node_modules && npm install

RUN pip3 install torch torchvision opencv-python==3.4.8.29 cython
RUN pip install -U 'git+https://github.com/facebookresearch/fvcore'
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install --user -e detectron2_repo

COPY package.json .
RUN npm install

COPY . .
#RUN mv /workspace/pre-trained-model.pkl /workspace/detectron2_repo/pre-trained-model.pkl
RUN mv /workspace/demo.py /workspace/detectron2_repo/demo.py

RUN wget http://images.cocodataset.org/val2017/000000439715.jpg -O /workspace/uploads/input.jpg
RUN python /workspace/detectron2_repo/demo.py /workspace/uploads/input.jpg /workspace/sample.jpg

RUN rm /workspace/sample.jpg

EXPOSE 80
ENTRYPOINT node server.js
