FROM rackspacedot/python37:latest

CMD ["bash"]

# Install Node.js 8 and npm 5
RUN apt-get update
RUN apt-get -y install curl gnupg
RUN curl -sL https://deb.nodesource.com/setup_11.x  | bash -
RUN apt-get -y install nodejs
RUN mkdir /workspace
RUN mkdir /workspace/uploads 
WORKDIR /workspace

RUN rm -rf node_modules && npm install

RUN pip3 install torch torchvision opencv-python==3.4.8.29 cython Pillow==6.2.2 scipy
RUN pip install -U 'git+https://github.com/facebookresearch/fvcore'
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install --user -e detectron2_repo

RUN  pip install --user -e detectron2_repo/projects/TensorMask

COPY package.json .
RUN npm install

#denspose model
RUN wget https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/143908701/model_final_dd99d2.pkl -O /workspace/detectron2_repo/densepose_rcnn_R_50_FPN_s1x.pkl
RUN wget http://images.cocodataset.org/val2017/000000439715.jpg -O /workspace/uploads/input.jpg

COPY . .
RUN mv /workspace/demo.py /workspace/detectron2_repo/demo.py
RUN mv /workspace/apply_net.py /workspace/detectron2_repo/apply_net.py
RUN mv /workspace/detectron2_repo/projects/DensePose/densepose/ /workspace/detectron2_repo/densepose    
RUN cp -rl /workspace/detectron2_repo/projects/DensePose/configs/ /workspace/detectron2_repo/
RUN mv /workspace/detectron2_repo/demo/predictor.py /workspace/detectron2_repo/predictor.py



RUN python /workspace/detectron2_repo/demo.py --input /workspace/uploads/input.jpg --config-file /workspace/detectron2_repo/configs/quick_schedules/panoptic_fpn_R_50_inference_acc_test.yaml
RUN python /workspace/detectron2_repo/demo.py --input /workspace/uploads/input.jpg --config-file /workspace/detectron2_repo/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml

RUN rm /workspace/uploads/output.jpg

EXPOSE 80
ENTRYPOINT node server.js
