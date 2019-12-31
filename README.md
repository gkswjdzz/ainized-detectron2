[![Run on Ainize](https://ainize.ai/static/images/run_on_ainize_button.svg)](https://ainize.web.app/redirect?git_repo=github.com/gkswjdzz/ainize-run-detectron2)
# Ainize-run-detectron2

This repository provides a server that infers instance segmentation for an image based on a Mast R-CNN R-50-FPN model. The model used in the server is from [Facebookresearch/detectron2](https://github.com/facebookresearch/detectron2), which is an implementation of FAIR(Facebook AI Research) paper "Mask R-CNN". 

The inference using server is done in the following steps:
1. User publishes an image file
2. server returns a instance segmentation.

Note that the server is implemented in Node.js.

# How to deploy

this server is dockerized, so it can be built and run using docker commands.

## Docker build
```
docker build -t detectron2 .
```

## Docker run
```
docker run -p 80:80 -it detectron2
```
<!--
### Upload image

<img src="/images/image1.png" width="250" />
<img src="/images/image2.png" width="250" />
-->

Now the server is available at http://localhost.

Note that the docker image can be deployed using any docker-based deploy platform (e.g. [ainize.ai](https://ainize.ai)).


# How to publish an image file

The image to be evaluated needs to be published first. You can refer to the two following examples of how to publish image files: 

Upload your image and submit image.

<img src="/images/image1.png" width="250" />  
<img src="/images/image2.png" width="250" />

The result is like this

<img src="/images/image3.png" width="250" />

# References
1. [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
