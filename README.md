[![Run on Ainize](https://ainize.ai/static/images/run_on_ainize_button.svg)](https://ainize.web.app/redirect?git_repo=github.com/gkswjdzz/ainized-detectron2)
# Ainized-Detectron2

[Detectron2](https://github.com/facebookresearch/detectron2) is the object detection open source project based on the pytorch made in the Facebook AI Research (FAIR). With modular design, Detectron2 is more flexible, extensible than the existing Detectron. Detectron2 provides models of object detection such as panoptic segmentation, DensePose, Cascade RCNN, and more based on a variety of backbones.

In this Ainize project, you can receive the inferred result image after selecting one of the inference models. All the inference models used Resnet 50 + FPN (Feature Pyramid Network) as a backbone.

The inference using server is done in the following steps:
1. User publishes an image file
2. server returns a inferred image or json which is information of detected objects.

Note that the server is implemented in Node.js.

You can see the demo server from below site

https://ainize.ai/deployments/github.com/gkswjdzz/ainized-detectron2

# How to deploy

this server is dockerized, so it can be built and run using docker commands.

## Docker build

```
docker build -t detectron2 -f Dockerfile-cpu .
```
or
```
docker build -t detectron2 -f Dockerfile-gpu .
```

## Run Docker

```
docker run -p 80:80 -it detectron2
```

Now the server is available at http://localhost.

Note that the docker image can be deployed using any docker-based deploy platform (e.g. [ainize.ai](https://ainize.ai)).

# References
1. [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
