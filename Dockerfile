FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN chsh -s /bin/bash

RUN apt-get update && apt-get install -y curl vim libgfortran3 git
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev

# Install Anaconda.
#RUN cd /tmp && \
#    curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh && \
#    bash Anaconda3-2020.02-Linux-x86_64.sh -b
#ENV PATH="/root/anaconda3/bin:/root/anaconda3/condabin:${PATH}"
##RUN source ~/anaconda3/etc/profile.d/conda.sh

# Install Miniconda.
RUN cd /tmp && \
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="/root/miniconda3/bin:/root/miniconda3/condabin:${PATH}"
#RUN source ~/miniconda3/etc/profile.d/conda.sh

COPY conda-gpu.yml .
COPY setup.py .

RUN conda env create -f conda-gpu.yml

RUN mkdir /projects

# Install OpenVINO
COPY openvino/l_openvino_toolkit_p_2020.3.194 /openvino
RUN cd /openvino && \
    ./install_openvino_dependencies.sh
RUN cd /openvino && \
    ./install.sh --silent silent.cfg
RUN apt-get update && apt-get upgrade -y
ENV INSTALLDIR /opt/intel/openvino

#RUN conda activate yolov3-tf2-gpu

#RUN nvcc --version

#RUN nvidia-smi

ENV SHELL="/bin/bash"

CMD /bin/bash

#docker build -t yolov3-tf2 .
#docker run --name yolov3-tf2-default --gpus all -it -v $(pwd):/projects yolov3-tf2 /bin/bash
#docker stop yolov3-tf2-default
#docker start yolov3-tf2-default
#docker exec -it yolov3-tf2-default bash
#docker rm yolov3-tf2-default
#docker system prune -af

#python train.py --dataset ./data/pedestrian-traffic-lights.v2-version-1.tfrecord/train/pedestrian-traffic-lights.tfrecord --val_dataset ./data/pedestrian-traffic-lights.v2-version-1.tfrecord/valid/pedestrian-traffic-lights.tfrecord --classes ./data/_classes.txt --num_classes 6 --mode fit --transfer none --batch_size 20 --epochs 200 --tiny --repetitions 1

#python detect.py --classes ./data/_classes.txt --num_classes 6 --weights ./checkpoints/yolov3_train_0.tf --image ./data/sample.jpg --tiny

#python detect_video.py --classes ./data/_classes.txt --num_classes 6 --weights ./checkpoints/yolov3_train_0.tf --video ./data/bcn_test_vids/VID_20200615_191815_257.mp4 --tiny

#python convert.py --weights ./best_checkpoints/yolov3_train_0.tf --output yolov3-tiny --tiny --num_classes 6 --save_model

#python detect_tflite.py --output ./output.mp4 --video ./data/sample2.mp4 --model ./model/best.onnx

#python -m tf2onnx.convert --opset 11 --saved-model ./model/tf_yolov3-tiny --output ./model/model.onnx

#python utils/tf-to-yolo.py --train_records data/pedestrian-traffic-lights.v2-version-1.tfrecord/train/pedestrian-traffic-lights.tfrecord --val_records data/pedestrian-traffic-lights.v2-version-1.tfrecord/valid/pedestrian-traffic-lights.tfrecord --label_map data/pedestrian-traffic-lights.v2-version-1.tfrecord/train/pedestrian-traffic-lights_label_map.pbtxt --obj_names data/pigall.names --source_file cfg/yolov4.cfg --destination_file data/train
