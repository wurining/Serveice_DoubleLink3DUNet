# Version 1
# FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
# # set the port
# ENV PORT 5000
# EXPOSE 5000

# # set the working directory
# WORKDIR /src
# # COPY ./.dockerignore /src/.dockerignore
# # COPY ./environment.yml /src/environment.yml
# COPY . /src

# ENV DEBIAN_FRONTEND noninteractive
# ENV DEBCONF_NOWARNINGS="yes"

# # install conda
# RUN apt-get update && apt-get install -y --no-install-recommends apt-utils wget bzip2 ca-certificates zip unzip \
#     && apt-get clean && rm -rf /var/lib/apt/lists/* \
#     && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_22.11.1-1-Linux-x86_64.sh -O ~/miniconda.sh \
#     && /bin/bash ~/miniconda.sh -b -p /opt/conda \
#     && rm ~/miniconda.sh \
#     && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
#     && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# # install package and clean
# RUN /opt/conda/bin/conda env create -f environment.yml \
#     && /opt/conda/bin/conda install -c conda-forge cudnn -y \
#     && echo "source activate brain-tumor" > ~/.bashrc \
#     && /opt/conda/bin/conda clean -tipy \
#     && rm -rf /opt/conda/pkgs/* 
# # RUN /opt/conda/bin/conda install -c "nvidia/label/cuda-11.7.1" cuda -y 
# # RUN /opt/conda/bin/conda install -c conda-forge cudnn -y 

# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV PATH /opt/conda/envs/brain-tumor/bin:$PATH
# ENV LD_LIBRARY_PATH /opt/conda/lib:$LD_LIBRARY_PATH

# RUN unzip model/model.zip -d model/ && rm model/model.zip
# # WORKDIR /app
# # run the app
# # to shell: /opt/conda/bin/conda run -n brain-tumor python app.py
# CMD ["/opt/conda/bin/conda","run","-n","brain-tumor","python","/src/app.py"]

# # docker build -t wurining/unet .
# # docker build -t coreharbor.wurining.com/leeds/unet:v1.1 .
# # docker push coreharbor.wurining.com/leeds/unet:v1.1

FROM tensorflow/tensorflow:2.5.1-gpu
# set the port
ENV PORT 5000
EXPOSE 5000

# set the working directory
WORKDIR /src
# COPY ./.dockerignore /src/.dockerignore
# COPY ./environment.yml /src/environment.yml
COPY . /src

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NOWARNINGS="yes"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV PATH /opt/conda/envs/bin:$PATH
ENV LD_LIBRARY_PATH /opt/conda/lib:$LD_LIBRARY_PATH

RUN /usr/bin/python3 -m pip install --upgrade pip && pip install flask flask_restful tensorflow-addons==0.14.0

# run the app
# to shell: python app.py
CMD ["python","/src/app.py"]

# docker build -t coreharbor.wurining.com/leeds/unet:v2.0 . 
# docker push coreharbor.wurining.com/leeds/unet:v2.0

# docker run -p 127.0.0.1:8679:5000 \
# --mount type=bind,source=/home/serverleeds/Documents/projects/Service_DoubleLink3DUNet/tmp,target=/src/tmp \
# -it coreharbor.wurining.com/leeds/unet:v2.0

