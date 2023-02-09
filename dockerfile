FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
# set the port
ENV PORT 5000
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
EXPOSE 5000

# set the working directory
WORKDIR /app
COPY ./.dockerignore /src/.dockerignore
COPY ./environment.yml /src/environment.yml
WORKDIR /src
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NOWARNINGS="yes"

# install conda
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils wget bzip2 ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_22.11.1-1-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda clean -tip \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && /opt/conda/bin/conda env create -f environment.yml \
    && echo "source activate brain-tumor" > ~/.bashrc

# RUN /opt/conda/bin/conda install -c "nvidia/label/cuda-11.7.1" cuda -y 
RUN /opt/conda/bin/conda install -c anaconda cudnn -y 

ENV PATH /opt/conda/envs/brain-tumor/bin:$PATH
ENV LD_LIBRARY_PATH /opt/conda/lib:$LD_LIBRARY_PATH
WORKDIR /app
# run the app
# to shell: /opt/conda/bin/conda run -n brain-tumor python app.py
CMD ["/opt/conda/bin/conda","run","-n","brain-tumor","python","/app/app.py"]

# docker build -t wurining/unet .
# docker build -t coreharbor.wurining.com/leeds/unet:v1.1 .
# docker push coreharbor.wurining.com/leeds/unet:v1.1
