
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04


# # Avoid Public GPG key error
# # https://github.com/NVIDIA/nvidia-docker/issues/1631
# RUN apt-key del 7fa2af80 \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install apt packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget curl\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

# Install miniconda
RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/miniconda/bin:$PATH"

# Update in bashrc
RUN echo "source /opt/miniconda/etc/profile.d/conda.sh" >> /root/.bashrc 

RUN git clone https://github.com/Wei-ucas/InterpAny-Clearer.git /InterpAny-Clearer \
    && source ~/.bashrc \
    && cd /InterpAny-Clearer \
    && conda create -n InterpAny python=3.8 -y\
    && source activate InterpAny \
    && pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 \
    && pip install -r requirements.txt \
    && conda clean -y --all\
    && pip cache purge 


WORKDIR /InterpAny-Clearer
# Download pretrained models
RUN wget https://pjlab-3090-sport.oss-cn-beijing.aliyuncs.com/downloads/InterpAny-Clearer/checkpoints.tar.gz\
    && tar -zxvf checkpoints.tar.gz \
    && rm checkpoints.tar.gz

# Prepare backend environment
RUN cd /InterpAny-Clearer/webapp/backend \
    && cd data/models\
    && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth


# Install webapp dependencies
ARG VERSION="v16.14.2"
ARG DISTRO="linux-x64"
RUN wget -c https://nodejs.org/dist/$VERSION/node-$VERSION-$DISTRO.tar.xz -P ~/Downloads\
    && mkdir -p /usr/local/lib/nodejs \
    && tar -xJvf ~/Downloads/node-$VERSION-$DISTRO.tar.xz -C /usr/local/lib/nodejs \
    && rm -rf ~/Downloads/node-$VERSION-$DISTRO.tar.xz \
    && sed -i 's/^#.*nodejs.*$//gi' ~/.profile \
    && sed -i 's/^export PATH=\/usr\/local\/lib\/nodejs\/node.*$//g' ~/.profile \
    && export PATH=/usr/local/lib/nodejs/node-$VERSION-$DISTRO/bin:$PATH | tee -a ~/.profile \
    && . ~/.profile \
    && ln -sf /usr/local/lib/nodejs/node-$VERSION-$DISTRO/bin/node /usr/bin/node \
    && ln -sf /usr/local/lib/nodejs/node-$VERSION-$DISTRO/bin/npm /usr/bin/npm \
    && ln -sf /usr/local/lib/nodejs/node-$VERSION-$DISTRO/bin/npx /usr/bin/npx \
    && echo "Node.js version: $(node -v)" \
    && echo "NPM version: $(npm -v)" \
    && npm install -g yarn \
    && ln -sf /usr/local/lib/nodejs/node-$VERSION-$DISTRO/bin/yarn /usr/bin/yarn \
    && echo "Yarn version: $(yarn -v)" 
    

RUN echo "conda activate InterpAny" >> ~/.bashrc 

EXPOSE 5001 8080 

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]
CMD [ "serve" ]





    


