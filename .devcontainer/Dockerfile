ARG TORCH_VERSION=1.9.1-cuda11.1-cudnn8
ARG DEVEL=devel
# ARG DEVEL=runtime

FROM pytorch/pytorch:${TORCH_VERSION}-${DEVEL} as build
RUN conda install -c omgarcia gcc-6
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 280
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub 89
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get update && apt-get autoclean && DEBIAN_FRONTEND=noninteractive TZ=CEST \
    apt-get --yes install git

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

COPY requirements*.txt /workspace/
RUN --mount=type=cache,target=/root/.cache/pip pip wheel --no-deps --wheel-dir /wheels -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip wheel --no-deps --wheel-dir /wheels-devel -r requirements-devel.txt


FROM pytorch/pytorch:${TORCH_VERSION}-runtime as base
COPY known_hosts /root/.ssh/
RUN rm -f /etc/apt/apt.conf.d/docker-clean
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get update && apt-get autoclean && DEBIAN_FRONTEND=noninteractive TZ=CEST \
    apt-get --yes install git libglib2.0-0 curl
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels --mount=type=cache,target=/root/.cache/pip pip install /wheels/*


FROM base as devel
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get update && apt-get autoclean && DEBIAN_FRONTEND=noninteractive TZ=CEST \
    apt-get --yes install libx11-6 libgl1 libglu1 fonts-freefont-ttf sudo fontconfig
RUN --mount=type=bind,from=build,source=/wheels-devel,target=/wheels --mount=type=cache,target=/root/.cache/pip pip install /wheels/*
ENV LOCAL_PYTHON=/opt/conda/bin/python

ARG USER=user
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $USER
RUN useradd -m -u $UID -g $GID -s /bin/bash -d /workspace/home $USER
RUN echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER && chmod 0440 /etc/sudoers.d/$USER
USER $USER

