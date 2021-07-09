FROM python:3.8

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTION
    
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
RUN git clone https://github.com/dheerajakula/Face-Mask-Detection.git
WORKDIR Face-Mask-Detection
RUN pip3 install -r requirements.txt --no-cache-dir
EXPOSE 5000
CMD ["python", "application.py"]
