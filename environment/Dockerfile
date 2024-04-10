# hash:sha256:ba58ee506b263fa92b988eebeb5da03e65608dc4e53ce2f29d354915acdc6999
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        cartopy==0.22.0 \
        jupyter==1.0.0 \
        jupyterlab==3.5.3 \
        matplotlib==3.8.0 \
        numpy==1.26.4 \
        pandas==2.2.1 \
        python==3.11.0 \
        scipy==1.11.4 \
        seaborn==0.12.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    xlrd==2.0.1
