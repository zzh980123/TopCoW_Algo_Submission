# Edit the base image here, e.g. to use
# Tensorflow: https://hub.docker.com/r/tensorflow/tensorflow/ 
# Pytorch: https://hub.docker.com/r/pytorch/pytorch/
# For Pytorch e.g.: FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
# Or list your pip packages in requirements.txt and install them
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Create the user
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app
RUN mkdir -p models workdir/mr
#RUN mkdir -p models workdir/mr/bin
ENV PATH="/home/user/.local/bin:${PATH}"

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install --user -U pip

# All required files are copied to the Docker container
COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user base_algorithm.py /opt/app/
COPY --chown=user:user workdir/mr /opt/app/workdir/mr
COPY --chown=user:user models /opt/app/models
# COPY --chown=user:user <somefile> /opt/app/
# ...

# Install required python packages via pip from your requirements.txt
RUN python -m pip install --user -r requirements.txt

# Entrypoint to your python code - executes process.py as a script
ENTRYPOINT [ "python", "-m", "process" ]
