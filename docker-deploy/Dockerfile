FROM ubuntu:22.04

LABEL maintainer="Purusharth Saxena <purusharth.saxena@stud.uni-heidelberg.de>"

# Create user
RUN useradd -ms /bin/bash scp_infer_test

# Set timezone & install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
    tzdata wget bzip2 libx11-6 build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda with Python 3.11
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda clean -afy

# ENV PATH="/home/scp_infer_test/.local/bin:${PATH}"
WORKDIR /home/scp_infer_test/inference-pkg
# RUN echo 'export PATH=$HOME/.local/bin:$PATH' >> /home/scp_infer_test/.bashrc



# Install Python dependencies
# RUN pwd
# RUN ls
# RUN cd inference-pkg
# WORKDIR /inference-pkg
COPY test /home/scp_infer_test/
RUN ls
RUN python -m pip install -r requirements.txt

# CMD ["/bin/bash"]
# Switch to user and set working directory
USER scp_infer_test

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Define environment variable
ENV NAME World

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
