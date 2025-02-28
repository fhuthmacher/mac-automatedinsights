FROM public.ecr.aws/lambda/python:3.11

# Install build dependencies first
RUN yum install libgomp git gcc gcc-c++ make -y \
 && yum clean all -y && rm -rf /var/cache/yum

ARG TORCH_VER=2.1.2
ARG TORCH_VISION_VER=0.16.2
ARG NUMPY_VER=1.24.3
RUN python3 -m pip --no-cache-dir install --upgrade --trusted-host pypi.org --trusted-host files.pythonhosted.org pip \
 && python3 -m pip --no-cache-dir install --upgrade wheel setuptools \
 && python3 -m pip uninstall -y dataclasses \
 && python3 -m pip --no-cache-dir install --upgrade torch=="${TORCH_VER}" torchvision=="${TORCH_VISION_VER}" -f https://download.pytorch.org/whl/torch_stable.html \
 && python3 -m pip --no-cache-dir install --upgrade numpy==${NUMPY_VER} \
 && python3 -m pip --no-cache-dir install --upgrade pandas \
 && python3 -m pip --no-cache-dir install --upgrade autogluon.tabular[all] \
 && python3 -m pip --no-cache-dir install --upgrade boto3 \
 && python3 -m pip --no-cache-dir install --upgrade pydantic

# Copy function code
WORKDIR /var/task
COPY ../datascientist/bedrock_data_scientist_agent.py .
COPY ../notebooks/utils/ utils/

# Set handler environment variable
ENV _HANDLER="bedrock_data_scientist_agent.lambda_handler"

# Let's go back to using the default entrypoint
ENTRYPOINT [ "/lambda-entrypoint.sh" ]
CMD [ "bedrock_data_scientist_agent.lambda_handler" ]
