FROM public.ecr.aws/lambda/python:3.11

# Install build dependencies first
RUN yum install libgomp git gcc gcc-c++ make -y \
 && yum clean all -y && rm -rf /var/cache/yum


RUN python3 -m pip --no-cache-dir install --upgrade --trusted-host pypi.org --trusted-host files.pythonhosted.org pip \
 && python3 -m pip --no-cache-dir install --upgrade wheel setuptools \
 && python3 -m pip --no-cache-dir install --upgrade pandas \
 && python3 -m pip --no-cache-dir install --upgrade boto3 \
 && python3 -m pip --no-cache-dir install --upgrade opensearch-py \
 && python3 -m pip --no-cache-dir install --upgrade Pillow \
 && python3 -m pip --no-cache-dir install --upgrade pyarrow \
 && python3 -m pip --no-cache-dir install --upgrade fastparquet \
 && python3 -m pip --no-cache-dir install --upgrade urllib3 \
 && python3 -m pip --no-cache-dir install --upgrade pydantic

# Copy function code
WORKDIR /var/task
COPY ../dataengineer/bedrock_data_engineer_agent.py .
COPY ../notebooks/utils/ utils/

# Set handler environment variable
ENV _HANDLER="bedrock_data_engineer_agent.lambda_handler"

# Let's go back to using the default entrypoint
ENTRYPOINT [ "/lambda-entrypoint.sh" ]
CMD [ "bedrock_data_engineer_agent.lambda_handler" ]
