# Multi-Agent Collaboration Evaluation

This repository is part of a blog series and contains sample code to demonstrate how to evaluate a multi-agent system with assertion based benchmarking using MLflow and Bedrock Agents.

## Architecture
![Architecture](./architecture.png)

The current implementation of the multi-agent system itself is just a proof of concept and is not production ready. It is designed as a starting point for the evaluation process.

## Multi-Agent System Details
This multi-agent system enables a user to extend a typical ML pipeline to ML use case identification and data collection.

![Sample Workflow](./sampleflow.png)

A user can simply upload a new dataset or point to an existing Athena database to start the ML process as shown below.
![Multi-Agent Prototype UI](./frontend.gif)

## Multi-Agent Evaluation Process and Results
![Overview of Evaluation Process](./mac_eval_process.png)

And the evaluation results are tracked with MLflow as shown below.
![Evaluation Results](./mlflowResults.gif)


## Environment
1. Ensure you've enabled Claude Sonnet 3.5 v1 and any other models that you want to use  in the Bedrock Console
2. Ensure you have adequate permissions to call Bedrock from the Python SDK (Boto3), S3, IAM, and Lambda.
3. Ensure you have Docker installed in your environment.

### Local
These notebooks were tested with Python 3.11. 
If you're running locally, ensure that you have the AWS CLI setup with the credentials you want to use.
These credentials need access to Amazon Bedrock Models, S3, IAM, and Lambda.

## Table of Contents

- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Authors](#authors)
- [License](#license)
- [Resources](#resources)

## Folder Structure

```
mac-automatedinsights/
├── data/                  # Sample datasets for multi-agent collaboration
├── notebooks/             # Notebooks for creating Agents and evaluating them
|__ dataengineer/          # Code for Data Engineer Agent
|__ businessanalyst/       # Code for Business Analyst Agent
|__ datascientist/         # Code for Data Scientist Agent
I__ supervisor/            # Code for the Supervisor Agent
|__ automated-insights/    # CDK code for frontend and backend of the automated insights prototype

```

### Details
The multi-agent collaboration system implements 4 agents, and is designed to be modular and can be extended to include new agents or new components. 

- `data/`: Contains the datasets used for the multi-agent collaboration evaluation.
- `notebooks/`: Jupyter notebooks demonstrating the creation and evaluation of:
  - The Data Engineer Agent
  - The Business Analyst Agent
  - The Data Scientist Agent
  - The Supervisor Agent

## Getting Started

1. Clone the repository:
   ```
   git clone git@github.com:fhuthmacher/mac-automatedinsights.git
   cd mac-automatedinsights
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   $ cd notebooks
   pip install -r notebooks/requirements.txt
   ```

4. Go to notebook & start jupyter notebooks!
   ``` bash
   
   $ jupyter notebook
   ```

5. Go to notebook folder and update dev.env to set respective environment variables:
   ``` bash
   $ cp notebooks/dev.env.example notebooks/dev.env
   ```
   Update the dev.env file with the appropriate values for your environment.


6. Start at notebook 1 and work your way through them!

7. Once you're done with the notebooks, you can deploy the CDK stack to create a sample application (Frontend and Backend).
   Follow the instructions in the respective `README.md` files in the `automated-insights/` sub-folders.

## Usage

Explore the example notebooks in the `notebooks/` directory to understand how you can configure and evaluate Bedrock Agents and how you can ultimately also evaluate your multi-agent collaboration system.

## Authors

- Felix Huthmacher  - *Initial work* - [github](https://github.com/fhuthmacher)


## License

This library is licensed under the MIT-0 License. See the LICENSE file.

## Resources
This repository was inspired by the excellent paper: 
[Towards Effective GenAI Multi-Agent Collaboration: Design and Evaluation for Enterprise Applications by Raphael Shu, Nilaksh Das, Michelle Yuan, Monica Sunkara, and Yi Zhang](https://arxiv.org/abs/2412.05449) 