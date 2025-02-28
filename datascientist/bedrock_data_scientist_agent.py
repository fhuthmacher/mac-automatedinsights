import json
import sys
import logging
from autogluon.tabular import TabularPredictor
import pandas as pd
import boto3
import os
import zipfile
from urllib.parse import urlparse
import base64
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union, Tuple
import time
import botocore
import traceback
import tempfile

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3_client = boto3.client('s3')

# get the environment variables
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
MODEL_ID = os.getenv('MODEL_ID')

ATHENA_QUERY_EXECUTION_LOCATION = f's3://{S3_BUCKET_NAME}/athena_results/'

class Parameters(BaseModel):
    target: Optional[str] = None
    model_location: Optional[str] = None
    data_location: Optional[str] = None
    train_data_location: Optional[str] = None
    test_data_location: Optional[str] = None
    result_data_location: Optional[str] = None
    hyperparameters: Optional[Dict[str, Dict[str, int]]] = None
    holdout_frac: Optional[float] = None
    api_path: str

    @validator('target')
    def validate_target(cls, v, values):
        # Only validate target if the API path requires it
        if values.get('api_path') in ['Train', 'Predict'] and not v:
            raise ValueError("Target is required for TrainModel and Predict operations")
        return v
    
    @validator('model_location')
    def validate_model_location(cls, v, values):
        # Only validate model_location if the API path requires it
        if values.get('api_path') in ['Train', 'Predict'] and not v:
            raise ValueError("Model location is required for TrainModel and Predict operations")
        return v

class APIResponse(BaseModel):
    message: str
    results: Dict[str, Any]

def normalize_s3_path(path: str, default_bucket: str) -> str:
    """Normalize S3 path to full path with bucket name"""
    if not path:
        return None
        
    # If path already starts with s3://, return as is
    if path.startswith('s3://'):
        return path
        
    # Remove any leading slashes
    clean_path = path.lstrip('/')
    
    # Construct full S3 path
    return f's3://{default_bucket}/{clean_path}'

def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """Parse S3 path into bucket and key"""
    if not s3_path:
        return None, None
        
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key

def parse_parameters(event: Dict[str, Any]) -> Parameters:
    parameters = {}
    
    # Extract APIPath from event
    api_path = event.get('apiPath', '').strip('/')
    
    # Try to extract parameters from inputText if it contains JSON
    if 'inputText' in event:
        try:
            json_start = event['inputText'].find('{')
            if json_start != -1:
                json_str = event['inputText'][json_start:]
                input_json = json.loads(json_str)
                if 'parameters' in input_json:
                    for param in input_json['parameters']:
                        parameters[param['name']] = param['value']
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from inputText")

    # Extract parameters from requestBody if present
    if 'requestBody' in event and 'content' in event['requestBody']:
        content = event['requestBody']['content']
        if 'application/json' in content and 'properties' in content['application/json']:
            for prop in content['application/json']['properties']:
                parameters[prop['name']] = prop['value']

    return Parameters(
        api_path=api_path,
        target=parameters.get('Target'),
        model_location=parameters.get('ModelLocation'),
        data_location=parameters.get('DataLocation'),
        train_data_location=parameters.get('TrainDataLocation'),
        test_data_location=parameters.get('TestDataLocation'),
        result_data_location=parameters.get('ResultDataLocation'),
        hyperparameters=parameters.get('Hyperparameters'),
        holdout_frac=parameters.get('HoldoutFrac')
    )

def train(train_df, test_df, target, model_location, hyperparameters) -> APIResponse:
    logger.info("Training model...")

    # Create a writable directory for AutoGluon
    temp_dir = "/tmp/model"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set the current working directory to the temp directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        if hyperparameters is None:
            hyperparameters = {
                'GBM': {'num_boost_round': 10000},
                'RF': {'n_estimators': 300},
                'XT': {'n_estimators': 300},
            }
        
        num_trials = 3  # try at most 3 different hyperparameter configurations for each type of model
        search_strategy = 'auto'  # to tune hyperparameters using random search routine with a local scheduler

        hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
            'num_trials': num_trials,
            'scheduler' : 'local',
            'searcher': search_strategy,
        }
        
        # Train the model
        time_limit = 3*60  # train various models for ~3 min

        predictor = TabularPredictor(label=target).fit(train_df, presets='medium', time_limit=time_limit, hyperparameters=hyperparameters, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs)

        logger.info("Model trained successfully")

        # Create absolute path and ensure the model directory exists
        model_path = os.path.abspath("/tmp/model")
        os.makedirs(model_path, exist_ok=True)
        logger.info(f"Created model directory at: {model_path}")

        # Save the model with absolute path
        predictor.save(model_path)
        logger.info(f"Model saved to: {model_path}")

        # Compile the model
        predictor.compile()
        logger.info("Model compiled successfully")

        # Get model metrics and convert to serializable format
        model_accuracy = predictor.evaluate(test_df)
        model_names = predictor.model_names()
        
        # Convert feature importance DataFrame to dict
        feature_importance = predictor.feature_importance(train_df)
        feature_importance_dict = feature_importance.to_dict() if isinstance(feature_importance, pd.DataFrame) else {}
        
        # Get leaderboard and convert to dict
        leaderboard = predictor.leaderboard(test_df, extra_info=True)
        leaderboard_dict = leaderboard.to_dict() if isinstance(leaderboard, pd.DataFrame) else {}

        # Verify the files were saved
        saved_files = os.listdir(model_path)
        logger.info(f"Files in model directory: {saved_files}")

        # zip contents of /tmp/model/   
        with zipfile.ZipFile("/tmp/model.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk("/tmp/model/"):
                for file in files:
                    if file != "model.zip":  # Skip the zip file itself
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, "/tmp/model/")
                        zipf.write(file_path, arcname)

        # Parse S3 location
        s3_url = urlparse(model_location)
        bucket = s3_url.netloc
        key = s3_url.path.lstrip('/')  # Remove leading slash

        # Upload model.zip to s3
        s3_client.upload_file(
            "/tmp/model.zip",
            Bucket=bucket,
            Key=key
        )

        return APIResponse(
            message="Model trained successfully",
            results={
                'model_location': model_location,
                'model_names': model_names,
                'model_accuracy': {k: float(v) if isinstance(v, (int, float)) else v 
                                 for k, v in model_accuracy.items()},
                'feature_importance': feature_importance_dict,
                'leaderboard': leaderboard_dict
            }
        )

    except Exception as e:
        logger.error(f"Error in train function: {str(e)}", exc_info=True)
        raise
    finally:
        # Restore original working directory
        os.chdir(original_dir)

def predict(df, model_location, result_data_location) -> APIResponse:
    logger.info("Predicting...")
    # Load the model
    logger.info("Loading model...")
    
    try:
        # Ensure the model directory exists
        os.makedirs("/tmp/model", exist_ok=True)
        
        # download model.zip from s3
        s3_url = urlparse(model_location)
        bucket = s3_url.netloc
        key = s3_url.path.lstrip('/')  # Remove leading slash
        s3_client.download_file(
            Bucket=bucket,
            Key=key,
            Filename="/tmp/model/model.zip"
        )

        # Clear the model directory first
        for root, dirs, files in os.walk("/tmp/model", topdown=False):
            for name in files:
                if name != "model.zip":  # Don't delete the zip we just downloaded
                    try:
                        os.remove(os.path.join(root, name))
                    except OSError as e:
                        logger.warning(f"Error removing file {name}: {e}")
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except OSError as e:
                    logger.warning(f"Error removing directory {name}: {e}")

        # unzip model.zip
        with zipfile.ZipFile("/tmp/model/model.zip", "r") as zip_ref:
            zip_ref.extractall("/tmp/model")
        
        # Find the AutogluonModels directory
        model_dir = "/tmp/model"
        for root, dirs, files in os.walk(model_dir):
            if "predictor.pkl" in files:
                model_dir = root
                break
        
        # Load the model from the correct path
        predictor = TabularPredictor.load(model_dir)
        logger.info(f"Model loaded successfully from {model_dir}")

        # Make prediction
        logger.info("Making prediction...")
        prediction = predictor.predict(df)
        logger.info(f"Prediction made: {prediction}")

        # join the predictions with the original dataframe by adding a new column 'prediction'
        df['prediction'] = prediction
        logger.info(f"Joined predictions with original dataframe: {df}")

        # save the predictions to a new csv file
        df.to_csv("/tmp/predictions.csv", index=False)
        logger.info(f"Predictions saved to /tmp/predictions.csv")

        # upload the predictions to s3
        results_s3_url = urlparse(result_data_location)
        results_bucket = results_s3_url.netloc
        results_key = results_s3_url.path.lstrip('/')  # Remove leading slash
        s3_client.upload_file(
            "/tmp/predictions.csv",
            Bucket=results_bucket,
            Key=results_key
        )
        logger.info(f"Predictions uploaded to s3: {result_data_location}")

        sample_predictions = prediction[:10].tolist() if len(prediction) > 10 else prediction.tolist()
        
        return APIResponse(
            message="Prediction made successfully",
            results={
                'prediction_results_location': result_data_location,
                'predictions': sample_predictions,
                'total_predictions': len(prediction),
                'model_info': str(predictor.model_names())
            }
        )
    
    except Exception as e:
        logger.error(f"Error in predict function: {str(e)}")
        raise

def feature_importance(df: pd.DataFrame, model_location: str) -> APIResponse:
    """Calculate feature importance for a trained model"""
    logger.info("Calculating feature importance...")
    
    if df is None:
        raise ValueError("Data is required to calculate feature importance")
    
    try:
        # Ensure the model directory exists
        os.makedirs("/tmp/model", exist_ok=True)
        
        # download model.zip from s3
        s3_url = urlparse(model_location)
        bucket = s3_url.netloc
        key = s3_url.path.lstrip('/')  # Remove leading slash
        
        try:
            s3_client.download_file(
                Bucket=bucket,
                Key=key,
                Filename="/tmp/model/model.zip"
            )
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise ValueError(f"Model not found at location: {model_location}")
            raise

        # unzip model.zip
        with zipfile.ZipFile("/tmp/model/model.zip", "r") as zip_ref:
            zip_ref.extractall("/tmp/model")
        
        # Find the AutogluonModels directory
        model_dir = "/tmp/model"
        for root, dirs, files in os.walk(model_dir):
            if "predictor.pkl" in files:
                model_dir = root
                break
        
        # Load the model from the correct path
        predictor = TabularPredictor.load(model_dir)
        logger.info(f"Model loaded successfully from {model_dir}")
        
        # Calculate feature importance
        feature_importance = predictor.feature_importance(df)
        logger.info(f"Feature importance calculated successfully")
        feature_importance_dict = feature_importance.to_dict() if isinstance(feature_importance, pd.DataFrame) else {}
        
            
        return APIResponse(
            message="Feature importance calculated successfully",
            results={
                'feature_importance': feature_importance_dict
            }
        )
        
    except Exception as e:
        logger.error(f"Error calculating feature importance: {str(e)}", exc_info=True)
        raise

def save_dataset(self, df: pd.DataFrame, use_case_name: str) -> str:
    """Save a dataset to S3 and return its location"""
    logger.info(f"Saving dataset for use case: {use_case_name}")
    logger.debug(f"Dataset shape: {df.shape}")
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            df.to_csv(tmp.name, index=False)
            s3_path = f'ml_datasets/{use_case_name}_dataset.csv'
            
            logger.info(f"Uploading dataset to s3://{self.s3_bucket_name}/{s3_path}")
            self.s3_client.upload_file(
                tmp.name,
                self.s3_bucket_name,
                s3_path
            )
            
            location = f's3://{self.s3_bucket_name}/{s3_path}'
            logger.info(f"Dataset successfully saved to {location}")
            return location
            
    except Exception as e:
        logger.error(f"Error in save_dataset: {str(e)}", exc_info=True)
        return ''
        
def load_data(data_location: str) -> pd.DataFrame:
    """Load data from S3 with proper error handling"""
    if not data_location:
        raise ValueError("DataLocation is required")
        
    logger.info(f"Loading data from {data_location}")
    
    try:
        # Ensure data directory exists
        os.makedirs("/tmp/data", exist_ok=True)
        
        # Parse and validate data location
        data_bucket, data_key = parse_s3_path(data_location)
        if not data_bucket or not data_key:
            raise ValueError(f"Invalid data location format: {data_location}")
            
        # Get filename and local path
        filename = os.path.basename(data_key)
        local_file_path = os.path.join("/tmp/data", filename)
        
        logger.info(f"Downloading from bucket: {data_bucket}, key: {data_key}")
        
        try:
            # Check if file exists
            s3_client.head_object(Bucket=data_bucket, Key=data_key)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise ValueError(f"Data file not found: {data_location}")
            else:
                raise
                
        # Download file
        s3_client.download_file(Bucket=data_bucket, Key=data_key, Filename=local_file_path)
        logger.info(f"Downloaded file to: {local_file_path}")
        
        # Handle zip files
        if data_location.endswith(".zip"):
            with zipfile.ZipFile(local_file_path, "r") as zip_ref:
                zip_ref.extractall("/tmp/data")
                
        # Load the specific file we just downloaded
        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(local_file_path)
            elif ext == '.json':
                df = pd.read_json(local_file_path)
            elif ext == '.parquet':
                df = pd.read_parquet(local_file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
            if df is None or df.empty:
                raise ValueError(f"No data found in file: {filename}")
                
            logger.info(f"Successfully loaded data from {filename}")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load data from {filename}: {str(e)}")
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to load data: {str(e)}")

def validate_data_location(data_location: str):
    """Validate that data_location is a valid S3 path and the file exists"""
    if not data_location:
        raise ValueError("DataLocation is required")
    
    bucket, key = parse_s3_path(data_location)
    if not bucket or not key:
        raise ValueError(f"Invalid data location format: {data_location}")
    
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            raise ValueError(f"Data file not found at location: {data_location}")
        raise

def validate_model_location(model_location: str):
    """Validate that model_location is a valid S3 path, ends with model.zip, and is in the 'models' subdirectory"""
    if not model_location:
        raise ValueError("ModelLocation is required")
    
    bucket, key = parse_s3_path(model_location)
    if not bucket or not key:
        raise ValueError(f"Invalid model location format: {model_location}")
    
    if not key.endswith('model.zip'):
        raise ValueError("ModelLocation must end with 'model.zip'")
    
    if 'models/' not in key:
        raise ValueError("ModelLocation must be in the 'models' subdirectory")

def validate_result_data_location(result_data_location: str):
    """Validate that result_data_location ends with a file extension and is in the 'results' subdirectory"""
    if not result_data_location:
        raise ValueError("ResultDataLocation is required")
    
    bucket, key = parse_s3_path(result_data_location)
    if not bucket or not key:
        raise ValueError(f"Invalid result data location format: {result_data_location}")
    
    # Check if the key has a file extension
    if '.' not in os.path.basename(key):
        raise ValueError("ResultDataLocation must end with a file extension")
    
    # Check if the key is in the 'results' subdirectory
    if 'results/' not in key:
        raise ValueError("ResultDataLocation must be in the 'results' subdirectory")

def validate_model_exists(model_location: str):
    """Validate that model exists in S3"""
    try:
        bucket, key = parse_s3_path(model_location)
        s3_client.head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            raise ValueError(f"Model not found at location: {model_location}")
        raise

def validate_target_column(df: pd.DataFrame, target: str):
    """Validate target column exists in dataframe"""
    if target not in df.columns:
        available_columns = list(df.columns)
        raise ValueError(
            f"Target column '{target}' not found in data. "
            f"Available columns: {available_columns}"
        )

def train_test_split_dataset(df: pd.DataFrame, holdout_frac: float, data_location: str, target: str = None) -> APIResponse:
    """
    Splits a dataset into training and testing sets and saves them to S3.
    Ensures balanced distribution of target classes for classification or representative 
    distribution of target values for regression.
    
    Args:
        df: Pandas DataFrame to split
        holdout_frac: Fraction of data to use for testing (between 0 and 1)
        data_location: Original S3 location of the data
        target: The target column to predict (optional)
    Returns:
        APIResponse: Object containing split results
    """
    logger.info(f"Splitting dataset with holdout fraction: {holdout_frac}")
    
    try:
        # Validate holdout fraction
        if not 0 < holdout_frac < 1:
            raise ValueError(f"Holdout fraction must be between 0 and 1, got {holdout_frac}")
        
        # If target column is provided, determine if it's classification or regression
        if target and target in df.columns:
            # Check data type and unique values to determine problem type
            is_numeric = pd.api.types.is_numeric_dtype(df[target])
            unique_count = df[target].nunique()
            
            # Heuristic: If numeric with many unique values (>10% of dataset size), 
            # likely regression, otherwise classification
            is_regression = is_numeric and unique_count > max(10, len(df) * 0.1)
            
            if is_regression:
                logger.info(f"Target column '{target}' appears to be for a regression problem")
                logger.info(f"Using binned stratification for regression target")
                
                # For regression, bin the target into quantiles for stratified sampling
                num_bins = min(10, unique_count)  # Use at most 10 bins
                
                # Create a temporary binned column for stratification
                df['_temp_bin'] = pd.qcut(df[target], q=num_bins, duplicates='drop')
                
                # Log the bin distribution
                bin_counts = df['_temp_bin'].value_counts().to_dict()
                logger.info(f"Target bin distribution: {bin_counts}")
                
                # Perform stratified split on the bins
                train_df = pd.DataFrame()
                test_df = pd.DataFrame()
                
                # For each bin
                for bin_value in df['_temp_bin'].unique():
                    # Get all rows for this bin
                    bin_df = df[df['_temp_bin'] == bin_value]
                    
                    # Shuffle the bin dataframe
                    bin_df = bin_df.sample(frac=1.0, random_state=42)
                    
                    # Calculate split index for this bin
                    bin_test_size = int(len(bin_df) * holdout_frac)
                    
                    # Split the bin dataframe
                    bin_test_df = bin_df.iloc[:bin_test_size].copy()
                    bin_train_df = bin_df.iloc[bin_test_size:].copy()
                    
                    # Append to the main dataframes
                    test_df = pd.concat([test_df, bin_test_df])
                    train_df = pd.concat([train_df, bin_train_df])
                
                # Remove the temporary bin column
                train_df = train_df.drop(columns=['_temp_bin'])
                test_df = test_df.drop(columns=['_temp_bin'])
                
                # Shuffle the final dataframes
                train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                test_df = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                
                # Log statistics about the target distribution in both sets
                train_target_mean = train_df[target].mean()
                test_target_mean = test_df[target].mean()
                train_target_std = train_df[target].std()
                test_target_std = test_df[target].std()
                
                logger.info(f"Training set target mean: {train_target_mean}, std: {train_target_std}")
                logger.info(f"Testing set target mean: {test_target_mean}, std: {test_target_std}")
                
                # Calculate and log the distribution similarity
                mean_diff_pct = abs((train_target_mean - test_target_mean) / train_target_mean) * 100 if train_target_mean != 0 else 0
                std_diff_pct = abs((train_target_std - test_target_std) / train_target_std) * 100 if train_target_std != 0 else 0
                
                logger.info(f"Target distribution difference: mean {mean_diff_pct:.2f}%, std {std_diff_pct:.2f}%")
                
                # Store regression-specific metrics
                regression_metrics = {
                    'train_mean': float(train_target_mean),
                    'test_mean': float(test_target_mean),
                    'train_std': float(train_target_std),
                    'test_std': float(test_target_std),
                    'mean_difference_percent': float(mean_diff_pct),
                    'std_difference_percent': float(std_diff_pct)
                }
                
            else:
                # Classification problem
                logger.info(f"Target column '{target}' appears to be for a classification problem")
                logger.info(f"Using stratified split for classification target")
                
                # Check if target has sufficient classes
                class_counts = df[target].value_counts()
                
                if unique_count < 2:
                    logger.warning(f"Target column '{target}' has only {unique_count} unique value. "
                                  f"This may not be suitable for classification tasks.")
                
                # Log class distribution
                logger.info(f"Target class distribution: {class_counts.to_dict()}")
                
                # Check for rare classes (less than 10 samples)
                rare_classes = class_counts[class_counts < 10].index.tolist()
                if rare_classes:
                    logger.warning(f"Found rare classes with fewer than 10 samples: {rare_classes}")
                    logger.warning("Consider data augmentation or different sampling strategies")
                
                # Custom implementation of stratified split
                train_df = pd.DataFrame()
                test_df = pd.DataFrame()
                
                # For each class in the target column
                for class_value in df[target].unique():
                    # Get all rows for this class
                    class_df = df[df[target] == class_value]
                    
                    # Shuffle the class dataframe
                    class_df = class_df.sample(frac=1.0, random_state=42)
                    
                    # Calculate split index for this class
                    class_test_size = int(len(class_df) * holdout_frac)
                    
                    # Split the class dataframe
                    class_test_df = class_df.iloc[:class_test_size].copy()
                    class_train_df = class_df.iloc[class_test_size:].copy()
                    
                    # Append to the main dataframes
                    test_df = pd.concat([test_df, class_test_df])
                    train_df = pd.concat([train_df, class_train_df])
                
                # Shuffle the final dataframes
                train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                test_df = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                
                # Verify class distribution in splits
                train_class_dist = train_df[target].value_counts().to_dict()
                test_class_dist = test_df[target].value_counts().to_dict()
                
                logger.info(f"Training set class distribution: {train_class_dist}")
                logger.info(f"Testing set class distribution: {test_class_dist}")
                
                # Check if all classes are represented in both splits
                train_classes = set(train_df[target].unique())
                test_classes = set(test_df[target].unique())
                all_classes = set(df[target].unique())
                
                if train_classes != all_classes or test_classes != all_classes:
                    logger.warning("Not all classes are represented in both splits!")
                    logger.warning(f"Missing in train: {all_classes - train_classes}")
                    logger.warning(f"Missing in test: {all_classes - test_classes}")
                
                # Store classification-specific metrics
                regression_metrics = None
        else:
            # If no target or target not in columns, use random split
            logger.info("Using random split (no target column specified)")
            
            # Shuffle the DataFrame
            shuffled_df = df.sample(frac=1.0, random_state=42)
            
            # Calculate the split point
            split_idx = int(len(shuffled_df) * (1 - holdout_frac))
            
            # Split the dataset
            train_df = shuffled_df.iloc[:split_idx].copy()
            test_df = shuffled_df.iloc[split_idx:].copy()
            
            # No regression metrics for random split
            regression_metrics = None
        
        logger.info(f"Split dataset into training set ({len(train_df)} rows) and testing set ({len(test_df)} rows)")
        
        # Create temporary directory if it doesn't exist
        os.makedirs("/tmp/data", exist_ok=True)
        
        # Save the split datasets to temporary files
        train_tmp_path = os.path.join("/tmp/data", "train_data.csv")
        test_tmp_path = os.path.join("/tmp/data", "test_data.csv")
        
        # Save the split datasets to temporary files
        train_df.to_csv(train_tmp_path, index=False)
        test_df.to_csv(test_tmp_path, index=False)
        
        # Generate S3 paths for the split datasets
        s3_base_path = os.path.dirname(data_location)
        train_s3_path = f"{s3_base_path}/train_data.csv"
        test_s3_path = f"{s3_base_path}/test_data.csv"
        
        # Upload the split datasets to S3
        train_bucket, train_key = parse_s3_path(train_s3_path)
        test_bucket, test_key = parse_s3_path(test_s3_path)
        
        s3_client.upload_file(train_tmp_path, train_bucket, train_key)
        s3_client.upload_file(test_tmp_path, test_bucket, test_key)
        
        logger.info(f"Uploaded training data to {train_s3_path}")
        logger.info(f"Uploaded testing data to {test_s3_path}")
        
        # Prepare results based on problem type
        results = {
            'training_data_location': train_s3_path,
            'testing_data_location': test_s3_path,
            'training_rows': len(train_df),
            'testing_rows': len(test_df),
            'holdout_fraction': holdout_frac,
            'target_column': target if target and target in df.columns else None,
        }
        
        # Add problem-specific metrics
        if target and target in df.columns:
            if is_regression:
                results['problem_type'] = 'regression'
                results['target_distribution'] = regression_metrics
            else:
                results['problem_type'] = 'classification'
                results['target_distribution'] = {
                    'training': train_df[target].value_counts().to_dict(),
                    'testing': test_df[target].value_counts().to_dict()
                }
        
        # Return the results using the correct APIResponse format
        return APIResponse(
            message="Dataset split successfully",
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error in train_test_split_dataset: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to split dataset: {str(e)}")

def exploratory_data_analysis(df: pd.DataFrame) -> APIResponse:
    """
    Performs basic exploratory data analysis on a dataset from S3.
    
    Args:
        df: Pandas DataFrame to analyze
        
    Returns:
        APIResponse: Object containing EDA results
    """
    logger.info(f"Performing exploratory data analysis on dataset")
    
    try:
        
        if df is None or df.empty:
            raise ValueError("No data found or empty dataset")
            
        # Get basic dataset information
        num_rows, num_cols = df.shape
        logger.info(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB
        logger.info(f"Memory usage: {memory_usage} MB")
        
        # Data types analysis
        dtypes_count = df.dtypes.value_counts().to_dict()
        logger.info(f"Data types count: {dtypes_count}")
        dtypes_by_column = {col: str(dtype) for col, dtype in df.dtypes.items()}
        logger.info(f"Data types by column: {dtypes_by_column}")
        
        # Missing values analysis
        missing_values = df.isnull().sum().to_dict()
        missing_percentage = {col: round((count / num_rows) * 100, 2) 
                             for col, count in missing_values.items() if count > 0}
        logger.info(f"Missing values: {missing_values}")
        logger.info(f"Missing percentage: {missing_percentage}")
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        numeric_stats = {}
        logger.info(f"Numeric columns: {numeric_cols}")
        
        if numeric_cols:
            # Basic statistics for numeric columns
            numeric_stats = {
                'summary': df[numeric_cols].describe().to_dict(),
                'skewness': {col: float(df[col].skew()) 
                            for col in numeric_cols if not df[col].isnull().all()},
                'kurtosis': {col: float(df[col].kurtosis()) 
                            for col in numeric_cols if not df[col].isnull().all()}
            }
            logger.info(f"Numeric stats: {numeric_stats}")
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_stats = {}
        
        if categorical_cols:
            # For each categorical column, get value counts (limited to top 10)
            categorical_stats = {
                col: {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(10).to_dict()
                } for col in categorical_cols
            }
            logger.info(f"Categorical stats: {categorical_stats}")
        
        # Correlation analysis for numeric columns (if at least 2 exist)
        correlation_matrix = {}
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().round(2)
            
            # Find highly correlated features (absolute correlation > 0.7)
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_correlations.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_matrix.iloc[i, j])
                        })
            
            # Convert NaN values to None for JSON serialization
            corr_dict = corr_matrix.to_dict()
            for col1 in corr_dict:
                for col2 in corr_dict[col1]:
                    if pd.isna(corr_dict[col1][col2]):
                        corr_dict[col1][col2] = None
            
            correlation_matrix = {
                'high_correlations': high_correlations,
                'matrix': corr_dict
            }
            logger.info(f"Correlation matrix: {correlation_matrix}")
        # Potential target column detection
        potential_targets = []
        for col in df.columns:
            # Check if column name contains keywords often used for target variables
            target_keywords = ['target', 'label', 'class', 'outcome', 'result', 'y']
            if any(keyword in col.lower() for keyword in target_keywords):
                potential_targets.append(col)
            
            # For categorical columns with few unique values
            if col in categorical_cols and df[col].nunique() < 10:
                potential_targets.append(col)
        
        # Compile all results
        eda_results = {
            'dataset_info': {
                'rows': num_rows,
                'columns': num_cols,
                'memory_usage_mb': round(memory_usage, 2),
                'column_names': df.columns.tolist()
            },
            'data_types': {
                'summary': {str(k): int(v) for k, v in dtypes_count.items()},
                'by_column': dtypes_by_column
            },
            'missing_values': {
                'columns_with_missing': {k: int(v) for k, v in missing_values.items() if v > 0},
                'missing_percentage': missing_percentage
            },
            'numeric_analysis': numeric_stats,
            'categorical_analysis': categorical_stats,
            'correlation_analysis': correlation_matrix,
            'potential_target_columns': potential_targets,
            'data_quality_issues': []
        }
        
        # Identify potential data quality issues
        data_quality_issues = []
        
        # Check for columns with high missing values (>20%)
        high_missing = [col for col, pct in missing_percentage.items() if pct > 20]
        if high_missing:
            data_quality_issues.append({
                'issue_type': 'high_missing_values',
                'description': f"Columns with >20% missing values: {', '.join(high_missing)}"
            })
        
        # Check for highly skewed numeric columns (skewness > 3 or < -3)
        if numeric_cols:
            high_skew = [col for col, skew in numeric_stats.get('skewness', {}).items() 
                        if abs(skew) > 3]
            if high_skew:
                data_quality_issues.append({
                    'issue_type': 'high_skewness',
                    'description': f"Highly skewed columns: {', '.join(high_skew)}"
                })
            logger.info(f"Data quality issues: {data_quality_issues}")
        
        # Check for highly correlated features
        if len(correlation_matrix.get('high_correlations', [])) > 0:
            corr_pairs = [f"{item['feature1']}/{item['feature2']}" 
                         for item in correlation_matrix.get('high_correlations', [])]
            data_quality_issues.append({
                'issue_type': 'high_correlation',
                'description': f"Highly correlated feature pairs: {', '.join(corr_pairs)}"
            })
            logger.info(f"Data quality issues: {data_quality_issues}")
        # Add data quality issues to results
        eda_results['data_quality_issues'] = data_quality_issues
        
        # Log the EDA results
        logger.info(f"EDA results: {eda_results}")
        logger.info(f"Data quality issues: {data_quality_issues}")
        
        return APIResponse(
            message="Exploratory data analysis completed successfully",
            results=eda_results
        )
        
    except Exception as e:
        logger.error(f"Error in exploratory_data_analysis: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to perform exploratory data analysis: {str(e)}")
    
def format_response(response_data: APIResponse, request_type: str, event: Dict) -> Dict:
    """Format and truncate response for the lambda"""
    try:
        # Convert response data to dict format
        response_body = {
            'application/json': {
                'body': {
                    'message': response_data.message,
                    'results': response_data.results
                }
            }
        }
        
        # Convert to JSON string to check size
        response_json = json.dumps(response_body)
        response_size = len(response_json.encode('utf-8'))
        MAX_RESPONSE_SIZE = 22 * 1024  # 22KB in bytes
        
        if response_size > MAX_RESPONSE_SIZE:
            logger.warning(f"Response size {response_size} bytes exceeds limit. Truncating content...")
            
            # Create truncated response structure
            truncated_response = {
                'message': f"Successfully completed {request_type} operation (results truncated)",
                'results': {
                    'truncated': True,
                    'original_size': response_size
                }
            }
            
            # Add essential information based on request type
            if request_type == 'Train':
                truncated_response['results'].update({
                    'model_location': response_data.results.get('model_location'),
                    'model_accuracy': response_data.results.get('model_accuracy')
                })
            elif request_type == 'Predict':
                truncated_response['results'].update({
                    'result_location': response_data.results.get('result_location'),
                    'total_predictions': response_data.results.get('total_predictions')
                })
            
            response_body['application/json']['body'] = truncated_response

        return {
            'messageVersion': '1.0',
            'response': {
                'actionGroup': event['actionGroup'],
                'apiPath': event['apiPath'],
                'httpMethod': event['httpMethod'],
                'httpStatusCode': 200,
                'responseBody': response_body
            }
        }
        
    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}", exc_info=True)
        return error_response(500, f"Error formatting response: {str(e)}", event)

def error_response(status_code: int, error_message: str, event: Dict) -> Dict:
    """Generate standardized error response"""
    return {
        'messageVersion': '1.0',
        'response': {
            'actionGroup': event.get('actionGroup'),
            'apiPath': event.get('apiPath'),
            'httpMethod': event.get('httpMethod', 'POST'),
            'httpStatusCode': status_code,
            'responseBody': {
                'application/json': {
                    'body': {
                        'error': error_message,
                        'status': 'ERROR'
                    }
                }
            }
        }
    }

def clean_tmp_directories():
    """Clean up temporary directories"""
    try:
        # Clean up model directory
        if os.path.exists('/tmp/model'):
            for root, dirs, files in os.walk('/tmp/model', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
                    
        # Clean up data directory
        if os.path.exists('/tmp/data'):
            for root, dirs, files in os.walk('/tmp/data', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
                    
    except Exception as e:
        logger.warning(f"Error cleaning temporary directories: {str(e)}")

def normalize_all_paths(parameters):
    """Normalize all S3 paths in the parameters"""
    model_location = parameters.model_location
    data_location = parameters.data_location
    train_data_location = parameters.train_data_location
    test_data_location = parameters.test_data_location
    result_data_location = parameters.result_data_location
    
    # Normalize paths
    if model_location:
        model_location = normalize_s3_path(model_location, S3_BUCKET_NAME)
    if data_location:
        data_location = normalize_s3_path(data_location, S3_BUCKET_NAME)
    if train_data_location:
        train_data_location = normalize_s3_path(train_data_location, S3_BUCKET_NAME)
    if test_data_location:
        test_data_location = normalize_s3_path(test_data_location, S3_BUCKET_NAME)
    if result_data_location:
        result_data_location = normalize_s3_path(result_data_location, S3_BUCKET_NAME)
    
    return {
        'model_location': model_location,
        'data_location': data_location,
        'train_data_location': train_data_location,
        'test_data_location': test_data_location,
        'result_data_location': result_data_location
    }

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Clean up temporary directories at start
        clean_tmp_directories()
        
        # Parse parameters
        parameters = parse_parameters(event)
        target = parameters.target
        api_path = parameters.api_path
        holdout_frac = parameters.holdout_frac
        hyperparameters = parameters.hyperparameters
        
        # Log the original parameters
        logger.info(f"Original parameters: {parameters}")
        
        # Normalize all paths
        normalized_paths = normalize_all_paths(parameters)
        model_location = normalized_paths['model_location']
        data_location = normalized_paths['data_location']
        train_data_location = normalized_paths['train_data_location']
        test_data_location = normalized_paths['test_data_location']
        result_data_location = normalized_paths['result_data_location']
        
        # Log the normalized paths
        logger.info(f"Normalized paths: {normalized_paths}")
        
        if api_path == 'Train':
            
            if not all([target, model_location, train_data_location, test_data_location]):
                raise ValueError("Missing required parameters. Need Target, ModelLocation, TrainDataLocation, TestDataLocation.")
            
            # Validate paths
            validate_model_location(model_location)
            validate_data_location(train_data_location)
            validate_data_location(test_data_location)
            
            # Load data
            train_df = load_data(train_data_location)
            test_df = load_data(test_data_location)

            # Validate target column exists
            validate_target_column(train_df, target)
            validate_target_column(test_df, target)
                
            # Validate target column has enough classes
            unique_classes = train_df[target].nunique()
            if unique_classes < 2:
                raise ValueError(
                    f"Target column '{target}' has only {unique_classes} unique value(s). "
                    "For classification, at least 2 different classes are required. "
                    f"Unique values found: {train_df[target].unique().tolist()}"
                )

            # Train model
            try:
                result = train(train_df, test_df, target, model_location, hyperparameters)
                # Use the format_response function to properly serialize the APIResponse object
                return format_response(result, "Train", event)
            except AssertionError as e:
                # Handle AutoGluon specific errors
                if "num_classes must be" in str(e):
                    raise ValueError(
                        f"Training failed: Target column has insufficient classes. {str(e)}. "
                        f"Check your target column '{target}' for data quality issues."
                    )
                raise
            except Exception as e:
                raise ValueError(f"Training failed: {str(e)}")
        elif api_path == 'Predict':
            if not all([model_location, data_location, result_data_location]):
                raise ValueError("Missing required parameters. Need ModelLocation, DataLocation, and ResultDataLocation.")
            
            # Validate paths
            validate_model_location(model_location)
            validate_data_location(data_location)
            validate_result_data_location(result_data_location)
            
            # Load data
            df = load_data(data_location)

            # Remove target column from dataframe
            df = df.drop(columns=[target])

            # Predict
            try:
                result = predict(df, model_location, result_data_location)
                # Use the format_response function to properly serialize the APIResponse object
                return format_response(result, "Predict", event)
            except Exception as e:
                raise ValueError(f"Prediction failed: {str(e)}")
            
        elif api_path == 'FeatureImportance':
            if not all([model_location, data_location]):
                raise ValueError("Missing required parameters. Need ModelLocation and DataLocation.")
                
            # Validate paths
            validate_model_location(model_location)
            validate_data_location(data_location)
            
            # Load data
            df = load_data(data_location)

            # Get feature importance
            try:
                result = feature_importance(df, model_location)
                # Use the format_response function to properly serialize the APIResponse object
                return format_response(result, "FeatureImportance", event)
            except Exception as e:
                raise ValueError(f"Feature importance calculation failed: {str(e)}")
        
        elif api_path == 'ExploratoryDataAnalysis':
            # For EDA, we only need the data location
            if not data_location:
                raise ValueError("Missing required parameter: DataLocation")

            # Validate data location
            validate_data_location(data_location)

            # Load data
            df = load_data(data_location)
            
            # Perform exploratory data analysis
            result = exploratory_data_analysis(df)

            # Return the results
            return format_response(result, "ExploratoryDataAnalysis", event)
            
        elif api_path == 'TrainTestSplit':
            
            if not all([holdout_frac]):
                raise ValueError("Missing required parameter: HoldoutFrac")
            
            if not data_location:
                raise ValueError("Missing required parameter: DataLocation")

            # Validate data location
            validate_data_location(data_location)

            # Load data
            df = load_data(data_location)
            
            # Train test split
            result = train_test_split_dataset(df, holdout_frac, data_location, target)

            # Return the results
            return format_response(result, "TrainTestSplit", event)
            
            
    except ValueError as e:
        # Handle validation errors (400)
        error_details = {
            "error": str(e),
            "error_type": "ValidationError",
            "parameters": parameters.dict() if 'parameters' in locals() else None,  # Convert Pydantic model to dict
            "data_info": {
                "shape": df.shape if 'df' in locals() else None,
                "columns": df.columns.tolist() if 'df' in locals() else None,
                "target_info": {
                    "unique_values": df[target].unique().tolist() if all(v in locals() for v in ['df', 'target']) else None,
                    "value_counts": df[target].value_counts().to_dict() if all(v in locals() for v in ['df', 'target']) else None
                } if all(v in locals() for v in ['df', 'target']) else None
            } if 'df' in locals() else None
        }
        
        logger.error(f"Validation error: {json.dumps(error_details, indent=2)}")
        
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event.get('actionGroup'),
                "apiPath": event.get('apiPath'),
                "httpMethod": event.get('httpMethod', 'POST'),
                "httpStatusCode": 400,
                "responseBody": {
                    "application/json": {
                        "body": error_details
                    }
                }
            }
        }
        
    except Exception as e:
        # Handle unexpected errors (500)
        error_details = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        
        logger.error(f"Internal error: {json.dumps(error_details, indent=2)}")
        
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event.get('actionGroup'),
                "apiPath": event.get('apiPath'),
                "httpMethod": event.get('httpMethod', 'POST'),
                "httpStatusCode": 500,
                "responseBody": {
                    "application/json": {
                        "body": error_details
                    }
                }
            }
        }
    finally:
        # Clean up temporary directories at end
        clean_tmp_directories()
