import boto3
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline

# --- CONFIGURATION ---
# I updated this with your specific Account ID and Role Name
ROLE_ARN = "arn:aws:iam::989654687856:role/SageMakerExecutionRole"

# I updated this with the bucket name we just chose
BUCKET = "mlops-nyc-taxi-989654687856" 

PROJECT_PREFIX = "nyc-taxi-duration"

# ------------------------------------

sagemaker_session = sagemaker.Session()

def create_training_pipeline():
    # 1. PREPROCESSING & FEATURE ENGINEERING STEP
    # We use a Scikit-Learn Processor to run your scripts
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        role=ROLE_ARN,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=f"{PROJECT_PREFIX}-process"
    )

    step_process = ProcessingStep(
        name="PreprocessAndFeatureEng",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=f"s3://{BUCKET}/data/raw/", destination="/opt/ml/processing/input")
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="model_data", source="/opt/ml/processing/model_data") # e.g. encoders
        ],
        # We assume you combine preprocessing logic or call this script
        code="scripts/preprocess.py" 
    )

    # 2. TRAINING STEP
    # We use a Scikit-Learn Estimator to run train.py
    sklearn_estimator = SKLearn(
        entry_point="scripts/train.py",
        framework_version="1.0-1",
        instance_type="ml.m5.xlarge",
        role=ROLE_ARN,
        output_path=f"s3://{BUCKET}/models/",
        base_job_name=f"{PROJECT_PREFIX}-train"
    )

    step_train = TrainingStep(
        name="TrainModel",
        estimator=sklearn_estimator,
        inputs={
            "train": step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            "test": step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri
        }
    )

    # 3. DEFINE PIPELINE
    pipeline = Pipeline(
        name=f"{PROJECT_PREFIX}-training-pipeline",
        steps=[step_process, step_train],
        sagemaker_session=sagemaker_session
    )

    return pipeline

if __name__ == "__main__":
    pipeline = create_training_pipeline()
    pipeline.upsert(role_arn=ROLE_ARN)
    execution = pipeline.start()
    print(f"Training Pipeline started. Execution ARN: {execution.arn}")
