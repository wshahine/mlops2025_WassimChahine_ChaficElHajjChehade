import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import ProcessingStep, TransformStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.inputs import TransformInput

# --- CONFIGURATION ---
ROLE_ARN = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole-2024"
BUCKET = "your-sagemaker-bucket-name"
PROJECT_PREFIX = "nyc-taxi-duration"
MODEL_S3_URI = f"s3://{BUCKET}/models/model.tar.gz" # Path to trained model
# ---------------------

sagemaker_session = sagemaker.Session()

def create_inference_pipeline():
    # 1. PREPROCESSING (Reuse the same processor logic)
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        role=ROLE_ARN,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=f"{PROJECT_PREFIX}-inf-process"
    )

    step_process = ProcessingStep(
        name="InferencePreprocess",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=f"s3://{BUCKET}/data/inference_input/", destination="/opt/ml/processing/input")
        ],
        outputs=[
            ProcessingOutput(output_name="inference_data", source="/opt/ml/processing/output")
        ],
        code="scripts/preprocess.py"
    )

    # 2. BATCH INFERENCE (Using SageMaker Transformer)
    transformer = Transformer(
        model_name="nyc-taxi-model", # This assumes the model is already registered in SageMaker
        instance_count=1,
        instance_type="ml.m5.xlarge",
        output_path=f"s3://{BUCKET}/predictions/",
        accept="text/csv",
        assemble_with="Line"
    )

    step_transform = TransformStep(
        name="BatchInference",
        transformer=transformer,
        inputs=TransformInput(
            data=step_process.properties.ProcessingOutputConfig.Outputs["inference_data"].S3Output.S3Uri,
            content_type="text/csv"
        )
    )

    # 3. DEFINE PIPELINE
    pipeline = Pipeline(
        name=f"{PROJECT_PREFIX}-inference-pipeline",
        steps=[step_process, step_transform],
        sagemaker_session=sagemaker_session
    )

    return pipeline

if __name__ == "__main__":
    pipeline = create_inference_pipeline()
    pipeline.upsert(role_arn=ROLE_ARN)
    execution = pipeline.start()
    print(f"Inference Pipeline started. Execution ARN: {execution.arn}")
