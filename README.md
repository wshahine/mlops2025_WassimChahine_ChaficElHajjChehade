# MLOps Project: NYC Taxi Trip Duration Predictor

This project implements an end-to-end MLOps pipeline to predict the duration of NYC taxi trips. It includes local development scripts, a containerized environment using Docker, and scalable cloud pipelines on AWS SageMaker.

## 📂 Project Structure

```text
├── .github/              # CI/CD workflows
├── data/                 # Local data storage (ignored by Git)
├── scripts/              # Python scripts for local execution
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── batch_inference.py
├── src/                  # Source code package
├── run_training_pipeline.py       # SageMaker Training Entrypoint
├── run_batch_inference_pipeline.py # SageMaker Inference Entrypoint
├── Dockerfile            # Container definition
├── docker-compose.yml    # Container orchestration
├── entrypoint.sh         # Docker entrypoint script
├── pyproject.toml        # Project dependencies
└── README.md
How to Run Locally
Install Dependencies: Ensure you have Python 3.11 installed.
(pip install -r requirements.txt
# OR if using uv
uv sync)

Run Training: Preprocesses data and trains the model.


(python scripts/train.py)
Run Inference: Generates predictions using the saved model.

(python scripts/batch_inference.py)

How to Run with Docker
We use Docker to ensure a consistent environment across development and production.

Build the Image:


(docker-compose build)
Run Training Container:


(docker-compose run app train)
Run Inference Container:

(docker-compose run app inference)
How to Run the SageMaker Pipeline
AWS SageMaker is used for scalable, automated pipeline execution.

Prerequisites:

AWS CLI configured with appropriate permissions.

S3 bucket created for data/artifacts.

Execute Training Pipeline: Spins up SageMaker instances to preprocess data and train the model.

(python run_training_pipeline.py)
Execute Batch Inference Pipeline: Running this script triggers a batch transform job on SageMaker.

(python run_batch_inference_pipeline.py)

Selected Metric & Justification
Metric: Root Mean Squared Error (RMSE)
Justification: We selected RMSE as our primary evaluation metric because:
Penalty for Large Errors: RMSE squares the errors before averaging, which penalizes large outliers more heavily than MAE. In taxi time prediction, being significantly wrong (e.g., predicting 10 minutes for a 50-minute ride) is much worse for user experience than small deviations.
Interpretability: The result is in the same unit as the target variable (seconds/minutes), making it easy to understand.
Model Choices & Performance
We trained and evaluated multiple models to select the best performer.

Linear Regression: RMSE = 10.87
Random Forest Regressor: RMSE = 10.75
Conclusion: The Random Forest Regressor was selected as the production model because it achieved a lower error rate (10.75 vs 10.87), demonstrating better capability in capturing the non-linear relationships between trip features (location, time) and duration.

Wassim Chahine (@wshahine)
Model Creation & Core Logic: Initiated the creation of the Linear Regression and Random Forest models within the training scripts.
SageMaker Architecture: Led the implementation of the Cloud Pipelines and authored the related entry scripts for training and inference.
Iterative Debugging: Identified and corrected logic errors in the partner’s preprocessing scripts during the critical integration phase.

Chafic El Hajj Chehade (@Cheff1999)
Model Execution & Environment: Managed the successful execution of models within the Docker container to ensure full reproducibility of results.
Docker & Infrastructure: Authored the Dockerfile, docker-compose configuration, and the entrypoint script.
Code Refinement & QA: Reviewed shared code, specifically fixing syntax errors and resolving Windows/Linux path compatibility issues in the model scripts to ensure smooth execution across different operating systems.
In summary, this project successfully transformed a raw machine learning experiment into a production-ready MLOps system. By leveraging Docker, we eliminated environment inconsistencies between team members, and by implementing AWS SageMaker Pipelines, we established a scalable infrastructure capable of handling large datasets efficiently. The selection of the Random Forest Regressor (RMSE 10.75) ensures accurate predictions, while the automated pipeline architecture allows for seamless model retraining and batch inference. This end-to-end solution demonstrates a robust workflow for deploying machine learning models in a collaborative, real-world setting.
