"""
Azure ML training pipeline for automated model training.
"""

import os
import logging
from typing import Dict, Any, Optional
from azureml.core import Workspace, Experiment, Environment, ComputeTarget
from azureml.core.compute import AmlCompute
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.core.conda_dependencies import CondaDependencies

from src.training.model_config import ModelConfig


class MLTrainingPipeline:
    """Azure ML training pipeline for automated model training."""
    
    def __init__(self, workspace: Workspace, config: ModelConfig):
        """Initialize the training pipeline."""
        self.workspace = workspace
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup compute target
        self.compute_target = self._setup_compute_target()
        
        # Setup environment
        self.environment = self._setup_environment()
        
        # Setup datasets
        self.datasets = self._setup_datasets()
    
    def _setup_compute_target(self) -> ComputeTarget:
        """Setup compute target for training."""
        compute_name = self.config.compute_target
        
        try:
            compute_target = self.workspace.compute_targets[compute_name]
            self.logger.info(f"Found existing compute target: {compute_name}")
        except KeyError:
            self.logger.info(f"Creating new compute target: {compute_name}")
            
            compute_config = AmlCompute.provisioning_configuration(
                vm_size='Standard_NC6',
                max_nodes=4,
                min_nodes=0,
                idle_seconds_before_scaledown=300
            )
            
            compute_target = AmlCompute.create(
                self.workspace,
                compute_name,
                compute_config
            )
            compute_target.wait_for_completion(show_output=True)
        
        return compute_target
    
    def _setup_environment(self) -> Environment:
        """Setup training environment."""
        env_name = "ml-training-env"
        
        try:
            environment = Environment.get(self.workspace, env_name)
            self.logger.info(f"Found existing environment: {env_name}")
        except Exception:
            self.logger.info(f"Creating new environment: {env_name}")
            
            environment = Environment(env_name)
            
            # Add conda dependencies
            conda_deps = CondaDependencies()
            conda_deps.add_conda_package("python=3.8")
            conda_deps.add_conda_package("tensorflow=2.13.0")
            conda_deps.add_conda_package("numpy=1.24.3")
            conda_deps.add_conda_package("pandas=2.0.3")
            conda_deps.add_conda_package("scikit-learn=1.3.0")
            conda_deps.add_conda_package("opencv-python=4.8.0.76")
            conda_deps.add_conda_package("mlflow=2.6.0")
            conda_deps.add_conda_package("azureml-core=1.53.0")
            
            # Add pip dependencies
            conda_deps.add_pip_package("azureml-mlflow==1.53.0")
            conda_deps.add_pip_package("opencensus-ext-azure==1.1.8")
            
            environment.python.conda_dependencies = conda_deps
            
            # Register environment
            environment.register(self.workspace)
        
        return environment
    
    def _setup_datasets(self) -> Dict[str, Any]:
        """Setup datasets for training."""
        datasets = {}
        
        # Training dataset
        try:
            train_dataset = self.workspace.datasets['training-data']
            datasets['train'] = train_dataset
            self.logger.info("Found existing training dataset")
        except Exception:
            self.logger.warning("Training dataset not found. Please register it first.")
        
        # Validation dataset
        try:
            val_dataset = self.workspace.datasets['validation-data']
            datasets['validation'] = val_dataset
            self.logger.info("Found existing validation dataset")
        except Exception:
            self.logger.warning("Validation dataset not found. Please register it first.")
        
        return datasets
    
    def create_training_pipeline(self) -> Pipeline:
        """Create the training pipeline."""
        # Pipeline data
        model_output = PipelineData(
            name="model_output",
            datastore=self.workspace.get_default_datastore()
        )
        
        metrics_output = PipelineData(
            name="metrics_output",
            datastore=self.workspace.get_default_datastore()
        )
        
        # Training step
        train_step = PythonScriptStep(
            name="train_model",
            script_name="train.py",
            source_directory="src/training",
            compute_target=self.compute_target,
            environment=self.environment,
            inputs=[
                DatasetConsumptionConfig("training_data", self.datasets.get('train')),
                DatasetConsumptionConfig("validation_data", self.datasets.get('validation'))
            ],
            outputs=[model_output, metrics_output],
            arguments=[
                "--config", self.config.to_dict(),
                "--model_output", model_output,
                "--metrics_output", metrics_output
            ],
            allow_reuse=False
        )
        
        # Model registration step
        register_step = PythonScriptStep(
            name="register_model",
            script_name="register_model.py",
            source_directory="src/training",
            compute_target=self.compute_target,
            environment=self.environment,
            inputs=[model_output, metrics_output],
            arguments=[
                "--model_path", model_output,
                "--metrics_path", metrics_output,
                "--model_name", f"{self.config.experiment_name}-model",
                "--workspace", self.workspace.name
            ],
            allow_reuse=False
        )
        
        # Create pipeline
        pipeline = Pipeline(
            workspace=self.workspace,
            steps=[train_step, register_step],
            description="ML Model Training Pipeline"
        )
        
        return pipeline
    
    def run_pipeline(self, experiment_name: Optional[str] = None) -> Any:
        """Run the training pipeline."""
        if experiment_name is None:
            experiment_name = self.config.experiment_name
        
        # Create experiment
        experiment = Experiment(self.workspace, experiment_name)
        
        # Create and submit pipeline
        pipeline = self.create_training_pipeline()
        pipeline_run = experiment.submit(pipeline)
        
        self.logger.info(f"Pipeline submitted: {pipeline_run.id}")
        
        # Wait for completion
        pipeline_run.wait_for_completion(show_output=True)
        
        return pipeline_run
    
    def create_hyperparameter_tuning_pipeline(self) -> Pipeline:
        """Create hyperparameter tuning pipeline."""
        from azureml.train.hyperdrive import HyperDriveConfig, RandomParameterSampling, PrimaryMetricGoal
        
        # Define hyperparameter search space
        param_sampling = RandomParameterSampling({
            '--learning_rate': [0.001, 0.01, 0.1],
            '--batch_size': [16, 32, 64],
            '--epochs': [50, 100, 150],
            '--dropout_rate': [0.3, 0.5, 0.7]
        })
        
        # Create estimator
        estimator = Estimator(
            source_directory="src/training",
            entry_script="train.py",
            compute_target=self.compute_target,
            environment=self.environment,
            hyperparameters={
                '--config': self.config.to_dict()
            }
        )
        
        # HyperDrive configuration
        hyperdrive_config = HyperDriveConfig(
            estimator=estimator,
            hyperparameter_sampling=param_sampling,
            policy=BanditPolicy(evaluation_interval=2, slack_factor=0.1),
            primary_metric_name='validation_accuracy',
            primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
            max_total_runs=20,
            max_concurrent_runs=4
        )
        
        # Create pipeline step
        hyperdrive_step = EstimatorStep(
            name="hyperparameter_tuning",
            estimator=hyperdrive_config,
            inputs=[
                DatasetConsumptionConfig("training_data", self.datasets.get('train')),
                DatasetConsumptionConfig("validation_data", self.datasets.get('validation'))
            ]
        )
        
        # Create pipeline
        pipeline = Pipeline(
            workspace=self.workspace,
            steps=[hyperdrive_step],
            description="Hyperparameter Tuning Pipeline"
        )
        
        return pipeline


def create_training_script():
    """Create the training script for the pipeline."""
    script_content = '''
import os
import sys
import json
import argparse
import logging
from azureml.core import Run
from azureml.core.model import Model
import mlflow
import mlflow.tensorflow

from model_config import ModelConfig
from data_loader import DataLoader
from trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Get Azure ML run context
    run = Run.get_context()
    
    # Load configuration
    config_dict = json.loads(args.config)
    config = ModelConfig(**config_dict)
    
    # Setup MLflow
    mlflow.set_tracking_uri(run.experiment.workspace.get_mlflow_tracking_uri())
    
    # Initialize components
    data_loader = DataLoader(config)
    trainer = ModelTrainer(config, run.experiment.workspace)
    
    # Load data
    train_data, val_data, test_data = data_loader.load_data_from_directory("data/")
    
    # Train model
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config.to_dict())
        
        # Train
        results = trainer.train(train_data, val_data, test_data)
        
        # Log metrics
        for metric_name, metric_value in results['test_metrics'].items():
            mlflow.log_metric(metric_name, metric_value)
            run.log(metric_name, metric_value)
        
        # Save model
        model_path = os.path.join(args.model_output, "model.h5")
        trainer.model.save(model_path)
        
        # Save metrics
        metrics_path = os.path.join(args.metrics_output, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log model
        mlflow.tensorflow.log_model(trainer.model, "model")
        
        logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
'''
    
    # Write script to file
    with open("src/training/train.py", "w") as f:
        f.write(script_content)
    
    print("Training script created: src/training/train.py")


def create_model_registration_script():
    """Create the model registration script."""
    script_content = '''
import os
import sys
import json
import argparse
import logging
from azureml.core import Workspace, Model
from azureml.core.conda_dependencies import CondaDependencies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--metrics_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--workspace", type=str, required=True)
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Get workspace
    workspace = Workspace.from_config()
    
    # Load metrics
    with open(args.metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create conda dependencies
    conda_deps = CondaDependencies()
    conda_deps.add_conda_package("tensorflow=2.13.0")
    conda_deps.add_conda_package("numpy=1.24.3")
    conda_deps.add_conda_package("pillow=10.0.0")
    
    # Register model
    model = Model.register(
        workspace=workspace,
        model_path=args.model_path,
        model_name=args.model_name,
        description="Trained ML model",
        conda_file=conda_deps,
        tags={
            'framework': 'tensorflow',
            'version': '2.13.0',
            'accuracy': str(metrics.get('test_metrics', {}).get('test_accuracy', 0))
        }
    )
    
    logger.info(f"Model registered with ID: {model.id}")

if __name__ == "__main__":
    main()
'''
    
    # Write script to file
    with open("src/training/register_model.py", "w") as f:
        f.write(script_content)
    
    print("Model registration script created: src/training/register_model.py")


if __name__ == "__main__":
    # Create training scripts
    create_training_script()
    create_model_registration_script()
    
    print("Training pipeline scripts created successfully!") 