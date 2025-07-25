"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
    Join,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TuningStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter




BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_default_bucket(region):
    boto_session = boto3.Session(region_name=region)
    sm_session = sagemaker.session.Session(boto_session=boto_session)
    return sm_session.default_bucket()

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client

def get_session(region, default_bucket=None):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")

    if default_bucket is None:
        default_bucket = get_default_bucket(region)

    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket=None):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    if default_bucket is None:
        default_bucket = get_default_bucket(region)

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{sagemaker_session.default_bucket()}/datasets/abalone/abalone.csv",

    )
    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data", input_data],
    )
    step_process = ProcessingStep(
        name="PreprocessAbaloneData",
        step_args=step_args,
    )

    # training step for generating model artifacts
    # model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AbaloneTrain"
    bucket = sagemaker_session.default_bucket()
    # define your prefix explicitly
    prefix = f"{base_job_prefix}/output"

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=f"s3://{bucket}/{prefix}",# model_path,
        base_job_name=f"{base_job_prefix}/abalone-train",
        sagemaker_session=pipeline_session,
        role=role,
    )
    # xgb_train.set_hyperparameters(
    #     objective="reg:linear",
    #     num_round=50,
    #     max_depth=5,
    #     eta=0.2,
    #     gamma=4,
    #     min_child_weight=6,
    #     subsample=0.7,
    #     silent=0,
    # )


    ##################### NEW LOGIC BEGINS ##########################################
    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        subsample=0.7,
        gamma=4,
    )

    hp_ranges = {
        "eta": ContinuousParameter(0.01, 0.3),
        "max_depth": IntegerParameter(3, 10),
        "min_child_weight": ContinuousParameter(0, 10),
        # "subsample": ContinuousParameter(0.5, 1.0),
        # "colsample_bytree": ContinuousParameter(0.5, 1.0),
        # "gamma": ContinuousParameter(0, 5),
        # "lambda": ContinuousParameter(0, 10),
        # "alpha": ContinuousParameter(0, 10),
    }

    objective_metric_name = "validation:rmse"

    tuner = HyperparameterTuner(
        estimator=xgb_train,
        objective_metric_name=objective_metric_name,  # or your chosen metric
        hyperparameter_ranges=hp_ranges,
        strategy="Bayesian",
        objective_type="Minimize",
        max_jobs=20,
        max_parallel_jobs=4,
    )

    tuning_step = TuningStep(
        name="BayesianTuning",
        tuner=tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )

    ### ScriptProcessor for Evaluation
    # script_eval = ScriptProcessor(
    #     image_uri=image_uri,
    #     command=["python3"],
    #     instance_type=processing_instance_type,
    #     instance_count=1,
    #     base_job_name=f"{base_job_prefix}/script-abalone-eval",
    #     sagemaker_session=pipeline_session,
    #     role=role,
    # )

    script_eval = SKLearnProcessor(
        framework_version="0.23-1",  # or another supported version
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-abalone-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )

    # Fetch best model artifact from tuning step
    # bucket = sagemaker_session.default_bucket()
    # prefix = f"{base_job_prefix}/AbaloneTrain"
    top_model_s3_uri = tuning_step.get_top_model_s3_uri(
        top_k=0,
        s3_bucket=bucket,
        prefix=prefix,
    )

    # model_bucket_key = f"{sagemaker_session.default_bucket()}/{base_job_prefix}/AbaloneTrain"

    # model_artifact = Join(
    #     on="/",
    #     values=[
    #         f"s3://{sagemaker_session.default_bucket()}",
    #         base_job_prefix,
    #         "AbaloneTrain",
    #         tuning_step.properties.BestTrainingJob.TrainingJobName,
    #         "output",
    #         "model.tar.gz",
    #     ],
    # )
    # model_prefix = f"{base_job_prefix}/AbaloneTrain"
    #
    # top_model_s3_uri = tuning_step.get_top_model_s3_uri(
    #     top_k=0,
    #     s3_bucket=sagemaker_session.default_bucket(),
    #     prefix=model_prefix,
    # )

    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=top_model_s3_uri,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )

    evaluation_report = PropertyFile(
        name="AbaloneEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateAbaloneModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{step_eval.arguments['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']}/evaluation.json", #"{}/evaluation.json".format(
                #step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            #),
            content_type="application/json",
        )
    )

    model = Model(
        image_uri=image_uri,
        model_data=top_model_s3_uri,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_register = ModelStep(
        name="RegisterBestModel",
        step_args=model.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium", "ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status,
            model_metrics=model_metrics,
        ),
    )

    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.rmse.value",
        ),
        right=0.4,  # <-- Adjust threshold based on expected performance
    )

    step_cond = ConditionStep(
        name="CheckAbaloneEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[
            step_process,
            tuning_step,
            step_eval,
            step_cond,
        ],
        sagemaker_session=pipeline_session,
    )
    return pipeline

