from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.utils.edgemodifier import Label
from docker.types import DeviceRequest, Mount

from datetime import datetime, timedelta

from certainty.common.constants import DATASET, NETWORK, QUEUE
from certainty.common.mlflow import create_evaluation_run, get_latest_model_run
from certainty.common.oxen import oxen_configure, oxen_checkout_dataset

default_args = {
    "owner": "airflow",
    "description": "Compute anchor for model explainability.",
    "depend_on_past": False,
    "start_date": datetime(2018, 1, 3),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

environment = {
    "MLFLOW_TRACKING_URI": "http://{{ var.value.mlflow_host }}:{{ var.value.mlflow_port }}",
    "MLFLOW_EXPERIMENT_NAME": "{{ dag.dag_id }}",
    "AWS_ACCESS_KEY_ID": "{{ var.value.minio_root_user }}",
    "AWS_SECRET_ACCESS_KEY": "{{ var.value.minio_root_password }}",
    "MLFLOW_S3_ENDPOINT_URL": "http://{{ var.value.minio_host }}:{{ var.value.minio_port }}",
}


@dag(
    # TODO: Rename this DAG. It should reflect the method(s) you are evaluating with this DAG!
    # Also make sure to change the name of the function below to match the DAG ID.
    dag_id=f"evaluate_anchors",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)
def evaluate_anchors(
    model_experiment_name_or_id: str = "yolov5_training",
    model_run_id: str = None,
    model_modality: Param = Param(
        default="image", type="string", enum=["image", "tabular", "text"]
    ),
    model_task: Param = Param(
        default="classify",
        type="string",
        enum=["classify", "detect", "regress"],
        description="The model's task. Note that detection models are treated as classifiers for the purpose of anchor computation, i.e. box positioning is not taken into account.",
    ),
    dataset: Param = Param(default=DATASET, type="string"),
    explanation_kwargs: Param = Param(
        default={"threshold": 0.95, "p_sample": 0.5, "tau": 0.25},
        type="object",
        description="Additional arguments to pass to the explainer. See https://docs.seldon.io/projects/alibi/en/latest/api/alibi.explainers.html#alibi.explainers.AnchorImage.explain and the respective methods for AnchorTabular and AnchorText.",
    ),
    segmentation_function: Param = Param(
        default="slic", type="string", enum=["felzenszwalb", "slic", "quickshift"]
    ),
    segmentation_kwargs: Param = Param(
        default={"n_segments": 15, "compactness": 20, "sigma": 0.5, "start_label": 0},
        type="object",
        description="Additional arguments to pass to the segmentation function. See: https://scikit-image.org/docs/dev/api/skimage.segmentation.html#module-skimage.segmentation",
    ),
    seed: Param = Param(default=42, type="integer"),
):
    """
    evaluate_anchors downloads a trained model and computes anchors for its predictions.

    Given an experiment name or id, it fetches the latest run id (c.f. get_latest_run task) and
    then loads the model from this run (c.f. consume_model task).

    Alternatively, one may directly provide a run id using model_run_id, in which case
    the model is loaded from this run directly.
    """

    @task.docker(
        image="bily/alibi-explain",
        api_version="auto",
        auto_remove=True,
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        shm_size=16000000000,
        environment=environment,
        queue=QUEUE,
        mounts=[
            Mount(
                target="/app/datasets",
                source="/raid/certainty/",
                type="bind",
            )
        ],
    )
    def compute_image_anchors(
        model_run_id: str, evaluation_run_id: str, dataset_dir: str, anchor_args={}
    ):
        """
        consume_model demonstrates how a model can be loaded from MLflow and used for
        downstream applications. It also shows how to log back metrics to mlflow.
        """
        from alibi.explainers import AnchorImage
        import mlflow
        import numpy as np
        from ultralytics import YOLO
        from torch.utils.data import ConcatDataset, DataLoader
        from ultralytics.data.dataset import YOLODataset
        import yaml

        from pathlib import Path

        from certainty.tasks.yolo.predictors import (
            classification_predictor,
            detection_predictor,
        )

        # First, load the model weights.
        weights_path = mlflow.artifacts.download_artifacts(
            run_id=model_run_id, artifact_path="best.pt"
        )
        model = YOLO(weights_path)

        # Below, we load the test set from the provided dataset.
        dataset_dir = Path(dataset_dir)
        with open(dataset_dir / "data.yaml", "r") as file:
            dataset_cfg = yaml.safe_load(file)

        test_dirs = [dataset_dir / dir for dir in dataset_cfg["test"]]
        test_sets = [
            YOLODataset(img_path=test_dir, data=dataset_cfg, augment=False)
            for test_dir in test_dirs
        ]
        test_set = ConcatDataset(test_sets)
        test_loader = DataLoader(test_set)

        sample = next(iter(test_loader))
        img = sample["img"]
        model.predict(img)  # NOTE: Required to set up the predictor object.

        if "classify" == anchor_args["model_task"]:
            predictor = classification_predictor(model)
        elif "detect" == anchor_args["model_task"]:
            predictor = detection_predictor(model)

        explanation_kwargs = anchor_args.pop("explanation_kwargs")
        segmentation_function = anchor_args.pop("segmentation_function")
        segmentation_kwargs = anchor_args.pop("segmentation_kwargs")

        explainer = AnchorImage(
            predictor,
            tuple(img.shape[1:]),
            segmentation_fn=segmentation_function,
            segmentation_kwargs=segmentation_kwargs,
            images_background=None,
        )
        explanation = explainer.explain(img.numpy()[0], **explanation_kwargs)

        anchor = np.moveaxis(explanation.anchor, 0, -1)
        segments = np.moveaxis(explanation.segments, 0, -1)

        with mlflow.start_run(evaluation_run_id):
            mlflow.log_param("segmentation_function", segmentation_function)
            mlflow.log_params(explanation_kwargs)
            mlflow.log_params(segmentation_kwargs)
            mlflow.log_image(anchor, "anchor.jpg")
            mlflow.log_image(segments, "anchor_segments.jpg")
        return

    @task(queue=QUEUE)
    def compute_tabular_anchors(
        model_run_id: str, evaluation_run_id: str, dataset_dir: str, anchor_args={}
    ):
        raise NotImplementedError

    @task(queue=QUEUE)
    def compute_text_anchors(
        model_run_id: str, evaluation_run_id: str, dataset_dir: str, anchor_args={}
    ):
        raise NotImplementedError

    @task.branch(queue=QUEUE)
    def modality_selector(modality):
        if "image" == modality:
            return ["image_branch"]
        elif "tabular" == modality:
            return ["tabular_branch"]
        else:
            return ["text_branch"]

    anchor_args = dict(
        model_task=model_task,
        explanation_kwargs=explanation_kwargs,
        segmentation_function=segmentation_function,
        segmentation_kwargs=segmentation_kwargs,
        seed=seed,
    )

    get_latest_model_run_instance = get_latest_model_run.override(
        environment=environment
    )(model_experiment_name_or_id, model_run_id)
    create_evaluation_run_instance = create_evaluation_run.override(
        environment=environment
    )(get_latest_model_run_instance)

    oxen_checkout_dataset_instance = oxen_checkout_dataset(dataset)
    oxen_configure() >> oxen_checkout_dataset_instance

    modality_selector_instance = modality_selector(model_modality)

    for modality in ["image", "tabular", "text"]:
        branch = EmptyOperator(task_id=f"{modality}_branch", queue=QUEUE)
        (
            modality_selector_instance
            >> branch
            >> eval(f"compute_{modality}_anchors")(
                get_latest_model_run_instance,
                create_evaluation_run_instance,
                oxen_checkout_dataset_instance,
                anchor_args,
            )
        )


evaluate_anchors()
