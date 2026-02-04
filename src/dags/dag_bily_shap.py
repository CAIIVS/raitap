from airflow.decorators import dag, task
from airflow.models.param import Param
from docker.types import DeviceRequest, Mount

from datetime import datetime, timedelta

from certainty.common.constants import DATASET, NETWORK, QUEUE
from certainty.common.evaluation import evaluate_attribution_method, EVALUATION_METRICS
from certainty.common.mlflow import create_evaluation_run, get_latest_model_run
from certainty.common.oxen import oxen_configure, oxen_checkout_dataset

default_args = {
    "owner": "airflow",
    "description": "DAG for computing SHAP values of a pre-trained YOLO model.",
    "depend_on_past": False,
    "start_date": datetime(2018, 1, 3),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

environment = {
    "MLFLOW_TRACKING_URI": "http://{{ var.value.mlflow_host }}:{{ var.value.mlflow_port }}",
    "MLFLOW_EXPERIMENT_NAME": "{{ dag.dag_id }}-testing",
    "AWS_ACCESS_KEY_ID": "{{ var.value.minio_root_user }}",
    "AWS_SECRET_ACCESS_KEY": "{{ var.value.minio_root_password }}",
    "MLFLOW_S3_ENDPOINT_URL": "http://{{ var.value.minio_host }}:{{ var.value.minio_port }}",
}


@dag(
    dag_id=f"evaluate_shapley_values",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)
def evaluate_shapley_values(
    model_experiment_name_or_id: str = "yolov5_training",
    model_run_id: str = None,
    model_task: Param = Param(
        default="classify",
        type="string",
        enum=["classify", "detect", "regress"],
        description="The model's task. Note that detection models are treated as classifiers for the purpose of anchor computation, i.e. box positioning is not taken into account.",
    ),
    dataset: Param = Param(default=DATASET, type="string"),
):
    """
    evaluate_shapley_values computes shapley values for the YOLO model trained in the run given by `mlflow_run_id`
    on the given dataset `dataset` and adds them to the run.
    """

    @task.docker(
        image="bily/shap",
        multiple_outputs=True,
        api_version="auto",
        auto_remove=True,
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                target="/app/datasets",
                source="/raid/certainty/",
                type="bind",
            )
        ],
        shm_size=16000000000,
        environment=environment,
        queue=QUEUE,
    )
    def partition_explainer(
        model_run_id: str, evaluation_run_id: str, dataset_dir: str, shap_args: dict
    ):
        from certainty.tasks.yolo.predictors import (
            detection_predictor,
            classification_predictor,
        )
        import h5py
        import matplotlib.pyplot as plt
        import mlflow
        import numpy as np
        import shap
        import torch
        from torch.utils.data import ConcatDataset, DataLoader
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.models.yolo import YOLO
        import yaml

        from pathlib import Path

        weights_path = mlflow.artifacts.download_artifacts(
            run_id=model_run_id, artifact_path="best.pt"
        )
        model = YOLO(weights_path)

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

        model = YOLO(weights_path)
        nc = dataset_cfg["nc"]
        names = dataset_cfg["names"]

        sample = next(iter(test_loader))
        img = sample["img"]
        cls = sample["cls"]
        model.predict(img)  # NOTE: Required to set up the predictor object.

        model_task = shap_args.pop("model_task")
        if "classify" == model_task:
            predict = classification_predictor(model)
        elif "detect" == model_task:
            predict = detection_predictor(model)
        else:
            raise ValueError(f"Unsupported model task: {model_task}")

        def predict_wrapper(img):
            tmp = img.copy()
            tmp = torch.tensor(tmp)
            tmp = tmp.permute(0, 3, 1, 2)
            return predict(tmp)

        img = img.permute(0, 2, 3, 1)
        for masker_name in [
            "blur(64, 64)",
            "blur(128, 128)",
            "blur(256, 256)",
            "inpaint_telea",
            "inpaint_ns",
        ]:
            masker = shap.maskers.Image(masker_name, img[0].shape)
            explainer = shap.Explainer(predict_wrapper, masker, output_names=names)

            shap_values = explainer(
                img,
                max_evals=500,
                batch_size=50,
                outputs=shap.Explanation.argsort.flip[:nc],
            )
            shap_values.data = shap_values.data.cpu().numpy()[0]
            shap_values.values = [
                val for val in np.moveaxis(shap_values.values[0], -1, 0)
            ]
            shap.image_plot(
                shap_values=shap_values.values,
                pixel_values=shap_values.data,
                labels=shap_values.output_names,
                show=False,
            )
            fig = plt.gcf()

            with h5py.File("shap_data.h5", "w") as f:
                f.create_dataset("input/data", data=img.permute(0, 3, 1, 2))
                f.create_dataset(
                    "input/labels",
                    data=cls.cpu().numpy().reshape(1, -1),
                )

            shap_value_for_cls = shap_values.values[int(cls)]
            shap_value_for_cls = np.expand_dims(shap_value_for_cls, axis=0)

            print(shap_value_for_cls.shape)

            with open("shap_values.npy", "wb") as f:
                np.save(f, np.transpose(shap_value_for_cls, (0, 3, 1, 2)))

            with mlflow.start_run(run_id=evaluation_run_id):
                mlflow.log_figure(fig, f"partition_{masker_name}_shap.png")
                mlflow.log_artifact("shap_data.h5")
                mlflow.log_artifact("shap_values.npy")
        return

    @task.docker(
        image="bily/fast-shap",
        multiple_outputs=True,
        api_version="auto",
        auto_remove=True,
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                target="/app/datasets",
                source="/raid/certainty/",
                type="bind",
            )
        ],
        shm_size=16000000000,
        environment=environment,
        queue=QUEUE,
    )
    def train_fast_shap_surrogate(
        model_run_id: str, evaluation_run_id: str, dataset_dir: str, shap_args: dict
    ):
        from certainty.tasks.shap.dataset import YOLODatasetWrapper
        from fastshap import ImageSurrogate
        from fastshap.utils import DatasetInputOnly, MaskLayer2d, KLDivLoss
        import mlflow
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, ConcatDataset
        from torchvision.models import resnet18
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.models.yolo import YOLO

        import yaml

        from pathlib import Path

        device = torch.device("cuda")

        weights_path = mlflow.artifacts.download_artifacts(
            run_id=model_run_id, artifact_path="best.pt"
        )

        dataset_dir = Path(dataset_dir)
        with open(dataset_dir / "data.yaml", "r") as file:
            dataset_cfg = yaml.safe_load(file)

        class ClassificationWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self._model = model

            def forward(self, img):
                prep = self._model.predictor.preprocess(img)
                prediction = self._model.predictor.inference(prep)[0]
                prediction = prediction.cpu()
                nc = prediction.shape[1] - 4  # number of classes
                nm = prediction.shape[1] - nc - 4
                mi = 4 + nc  # mask start index
                xc = prediction[:, 4:mi].amax(1) > 0.0  # candidates
                prediction = prediction.transpose(-1, -2)[0]
                output = []
                for _, x in enumerate(prediction):
                    x = prediction[xc[0]]  # confidence
                    _, cls, _ = x.split((4, nc, nm), 1)
                    best_pred = cls[torch.argmax(cls).item() // cls.shape[1]]
                    output.append(best_pred)
                output = torch.vstack(output)
                return output

        resnet = resnet18(num_classes=dataset_cfg["nc"])
        # Patch resent input conv to support 4 channels.
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)

        surrogate_model = nn.Sequential(
            MaskLayer2d(value=0, append=True),
            resnet,
        ).to(device)
        surrogate = ImageSurrogate(
            surrogate_model,
            width=640,
            height=640,
            superpixel_size=40,  # FIXME: don't hardcode sizes
        )

        train_dirs = [dataset_dir / dir for dir in dataset_cfg["train"]]
        val_dirs = [dataset_dir / dir for dir in dataset_cfg["val"]]
        train_sets = [
            YOLODatasetWrapper(YOLODataset(img_path=test_dir, data=dataset_cfg))
            for test_dir in train_dirs
        ]
        val_sets = [
            YOLODatasetWrapper(YOLODataset(img_path=test_dir, data=dataset_cfg))
            for test_dir in val_dirs
        ]
        train_set = DatasetInputOnly(ConcatDataset(train_sets))
        val_set = DatasetInputOnly(ConcatDataset(val_sets))

        yolo = YOLO(weights_path)
        yolo(train_sets[0][0]["img"])  # initialize predictor
        model = ClassificationWrapper(yolo)

        surrogate.train_original_model(
            train_set,
            val_set,
            model,
            batch_size=128,
            validation_batch_size=32,
            max_epochs=100,
            loss_fn=KLDivLoss(),
            lookback=10,
            bar=True,
            verbose=True,
        )
        surrogate_model.cpu()
        torch.save(surrogate_model, "surrogate.pt")

        with mlflow.start_run(run_id=evaluation_run_id):
            mlflow.log_artifact("surrogate.pt")
        return evaluation_run_id

    @task.docker(
        image="bily/shap",
        multiple_outputs=True,
        api_version="auto",
        auto_remove=True,
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                target="/app/datasets",
                source="/raid/certainty/",
                type="bind",
            )
        ],
        shm_size=16000000000,
        environment=environment,
        queue=QUEUE,
    )
    def train_fast_shap_explainer(
        evaluation_run_id: str, dataset_dir: str, shap_args: dict
    ):
        from fastshap import FastSHAP
        from fastshap.unet import UNet
        from fastshap.utils import DatasetInputOnly
        import mlflow
        import torch
        import torch.nn as nn
        from torch.utils.data import ConcatDataset
        from ultralytics.data.dataset import YOLODataset
        import yaml

        from pathlib import Path

        device = torch.device("cuda")

        weights_path = mlflow.artifacts.download_artifacts(
            run_id=evaluation_run_id, artifact_path="surrogate.pt"
        )
        surrogate = torch.load(weights_path)

        explainer = UNet(n_classes=10, num_down=2, num_up=1, num_convs=3).to(device)
        fastshap = FastSHAP(explainer, surrogate, link=nn.LogSoftmax(dim=1))

        dataset_dir = Path(dataset_dir)
        with open(dataset_dir / "data.yaml", "r") as file:
            dataset_cfg = yaml.safe_load(file)

        train_dirs = [dataset_dir / dir for dir in dataset_cfg["train"]]
        val_dirs = [dataset_dir / dir for dir in dataset_cfg["val"]]
        train_sets = [
            YOLODataset(img_path=test_dir, data=dataset_cfg) for test_dir in train_dirs
        ]
        val_sets = [
            YOLODataset(img_path=test_dir, data=dataset_cfg) for test_dir in val_dirs
        ]
        train_set = ConcatDataset(train_sets)
        val_set = DatasetInputOnly(ConcatDataset(val_sets))

        # Train
        fastshap.train(
            train_set,
            val_set,
            batch_size=128,
            num_samples=2,
            max_epochs=200,
            eff_lambda=1e-2,
            validation_samples=1,
            lookback=10,
            bar=True,
            verbose=True,
        )

        # Save explainer
        explainer.cpu()
        torch.save(explainer, "explainer.pt")

    shap_args = dict(
        model_task=model_task,
    )

    get_latest_model_run_instance = get_latest_model_run.override(
        environment=environment
    )(model_experiment_name_or_id, model_run_id)
    create_evaluation_run_instance = create_evaluation_run.override(
        environment=environment
    )(get_latest_model_run_instance)

    oxen_checkout_dataset_instance = oxen_checkout_dataset(dataset)
    oxen_configure() >> oxen_checkout_dataset_instance

    partition_explainer_instance = partition_explainer(
        get_latest_model_run_instance,
        create_evaluation_run_instance,
        oxen_checkout_dataset_instance,
        shap_args,
    )

    for metric in EVALUATION_METRICS:
        evaluate_attribution_method_instance = evaluate_attribution_method.override(
            task_id=f"compute_{metric}", environment=environment
        )(
            get_latest_model_run_instance,
            model_task,
            create_evaluation_run_instance,
            "shap_data.h5",
            "shap_values.npy",
            metric,
        )
        partition_explainer_instance >> evaluate_attribution_method_instance

    train_fast_shap_surrogate_instance = train_fast_shap_surrogate(
        get_latest_model_run_instance,
        create_evaluation_run_instance,
        oxen_checkout_dataset_instance,
        shap_args,
    )
    train_fast_shap_explainer(
        train_fast_shap_surrogate_instance, oxen_checkout_dataset_instance, shap_args
    )


evaluate_shapley_values()
