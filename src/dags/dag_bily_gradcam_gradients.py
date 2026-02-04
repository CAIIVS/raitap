from airflow.decorators import dag, task
from docker.types import DeviceRequest, Mount

from datetime import datetime, timedelta

from certainty.common.constants import DATASET, NETWORK, QUEUE
from certainty.common.mlflow import create_evaluation_run, get_latest_model_run
from certainty.common.oxen import oxen_configure, oxen_checkout_dataset


default_args = {
    "owner": "airflow",
    # TODO: Write a short description of this DAG.
    "description": "An example of a DAG for method evaluation.",
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
    dag_id=f"evaluate_gradcam_gradients",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)
def evaluate_gradcam_gradients(
    model_experiment_name_or_id: str = "yolov5_training", model_run_id: str = None
):
    """
    Evaluate the GradCAM gradients attributes for the YOLO model.
    """

    @task.docker(
        # TODO: Use your docker image below. Either ensure it is available on DGX3
        # or add the Dockerfile to bily/docker-tasks/ in this repository.
        image="bily/gradients",
        multiple_outputs=True,
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
    def gradients_explainer(model_run_id: str, evaluation_run_id: str, dataset_dir: str):
        """
        gradients_explainer
        
        """
        import numpy as np
        import yaml
        from pathlib import Path

        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        import torch
        import mlflow
        from ultralytics import YOLO
        from torch.utils.data import ConcatDataset, DataLoader
        from ultralytics.data.dataset import YOLODataset
        
        from captum.attr import GuidedGradCam
        
        # Imports for custom version of Captum function
        import warnings
        from typing import Optional, Tuple, Union
        from enum import Enum

        import numpy as np
        from matplotlib import pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.figure import Figure
        from matplotlib.pyplot import axis, figure
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from numpy import ndarray

        class ImageVisualizationMethod(Enum):
            heat_map = 1
            blended_heat_map = 2
            original_image = 3
            masked_image = 4
            alpha_scaling = 5

        class VisualizeSign(Enum):
            positive = 1
            absolute_value = 2
            negative = 3
            all = 4

        def _prepare_image(attr_visual: ndarray):
            return np.clip(attr_visual.astype(int), 0, 255)

        def _normalize_attr(
            attr: ndarray,
            sign: str,
            outlier_perc: Union[int, float] = 2,
            reduction_axis: Optional[int] = None,
        ):
            attr_combined = attr
            if reduction_axis is not None:
                attr_combined = np.sum(attr, axis=reduction_axis)

            # Choose appropriate signed values and rescale, removing given outlier percentage.
            if VisualizeSign[sign] == VisualizeSign.all:
                threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
            elif VisualizeSign[sign] == VisualizeSign.positive:
                attr_combined = (attr_combined > 0) * attr_combined
                threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
            elif VisualizeSign[sign] == VisualizeSign.negative:
                attr_combined = (attr_combined < 0) * attr_combined
                threshold = -1 * _cumulative_sum_threshold(
                    np.abs(attr_combined), 100 - outlier_perc
                )
            elif VisualizeSign[sign] == VisualizeSign.absolute_value:
                attr_combined = np.abs(attr_combined)
                threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
            else:
                raise AssertionError("Visualize Sign type is not valid.")
            return _normalize_scale(attr_combined, threshold)

        def _cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
            # given values should be non-negative
            assert percentile >= 0 and percentile <= 100, (
                "Percentile for thresholding must be " "between 0 and 100 inclusive."
            )
            sorted_vals = np.sort(values.flatten())
            cum_sums = np.cumsum(sorted_vals)
            threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
            return sorted_vals[threshold_id]

        def _normalize_scale(attr: ndarray, scale_factor: float):
            assert scale_factor != 0, "Cannot normalize by scale factor = 0"
            if abs(scale_factor) < 1e-5:
                warnings.warn(
                    "Attempting to normalize by value approximately 0, visualized results"
                    "may be misleading. This likely means that attribution values are all"
                    "close to 0."
                )
            attr_norm = attr / scale_factor
            return np.clip(attr_norm, -1, 1)

        # Custom version of Captum function XXX to circumvent error due to updated pyplot version
        # Adapted line is highlighted, search for "<-- This line was changed"
        def visualize_image_attr(
            attr: ndarray,
            original_image: Union[None, ndarray] = None,
            method: str = "heat_map",
            sign: str = "absolute_value",
            plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
            outlier_perc: Union[int, float] = 2,
            cmap: Union[None, str] = None,
            alpha_overlay: float = 0.5,
            show_colorbar: bool = False,
            title: Union[None, str] = None,
            fig_size: Tuple[int, int] = (6, 6),
            use_pyplot: bool = True,
        ):
            r"""
            Visualizes attribution for a given image by normalizing attribution values
            of the desired sign (positive, negative, absolute value, or all) and displaying
            them using the desired mode in a matplotlib figure.

            Args:

                attr (numpy.ndarray): Numpy array corresponding to attributions to be
                            visualized. Shape must be in the form (H, W, C), with
                            channels as last dimension. Shape must also match that of
                            the original image if provided.
                original_image (numpy.ndarray, optional): Numpy array corresponding to
                            original image. Shape must be in the form (H, W, C), with
                            channels as the last dimension. Image can be provided either
                            with float values in range 0-1 or int values between 0-255.
                            This is a necessary argument for any visualization method
                            which utilizes the original image.
                            Default: None
                method (str, optional): Chosen method for visualizing attribution.
                            Supported options are:

                            1. `heat_map` - Display heat map of chosen attributions

                            2. `blended_heat_map` - Overlay heat map over greyscale
                            version of original image. Parameter alpha_overlay
                            corresponds to alpha of heat map.

                            3. `original_image` - Only display original image.

                            4. `masked_image` - Mask image (pixel-wise multiply)
                            by normalized attribution values.

                            5. `alpha_scaling` - Sets alpha channel of each pixel
                            to be equal to normalized attribution value.

                            Default: `heat_map`
                sign (str, optional): Chosen sign of attributions to visualize. Supported
                            options are:

                            1. `positive` - Displays only positive pixel attributions.

                            2. `absolute_value` - Displays absolute value of
                            attributions.

                            3. `negative` - Displays only negative pixel attributions.

                            4. `all` - Displays both positive and negative attribution
                            values. This is not supported for `masked_image` or
                            `alpha_scaling` modes, since signed information cannot
                            be represented in these modes.

                            Default: `absolute_value`
                plt_fig_axis (tuple, optional): Tuple of matplotlib.pyplot.figure and axis
                            on which to visualize. If None is provided, then a new figure
                            and axis are created.
                            Default: None
                outlier_perc (float or int, optional): Top attribution values which
                            correspond to a total of outlier_perc percentage of the
                            total attribution are set to 1 and scaling is performed
                            using the minimum of these values. For sign=`all`, outliers
                            and scale value are computed using absolute value of
                            attributions.
                            Default: 2
                cmap (str, optional): String corresponding to desired colormap for
                            heatmap visualization. This defaults to "Reds" for negative
                            sign, "Blues" for absolute value, "Greens" for positive sign,
                            and a spectrum from red to green for all. Note that this
                            argument is only used for visualizations displaying heatmaps.
                            Default: None
                alpha_overlay (float, optional): Alpha to set for heatmap when using
                            `blended_heat_map` visualization mode, which overlays the
                            heat map over the greyscaled original image.
                            Default: 0.5
                show_colorbar (bool, optional): Displays colorbar for heatmap below
                            the visualization. If given method does not use a heatmap,
                            then a colormap axis is created and hidden. This is
                            necessary for appropriate alignment when visualizing
                            multiple plots, some with colorbars and some without.
                            Default: False
                title (str, optional): Title string for plot. If None, no title is
                            set.
                            Default: None
                fig_size (tuple, optional): Size of figure created.
                            Default: (6,6)
                use_pyplot (bool, optional): If true, uses pyplot to create and show
                            figure and displays the figure after creating. If False,
                            uses Matplotlib object oriented API and simply returns a
                            figure object without showing.
                            Default: True.

            Returns:
                2-element tuple of **figure**, **axis**:
                - **figure** (*matplotlib.pyplot.figure*):
                            Figure object on which visualization
                            is created. If plt_fig_axis argument is given, this is the
                            same figure provided.
                - **axis** (*matplotlib.pyplot.axis*):
                            Axis object on which visualization
                            is created. If plt_fig_axis argument is given, this is the
                            same axis provided.

            Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities.
                >>> net = ImageClassifier()
                >>> ig = IntegratedGradients(net)
                >>> # Computes integrated gradients for class 3 for a given image .
                >>> attribution, delta = ig.attribute(orig_image, target=3)
                >>> # Displays blended heat map visualization of computed attributions.
                >>> _ = visualize_image_attr(attribution, orig_image, "blended_heat_map")
            """
            # Create plot if figure, axis not provided
            if plt_fig_axis is not None:
                plt_fig, plt_axis = plt_fig_axis
            else:
                if use_pyplot:
                    plt_fig, plt_axis = plt.subplots(figsize=fig_size)
                else:
                    plt_fig = Figure(figsize=fig_size)
                    plt_axis = plt_fig.subplots()

            if original_image is not None:
                if np.max(original_image) <= 1.0:
                    original_image = _prepare_image(original_image * 255)
            elif ImageVisualizationMethod[method] != ImageVisualizationMethod.heat_map:
                raise ValueError(
                    "Original Image must be provided for"
                    "any visualization other than heatmap."
                )

            # Remove ticks and tick labels from plot.
            plt_axis.xaxis.set_ticks_position("none")
            plt_axis.yaxis.set_ticks_position("none")
            plt_axis.set_yticklabels([])
            plt_axis.set_xticklabels([])
            plt_axis.grid(visible=None) # <-- This line was changed

            heat_map = None
            # Show original image
            if ImageVisualizationMethod[method] == ImageVisualizationMethod.original_image:
                assert (
                    original_image is not None
                ), "Original image expected for original_image method."
                if len(original_image.shape) > 2 and original_image.shape[2] == 1:
                    original_image = np.squeeze(original_image, axis=2)
                plt_axis.imshow(original_image)
            else:
                # Choose appropriate signed attributions and normalize.
                norm_attr = _normalize_attr(attr, sign, outlier_perc, reduction_axis=2)

                # Set default colormap and bounds based on sign.
                if VisualizeSign[sign] == VisualizeSign.all:
                    default_cmap = LinearSegmentedColormap.from_list(
                        "RdWhGn", ["red", "white", "green"]
                    )
                    vmin, vmax = -1, 1
                elif VisualizeSign[sign] == VisualizeSign.positive:
                    default_cmap = "Greens"
                    vmin, vmax = 0, 1
                elif VisualizeSign[sign] == VisualizeSign.negative:
                    default_cmap = "Reds"
                    vmin, vmax = 0, 1
                elif VisualizeSign[sign] == VisualizeSign.absolute_value:
                    default_cmap = "Blues"
                    vmin, vmax = 0, 1
                else:
                    raise AssertionError("Visualize Sign type is not valid.")
                cmap = cmap if cmap is not None else default_cmap

                # Show appropriate image visualization.
                if ImageVisualizationMethod[method] == ImageVisualizationMethod.heat_map:
                    heat_map = plt_axis.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax)
                elif (
                    ImageVisualizationMethod[method]
                    == ImageVisualizationMethod.blended_heat_map
                ):
                    assert (
                        original_image is not None
                    ), "Original Image expected for blended_heat_map method."
                    plt_axis.imshow(np.mean(original_image, axis=2), cmap="gray")
                    heat_map = plt_axis.imshow(
                        norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha_overlay
                    )
                elif ImageVisualizationMethod[method] == ImageVisualizationMethod.masked_image:
                    assert VisualizeSign[sign] != VisualizeSign.all, (
                        "Cannot display masked image with both positive and negative "
                        "attributions, choose a different sign option."
                    )
                    plt_axis.imshow(
                        _prepare_image(original_image * np.expand_dims(norm_attr, 2))
                    )
                elif ImageVisualizationMethod[method] == ImageVisualizationMethod.alpha_scaling:
                    assert VisualizeSign[sign] != VisualizeSign.all, (
                        "Cannot display alpha scaling with both positive and negative "
                        "attributions, choose a different sign option."
                    )
                    plt_axis.imshow(
                        np.concatenate(
                            [
                                original_image,
                                _prepare_image(np.expand_dims(norm_attr, 2) * 255),
                            ],
                            axis=2,
                        )
                    )
                else:
                    raise AssertionError("Visualize Method type is not valid.")

            # Add colorbar. If given method is not a heatmap and no colormap is relevant,
            # then a colormap axis is created and hidden. This is necessary for appropriate
            # alignment when visualizing multiple plots, some with heatmaps and some
            # without.
            if show_colorbar:
                axis_separator = make_axes_locatable(plt_axis)
                colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
                if heat_map:
                    plt_fig.colorbar(heat_map, orientation="horizontal", cax=colorbar_axis)
                else:
                    colorbar_axis.axis("off")
            if title:
                plt_axis.set_title(title)

            if use_pyplot:
                plt.show()

            return plt_fig, plt_axis

        

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
            YOLODataset(img_path=test_dir, data=dataset_cfg) for test_dir in test_dirs
        ]
        test_set = ConcatDataset(test_sets)
        test_loader = DataLoader(test_set)

        # GRADIENTS METHOD
        # Load image
        img = next(iter(test_loader))["img"]

        # Cast from uint8 to float32
        norm_img = img/255.0

        # Enable GPU support
        norm_img = norm_img.to(device='cuda')

        # Get class with maximum predicted probability
        out = model(img)  # See https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results for a description of the YOLO model output.
        
        max_idx = torch.argmax(out[0].probs.data).item()
        print('Predicted class: ', max_idx)

        # Extract backbone model and final layer
        bb_model = model.model.model
        layer = bb_model[9].conv.conv

        # Create gradients explainer
        explainer = GuidedGradCam(bb_model, layer=layer)

        # Call attribution function (size matches input size, Nx3x640x640)
        attribution = explainer.attribute(norm_img, target=max_idx)

        # Create and save attribution image
        default_cmap = LinearSegmentedColormap.from_list('custom black', 
                                                        [(0, '#ffffff'),
                                                        (0.25, '#000000'),
                                                        (1, '#000000')], N=256)

        visualize_image_attr(np.transpose(attribution.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive')

        fig = plt.gcf()
        fig.savefig("gradcam_gradients.png")

        # Save image to mlflow
        mlflow.log_artifact("gradcam_gradients.png")

    

    get_latest_model_run_instance = get_latest_model_run.override(
        environment=environment
    )(model_experiment_name_or_id, model_run_id)
    create_evaluation_run_instance = create_evaluation_run.override(
        environment=environment
    )(get_latest_model_run_instance)

    oxen_checkout_dataset_instance = oxen_checkout_dataset(DATASET)
    oxen_configure() >> oxen_checkout_dataset_instance

    gradients_explainer(
        get_latest_model_run_instance,
        create_evaluation_run_instance,
        oxen_checkout_dataset_instance,
    )


evaluate_gradcam_gradients()