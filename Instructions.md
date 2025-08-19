# Instructions for Running the Tech Assessment Platform
## Creating a new DAG
A new DAG contains parameters, a task with a decorator that specifies the operator, and subtasks.

Use the following DAG definition (needs to be adapted):
```
with DAG(
    dag_id="example_python_operator",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example"],
) as dag:
    operator = PythonOperator(
        task_id=get_task_id(),
        python_callable=get_array,
        dag=dag,
    )
```

Use the following decorator and task:

## Debugging a DAG
You can refresh the DAG list in airflow by shutting down and relaunching airflow. Otherwise, just wait a few minutes until the DAG list is updated.

## Running the Pipeline Locally
### Running AirFlow
1. Start airflow with the following command:
```
airflow standalone
```

2. Connect to http://localhost:8080/home.

3. Refresh the DAGs.

4. Click on the DAG you want to start and click the "Play" button.

5. View the console output by clicking on the respective run and then on the "Logs" section.

### Running MLFlow
1. Start mlflow with the following command:
```
mlflow server --host 127.0.0.1 --port 8081
```

2. Connect to http://localhost:8081.

## Running the Pipeline on the Cluster
Connect to the cluster with the following command:

```
ssh <your account>@dallas.zhaw.ch
```

Get a shell on a worker:

```
srun --account=cai_res --partition=cpu_ia --pty bash
```

Activate venv for podman compose:

```
source /raid/persistent_scratch/foro/venvs/podcomp/bin/activate
```

Run podman compose:

```
podman-compose -f simple_podman_compose.yaml up --build
```

If it works correctly, there will be messages in the terminal from the postgres, redis, and airflow-apiserver services.

Access the airflow apiserver in your browser on e. g. http://sandiego.zhaw.ch:8660/. This command depends on which worker you have been connected to when you asked for an interactive session.