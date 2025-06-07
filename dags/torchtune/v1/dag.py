import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.python import PythonOperator, PythonVirtualenvOperator
from airflow.sdk import DAG, Variable
from docker.types import DeviceRequest, Mount

with DAG(
    dag_id="torchtune",
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="PyTorch fine-tuning example DAG",
    start_date=datetime.now(timezone.utc) + timedelta(seconds=10),
    catchup=False,
    tags=["example"],
) as dag:
    def _print_python_version() -> None:
        print(f"Python version: {sys.version}")

    t1 = PythonOperator(
        task_id="print_python_version",
        python_callable=_print_python_version,
    )

    def _print_torchtune_version() -> None:
        # import torchtune
        # print(f"Torchtune version: {torchtune.__version__}")
        pass

    t2 = PythonVirtualenvOperator(
        task_id="print_torchtune_version",
        python_callable=_print_torchtune_version,
        # requirements=["torch", "torchao", "torchtune"],
        system_site_packages=False,
    )

    t3 = DockerOperator(
        task_id="print_docker_torchtune_version",
        image="nvcr.io/nvidia/pytorch:25.05-py3",
        auto_remove="force",
        cpus=1.0,
        shm_size=None,
        mem_limit="4g",
        environment={
            "HF_TOKEN": Variable.get("HF_TOKEN"),
        },
        # https://docs.docker.com/compose/how-tos/gpu-support/#example-of-a-compose-file-for-running-a-service-with-access-to-1-gpu-device
        device_requests=[
            DeviceRequest(
                driver="nvidia",
                count=-1,
                capabilities=[["gpu"]],
            ),
        ],
        # env_file=str(Path(__file__).absolute().parent / ".env"),
        network_mode="bridge",
        privileged=False,
        mounts=[
            Mount(
                source=str(Path(__file__).absolute().parent / "entrypoint.sh"),
                target="/entrypoint.sh",
                type="bind",
            ),
        ],
        entrypoint="/entrypoint.sh",
        # docker_url="unix://var/run/docker.sock",
    )

    t1 >> t2 >> t3
