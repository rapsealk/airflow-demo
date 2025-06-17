import os
from datetime import datetime, timedelta, timezone

from airflow.providers.standard.operators.python import PythonVirtualenvOperator
from airflow.sdk import DAG

HF_TOKEN = os.getenv("HF_TOKEN")

with DAG(
    dag_id="mnist",
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="PyTorch MNIST classification example DAG",
    params={
        "hf_token": HF_TOKEN,
    },
    start_date=datetime.now(timezone.utc) + timedelta(seconds=10),
    catchup=False,
    tags=["example"],
) as dag:
    def download_mnist_dataset(**kwargs) -> None:
        import os
        from pathlib import Path

        import datasets

        AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
        dataset_path = Path(AIRFLOW_HOME) / "datasets" / "mnist"
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True, exist_ok=True)

        dataset = datasets.load_dataset(
            "ylecun/mnist",
            token=kwargs.get("hf_token"),
        )
        dataset.save_to_disk(dataset_path)

        print(f"Downloaded MNIST dataset: {dataset}")

    download_mnist_dataset_task = PythonVirtualenvOperator(
        task_id="download_mnist_dataset",
        python_callable=download_mnist_dataset,
        requirements=["datasets"],
        system_site_packages=False,
        op_kwargs={
            "params": {
                "hf_token": dag.params.get("hf_token"),
            },
        },
    )

    def train_mnist_model(**kwargs) -> None:
        # https://github.com/pytorch/examples/blob/65722fe3ce14447420cf922fdd713bfe900b410f/mnist/main.py
        import os
        from pathlib import Path

        import torch
        import torch.optim as optim
        from torchvision import datasets, transforms
        from torch.optim.lr_scheduler import StepLR

        from mnist.v1.models import Net
        from mnist.v1.trainer import Trainer

        AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
        dataset_path = Path(AIRFLOW_HOME) / "datasets" / "mnist"

        def main() -> None:
            use_accel = not kwargs["params"]["no_accel"] and torch.accelerator.is_available()

            torch.manual_seed(kwargs["params"]["seed"])

            if use_accel:
                device = torch.accelerator.current_accelerator()
            else:
                device = torch.device("cpu")

            train_kwargs = {"batch_size": kwargs["params"]["batch_size"]}
            test_kwargs = {"batch_size": kwargs["params"]["test_batch_size"]}
            if use_accel:
                accel_kwargs = {
                    "num_workers": 1,
                    "pin_memory": True,
                    "shuffle": True,
                }
                train_kwargs.update(accel_kwargs)
                test_kwargs.update(accel_kwargs)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset1 = datasets.MNIST(dataset_path, train=True, download=True, transform=transform)
            dataset2 = datasets.MNIST(dataset_path, train=False, transform=transform)
            train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
            test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

            model = Net().to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=kwargs["params"]["lr"])
            scheduler = StepLR(optimizer, step_size=1, gamma=kwargs["params"]["gamma"])

            trainer = Trainer(model)

            class AttributeDict(dict):
                def __getattr__(self, item):
                    return self[item]

            for epoch in range(1, kwargs["params"]["epochs"] + 1):
                trainer.train(AttributeDict(kwargs["params"]), device, train_loader, optimizer, epoch)
                trainer.test(device, test_loader)
                scheduler.step()

            if kwargs["params"]["save_model"]:
                torch.save(model.state_dict(), "mnist_cnn.ckpt")

        main()

    train_mnist_model_task = PythonVirtualenvOperator(
        task_id="train_mnist_model",
        python_callable=train_mnist_model,
        requirements=["torch", "torchvision"],
        system_site_packages=False,
        op_kwargs={
            "params": {
                "lr": 1.0,
                "epochs": 14,
                "batch_size": 64,
                "test_batch_size": 1000,
                "gamma": 0.7,
                "no_accel": False,
                "dry_run": False,
                "seed": 1,
                "log_interval": 10,
                "save_model": True,
            },
        },
    )

    # t3 = DockerOperator(
    #     task_id="print_docker_torchtune_version",
    #     image="nvcr.io/nvidia/pytorch:25.05-py3",
    #     auto_remove="force",
    #     cpus=1.0,
    #     shm_size=None,
    #     mem_limit="4g",
    #     # env_file=str(Path(__file__).absolute().parent / ".env"),
    #     # env_file="{\"HF_TOKEN\": \"hf_XXXXXXXXX\"}",
    #     network_mode="bridge",
    #     privileged=False,
    #     mounts=[
    #         Mount(
    #             source=str(Path(__file__).absolute().parent / "entrypoint.sh"),
    #             target="/entrypoint.sh",
    #             type="bind",
    #         ),
    #     ],
    #     entrypoint="/entrypoint.sh",
    #     docker_url="unix://var/run/docker.sock",
    # )

    download_mnist_dataset_task >> train_mnist_model_task
