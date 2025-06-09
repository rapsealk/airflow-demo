import sys
from datetime import datetime, timedelta, timezone

from airflow.providers.standard.operators.python import PythonOperator, PythonVirtualenvOperator
from airflow.sdk import DAG, Variable

HF_TOKEN = Variable.get("HF_TOKEN")

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

    def download_mnist_dataset() -> None:
        import os
        from pathlib import Path

        import datasets

        AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
        dataset_path = Path(AIRFLOW_HOME) / "datasets" / "mnist"
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True, exist_ok=True)

        hf_token = "hf_XXXXXXXXXXX"  # TODO:

        dataset = datasets.load_dataset(
            "ylecun/mnist",
            # data_dir=dataset_path.as_posix(),
            token=hf_token,
        )
        dataset.save_to_disk(dataset_path)

        print(f"Downloaded MNIST dataset: {dataset}")

    t2 = PythonVirtualenvOperator(
        task_id="download_mnist_dataset",
        python_callable=download_mnist_dataset,
        requirements=["datasets"],
        system_site_packages=False,
    )

    def train_mnist_model() -> None:
        # https://github.com/pytorch/examples/blob/65722fe3ce14447420cf922fdd713bfe900b410f/mnist/main.py
        import os
        from pathlib import Path

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torchvision import datasets, transforms
        from torch.optim.lr_scheduler import StepLR

        AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
        dataset_path = Path(AIRFLOW_HOME) / "datasets" / "mnist"

        lr = 1.0
        epochs = 14
        batch_size = 64
        test_batch_size = 1000
        gamma = 0.7
        no_accel = False
        dry_run = False
        seed = 1
        log_interval = 10
        save_model = True

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout(0.25)
                self.dropout2 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                output = F.log_softmax(x, dim=1)
                return output
            
        def train(args, model, device, train_loader, optimizer, epoch):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                    if dry_run:
                        break
        
        def test(model, device, test_loader):
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

        def main() -> None:
            use_accel = not no_accel and torch.accelerator.is_available()

            torch.manual_seed(seed)

            if use_accel:
                device = torch.accelerator.current_accelerator()
            else:
                device = torch.device("cpu")

            train_kwargs = {'batch_size': batch_size}
            test_kwargs = {'batch_size': test_batch_size}
            if use_accel:
                accel_kwargs = {
                    'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True,
                }
                train_kwargs.update(accel_kwargs)
                test_kwargs.update(accel_kwargs)

            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset1 = datasets.MNIST(dataset_path, train=True, download=True, transform=transform)
            dataset2 = datasets.MNIST(dataset_path, train=False, transform=transform)
            train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
            test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

            model = Net().to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=lr)

            scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
            for epoch in range(1, epochs + 1):
                train(None, model, device, train_loader, optimizer, epoch)
                test(model, device, test_loader)
                scheduler.step()

            if save_model:
                torch.save(model.state_dict(), "mnist_cnn.ckpt")

        main()

    t3 = PythonVirtualenvOperator(
        task_id="train_mnist_model",
        python_callable=train_mnist_model,
        requirements=["torch", "torchvision"],
        system_site_packages=False,
        # op_args=
    )

    # t3 = DockerOperator(
    #     task_id="print_docker_torchtune_version",
    #     image="nvcr.io/nvidia/pytorch:25.05-py3",
    #     auto_remove="force",
    #     cpus=1.0,
    #     shm_size=None,
    #     mem_limit="4g",
    #     # env_file=str(Path(__file__).absolute().parent / ".env"),
    #     env_file="{\"HF_TOKEN\": \"hf_XXXXXXXXX\"}",
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

    t1 >> t2 >> t3
