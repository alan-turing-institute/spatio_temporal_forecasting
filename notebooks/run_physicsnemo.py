#!/usr/bin/env python
# coding: utf-8

# # PhysicsNEMO FNO Training with Hydra Configuration
#
# This notebook demonstrates training an autoregressive FNO model using PhysicsNEMO with configuration management via Hydra.


"""https://docs.nvidia.com/physicsnemo/latest/user-guide/simple_training_example.html."""

from pathlib import Path
import physicsnemo
import torch
from physicsnemo.metrics.general.mse import mse
import logging
from the_well.data.datamodule import WellDataModule
from the_well.data.datasets import WellDataset
from the_well.benchmark.metrics import VRMSE, RMSE
from einops import rearrange
from hydra import compose, initialize
from omegaconf import OmegaConf
from physicsnemo.utils import StaticCaptureTraining
from physicsnemo.distributed import DistributedManager
import torch
from spatio_temporal_forecasting.config_schema import Config, register_configs

root_path = Path("../../autoemulate/autoemulate/experimental/")


# Initialize the DistributedManager. This will automatically
# detect the number of processes the job was launched with and
# set those configuration parameters appropriately.
DistributedManager.initialize()

# Get instance of the DistributedManager
dist = DistributedManager()


# Register structured configs for type safety
register_configs()

# Initialize Hydra and load configuration with schema validation
# By registering the schema and referencing it in defaults, Hydra automatically
# creates a properly typed DictConfig that matches our Config dataclass
with initialize(version_base=None, config_path="configs"):
    cfg: Config = compose(config_name="config")  # type: ignore

print("Configuration loaded with type validation:")
print(f"Model: {cfg.model.name}")
print(f"Device: {cfg.device}")
print(f"Training epochs: {cfg.training.epochs}")
print(f"Learning rate: {cfg.training.learning_rate}")


# Make a datamodule using Hydra config
logging.basicConfig(level=logging.INFO)

ae_data_module = WellDataModule(
    well_base_path=str(root_path / "exploratory/data/the_well/datasets"),
    well_dataset_name=cfg.dataset.name,
    n_steps_input=cfg.dataset.n_steps_input,
    n_steps_output=cfg.dataset.n_steps_output,
    batch_size=cfg.dataset.batch_size,
    train_dataset=WellDataset,
)

output_path = cfg.output_path

# CUDA check
print(torch.cuda.is_available())


dataloader = ae_data_module.train_dataloader()
dataloader_iter = iter(dataloader)
batch = next(dataloader_iter)


_, n_time_steps, height, width, n_channels = batch["input_fields"].shape


# In[ ]:


from spatio_temporal_forecasting.AR_FNO import AutoregressiveFNO
from spatio_temporal_forecasting.fno_emulator import MultivariableFNO

# Build FNO model from Hydra config
device = cfg.device
fno_base = MultivariableFNO(
    n_vars=cfg.model.n_vars,
    n_modes=tuple(cfg.model.n_modes),
    hidden_channels=cfg.model.hidden_channels,
    n_layers=cfg.model.n_layers,
    use_skip_connections=cfg.model.use_skip_connections,
)
model = AutoregressiveFNO(
    fno_model=fno_base, t_in=cfg.model.t_in, t_out=cfg.model.t_out
).to(dist.device)

# Set up DistributedDataParallel if using more than a single process.
if dist.distributed:
    ddps = torch.cuda.Stream()
    with torch.cuda.stream(ddps):
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                dist.local_rank
            ],  # Set the device_id to be the local rank of this process on this node
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )
    torch.cuda.current_stream().wait_stream(ddps)

dataloader = ae_data_module.train_dataloader()


# In[ ]:


from physicsnemo.launch.logging import LaunchLogger, PythonLogger

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda step: 0.85**step
)

# Initialize the logger
logger = PythonLogger("main")  # General python logger
LaunchLogger.initialize()


# Create training step function with optimization wrapper
# StaticCaptureTraining calls `backward` on the loss and
# `optimizer.step()` so you don't have to do that
# explicitly.
@StaticCaptureTraining(
    model=model,
    optim=optimizer,
    cuda_graph_warmup=11,
)
def training_step(batch):
    y_true = batch["output_fields"].to(device)
    y_true = y_true[..., :1]  # only first channel
    x = batch["input_fields"].to(device)
    x = x[..., :1]  # only first channel
    x = rearrange(x, "b t h w c -> b t c h w")
    y_pred = model(x)
    y_pred = rearrange(y_pred, "b t c h w -> b t h w c")
    loss = mse(y_pred, y_true)
    return loss


# Use logger methods to track various information during training
logger.info("Starting Training!")
for epoch in range(2):
    with LaunchLogger("train", epoch=epoch) as launchlog:
        for batch_idx, batch in enumerate(ae_data_module.train_dataloader()):
            optimizer.zero_grad()
            loss = training_step(batch)

            scheduler.step()
            launchlog.log_minibatch({"Loss": loss.detach().cpu().numpy()})

        launchlog.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
logger.info("Finished Training!")
