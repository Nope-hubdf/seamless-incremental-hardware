import asyncio
import typing
from copy import deepcopy

from flyteplugins.dask import Dask, Scheduler, WorkerGroup

import flyte
import flyte.remote


def compute_squares(data):
    """Simple function to compute squares of numbers"""
    return [x * x for x in data]


image = (
    flyte.Image.from_debian_base(name="dask")
    .with_apt_packages("wget")
    .with_pip_packages("dask[complete]", "flyteplugins-dask", "distributed")
)

task_env = flyte.TaskEnvironment(
    name="hello_dask", 
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi")), 
    image=image
)

dask_config = Dask(
    workers=WorkerGroup(
        number_of_workers=1,  
        resources=flyte.Resources(cpu="1", memory="2Gi"), 
    ),
    scheduler=Scheduler(
        resources=flyte.Resources(cpu="1", memory="2Gi"),  
    ),
)

dask_env = flyte.TaskEnvironment(
    name="dask_env",
    plugin_config=dask_config,
    image=image,
    resources=flyte.Resources(cpu=(2, 3), memory=("800Mi", "1500Mi")),
)


@dask_env.task
async def hello_dask_nested(n: int = 5) -> typing.List[int]:
    """Dask task that processes data using distributed computing"""
    print("running dask task")
    
  
    from distributed import get_client
    import dask.array as da
    
    # Get the existing distributed client (set up by Flyte)
    client = get_client()
    print(f"Connected to scheduler: {client.scheduler.address}")
    
    # Use distributed arrays 
    data = da.arange(n, chunks=2)
    result = (data * data).compute()
    
    return result.tolist()


@task_env.task
async def dask_overrider(worker_replicas: int, n: int, worker_cpu: str, worker_memory: str ) -> typing.List[int]: #can be extended to handle limits
    """Task that dynamically overrides the Dask configuration"""
    updated_dask_config = deepcopy(dask_config)
    updated_dask_config.workers.number_of_workers = worker_replicas
    updated_dask_config.workers.resources = flyte.Resources(cpu=worker_cpu, memory=worker_memory)

    return await hello_dask_nested.override(plugin_config=updated_dask_config)(n=n)


if __name__ == "__main__":
    flyte.init_from_config()
    
    # Run with custom worker replicas and resources
    run = flyte.run(dask_overrider, worker_replicas=6, n=15, worker_cpu="2", worker_memory="4Gi")
    print("run name:", run.name)
    print("run url:", run.url)

    action_details = flyte.remote.ActionDetails.get(run_name=run.name, name="a0")
    if action_details.pb2.attempts:
        for log in action_details.pb2.attempts[-1].log_info:
            print(f"{log.name}: {log.uri}")
    else:
        print("No execution attempts found yet. Check the run status.")