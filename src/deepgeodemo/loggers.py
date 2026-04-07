# Experimental logger for trackio

# import trackio
# from torch import Tensor
# from lightning.pytorch.loggers.logger import Logger
# from lightning.pytorch.utilities.rank_zero import rank_zero_only
# import warnings

# class TrackioLogger(Logger):
#     def __init__(self, project_name, experiment_name):
#         super().__init__()
#         self.project_name = project_name
#         self.experiment_name = experiment_name
#         trackio.init(project=self.project_name, name=self.experiment_name)

#     @rank_zero_only
#     def log_metrics(self, metrics, step):
#         assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
#         for k, v in metrics.items():
#             if isinstance(v, Tensor):
#                 v = v.item()
#         trackio.log(metrics)
    
#     @rank_zero_only
#     def log_hyperparams(self, params):
#         warnings.warn("Hyperparameters not logged", UserWarning)

#     @property
#     def name(self):
#         return "TrackioLogger"

#     @property
#     def version(self):
#         return "0.0.1"