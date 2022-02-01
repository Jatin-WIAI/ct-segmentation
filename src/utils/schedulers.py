"""All Learning Rate Schedulers. Currently only support torch LR Schedulers.
"""

from torch.optim.lr_scheduler import (ChainedScheduler, ConstantLR,
                                      CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, CyclicLR,
                                      ExponentialLR, LambdaLR, LinearLR,
                                      MultiplicativeLR, MultiStepLR,
                                      OneCycleLR, ReduceLROnPlateau,
                                      SequentialLR, StepLR)
