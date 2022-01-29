from segmentation_models_pytorch.losses import (DiceLoss, FocalLoss,
                                                JaccardLoss, LovaszLoss,
                                                MCCLoss, SoftBCEWithLogitsLoss,
                                                SoftCrossEntropyLoss,
                                                TverskyLoss)
# Read more here - https://smp.readthedocs.io/en/latest/losses.html
from torch.nn import (BCELoss, BCEWithLogitsLoss, CosineEmbeddingLoss,
                      CrossEntropyLoss, CTCLoss, GaussianNLLLoss,
                      HingeEmbeddingLoss, HuberLoss, KLDivLoss, L1Loss,
                      MarginRankingLoss, MSELoss, MultiLabelMarginLoss,
                      MultiLabelSoftMarginLoss, MultiMarginLoss, NLLLoss,
                      PoissonNLLLoss, SmoothL1Loss, SoftMarginLoss,
                      TripletMarginLoss, TripletMarginWithDistanceLoss)
# Read more here - https://pytorch.org/docs/stable/nn.html#loss-functions
