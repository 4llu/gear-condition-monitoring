import logging
import platform

import torch

log = logging.getLogger("gear-cm")


def setup_model(config, device):

    # Model selection
    #################

    backbone = None
    if config["backbone"] == "MLP":
        from src.models.backbones.MLP import MLP

        backbone = MLP(config)
    elif config["backbone"] == "MlpMixer":
        from src.models.backbones.MlpMixer import MlpMixer

        backbone = MlpMixer(config)
    else:
        s = f"No such backbone name as: {config['backbone']}!"
        log.error(s)
        raise Exception(s)

    if config["model"] != "embedding":
        s = f"No such model name as: {config['model']}! Only 'embedding' is supported"
        log.error(s)
        raise Exception(s)

    from src.models.embedding import Embedding

    model = Embedding(backbone, config, device)

    # Move model to correct device
    model = model.to(device)

    # * torch.compile doesn't work on windows currently
    if device.type == "cuda" and platform.system() != "Windows":
        log.debug("")
        log.debug("#####################")
        log.debug("Using torch.compile()")
        log.debug("#####################")
        log.debug("")
        model = torch.compile(model)

    return model
