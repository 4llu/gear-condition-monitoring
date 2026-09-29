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

    model = None
    if config["model"] == "embedding":
        from src.models.embedding import Embedding

        model = Embedding(backbone, config, device)
    elif config["model"] == "prototypical":
        from src.models.prototypical import Prototypical

        model = Prototypical(backbone, config)
    elif config["model"] == "classical":
        from src.models.classical import Classical

        model = Classical(backbone, config, device)
    else:
        s = f"No such model name as: {config['model']}!"
        log.error(s)
        raise Exception(s)

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

    # Print model
    # FIXME
    # rows = []
    # t_params = 0
    # for name, parameter in model.named_parameters():
    #     if not parameter.requires_grad:
    #         continue

    #     param = parameter.numel()
    #     rows.append([name, param])
    #     t_params += param

    # for r in rows:
    #     log.debug("{:<35} {:<10}".format(r[0], r[1]))
    # log.debug("Total parameters:", t_params)
    # quit()

    return model
