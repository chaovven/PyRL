import torch as th


def args_sanity_check(config, _log):
    # set CUDA flags
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    return config
