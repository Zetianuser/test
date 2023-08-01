import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    # 有额外超参，加入args中
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        # 将模型超参载入model_args
        model_args = model_spec['args']
    # model字典 key为模型名称sam，value为模型超参
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model
