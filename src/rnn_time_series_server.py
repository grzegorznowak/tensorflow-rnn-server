# import something sensible here ?


def load_model_from_path(absolutePath, moduleName, nameOfTheFactoryMethod):
    import importlib.util
    spec = importlib.util.spec_from_file_location(moduleName, absolutePath)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    method_to_call = getattr(foo, nameOfTheFactoryMethod)
    return method_to_call()


