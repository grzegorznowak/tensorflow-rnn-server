# import something sensible here ?


def loadModelFromPath(absolutePath, moduleName, nameOfTheFactoryMethod):
    import importlib.util
    spec = importlib.util.spec_from_file_location(moduleName, absolutePath)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    method_to_call = getattr(foo, nameOfTheFactoryMethod)
    return method_to_call()

#def loadModel(path):
#    # loads model from a path