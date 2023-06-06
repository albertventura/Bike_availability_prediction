from bike_availability.config.config import conf
import importlib

def get_model_from_config(config):
    #name of the file
    model_name = config.get_config('model_to_use')
    module_name = f"bike_availability.models.{model_name.lower()}"
    module = importlib.import_module(module_name)

    class_name = "".join(word.capitalize() for word in model_name.split("_"))
    ModelClass = getattr(module, class_name)
    model = ModelClass(config)

    return model.get_artifacts()

