"""Algorithm registry with optional dependencies loaded defensively."""

REGISTRY = {}

from .ea.vanilla_ea import VanillaEA

REGISTRY["ea"] = VanillaEA


def _register_optional(key, module_name, class_name):
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
    except (ImportError, ModuleNotFoundError):
        return
    REGISTRY[key] = cls


_register_optional("bo", "algorithm.bo.bo", "BO")
_register_optional("sa", "algorithm.sa.sa", "SA")
_register_optional("es", "algorithm.ea.es", "ES")
_register_optional("pso", "algorithm.ea.pso", "PSO")
_register_optional("rs", "algorithm.rs.rs", "RS")
_register_optional("saasbo", "algorithm.bo.saasbo", "SAASBO")
