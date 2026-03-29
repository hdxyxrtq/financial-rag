import importlib

try:
    importlib.import_module("src.evaluation")
except ImportError:
    pass
