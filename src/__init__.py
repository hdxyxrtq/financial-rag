import contextlib
import importlib

with contextlib.suppress(ImportError):
    importlib.import_module("src.evaluation")
