# Copyright (c) 2022-2026 Jakub Więckowski & contributors

"""
Intuitionistic Fuzzy MCDA methods.

Imports are defensive so that unit tests for the base class / validator
can run even when algorithm subpackages (aras/, topsis/, …) are absent.
"""

from .base import IFMCDAMethod
from .validator import Validator

__all__ = [
    "IFMCDAMethod",
    "Validator",
]

def _safe_import(name: str, attr: str):
    try:
        module = __import__(f"pyifdm.methods.{name}", fromlist=[attr])
        globals()[attr] = getattr(module, attr)
        __all__.append(attr)
    except ImportError:
        pass

_safe_import("if_aras", "ifARAS")
_safe_import("if_codas", "ifCODAS")
_safe_import("if_copras", "ifCOPRAS")
_safe_import("if_edas", "ifEDAS")
_safe_import("if_mabac", "ifMABAC")
_safe_import("if_mairca", "ifMAIRCA")
_safe_import("if_marcos", "ifMARCOS")
_safe_import("if_moora", "ifMOORA")
_safe_import("if_ocra", "ifOCRA")
_safe_import("if_topsis", "ifTOPSIS")
_safe_import("if_vikor", "ifVIKOR")
_safe_import("if_waspas", "ifWASPAS")
_safe_import("if_wpm", "ifWPM")
_safe_import("if_wsm", "ifWSM")
