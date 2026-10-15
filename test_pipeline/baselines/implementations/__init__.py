"""
Baseline implementations.

Each baseline provides concrete implementations of all four hook types.
The 'ours' baseline serves as the reference implementation, extracting
the current inference logic from BatchProcessor into hook form.
"""

# Import implementations to trigger registration
from . import ours

# Future baselines:
# from . import egoego
# from . import egoallo
# from . import rohm
# from . import pip
