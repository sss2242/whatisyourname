"""Staged pipeline sub-stage functions.

Each module contains per-model sub-stage functions that read from and
write to a PipelineState object.  Between sub-stages the state is
serialized to disk so execution can resume in a fresh process.
"""
