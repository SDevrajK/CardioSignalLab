"""Composable processing pipeline for physiological signals.

Maintains an ordered list of ProcessingStep records that can be replayed,
serialized to JSON for session persistence, and reset to revert to raw signal.
"""
from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Callable

import numpy as np
from loguru import logger

from cardio_signal_lab.core.data_models import ProcessingStep


# Registry of processing operations (name -> callable)
_OPERATIONS: dict[str, Callable] = {}


def register_operation(name: str, func: Callable):
    """Register a processing operation by name.

    Args:
        name: Operation name (must be unique)
        func: Callable with signature (samples, sampling_rate, **params) -> samples
    """
    if name in _OPERATIONS:
        logger.warning(f"Overwriting registered operation: {name}")
    _OPERATIONS[name] = func
    logger.debug(f"Registered processing operation: {name}")


def get_operation(name: str) -> Callable:
    """Get a registered operation by name.

    Args:
        name: Operation name

    Returns:
        The registered callable

    Raises:
        KeyError: If operation not found
    """
    if name not in _OPERATIONS:
        raise KeyError(f"Unknown operation: {name}. Available: {list(_OPERATIONS.keys())}")
    return _OPERATIONS[name]


def list_operations() -> list[str]:
    """List all registered operation names."""
    return list(_OPERATIONS.keys())


class ProcessingPipeline:
    """Composable processing pipeline that replays ordered operations.

    Each step is recorded as a ProcessingStep (operation name + parameters dict).
    The pipeline can be applied to raw signal data, serialized for session
    persistence, and reset to revert to raw signal.
    """

    def __init__(self):
        self.steps: list[ProcessingStep] = []

    def add_step(self, operation: str, parameters: dict[str, Any] | None = None):
        """Append a processing step to the pipeline.

        Args:
            operation: Name of a registered operation
            parameters: Parameters dict passed to the operation

        Raises:
            KeyError: If operation is not registered
        """
        # Validate operation exists
        get_operation(operation)

        step = ProcessingStep(
            operation=operation,
            parameters=parameters or {},
            timestamp=time.time(),
        )
        self.steps.append(step)
        logger.info(f"Added pipeline step: {operation} (params: {parameters})")

    def apply(
        self, samples: np.ndarray, sampling_rate: float, *, skip_on_error: bool = False
    ) -> np.ndarray:
        """Apply all pipeline steps in order to signal data.

        Args:
            samples: Raw signal samples (1D array)
            sampling_rate: Signal sampling rate in Hz
            skip_on_error: If True, skip failed steps and continue; if False, raise on first error

        Returns:
            Processed signal samples (partial results if skip_on_error=True)

        Raises:
            Exception: If a step fails and skip_on_error=False
        """
        result = samples.copy()
        skipped_steps = []

        for i, step in enumerate(self.steps):
            try:
                func = get_operation(step.operation)
                result = func(result, sampling_rate, **step.parameters)
                logger.debug(
                    f"Pipeline step {i + 1}/{len(self.steps)}: {step.operation} applied"
                )
            except Exception as e:
                error_msg = (
                    f"Pipeline step {i + 1}/{len(self.steps)} failed: {step.operation} "
                    f"(params: {step.parameters}): {e}"
                )
                if skip_on_error:
                    logger.warning(f"{error_msg} - skipping and continuing")
                    skipped_steps.append((i + 1, step.operation, str(e)))
                else:
                    logger.error(error_msg)
                    raise

        if skipped_steps:
            logger.warning(
                f"Pipeline completed with {len(skipped_steps)} skipped steps: "
                f"{[s[1] for s in skipped_steps]}"
            )

        return result

    def reset(self):
        """Clear all processing steps."""
        n_steps = len(self.steps)
        self.steps.clear()
        logger.info(f"Pipeline reset ({n_steps} steps cleared)")

    def remove_last(self) -> ProcessingStep | None:
        """Remove and return the last step.

        Returns:
            The removed step, or None if pipeline was empty
        """
        if not self.steps:
            return None
        step = self.steps.pop()
        logger.info(f"Removed last pipeline step: {step.operation}")
        return step

    def serialize(self) -> dict[str, Any]:
        """Serialize pipeline to a JSON-compatible dict.

        Returns:
            Dict with 'steps' list of {operation, parameters, timestamp} dicts
        """
        return {
            "steps": [
                {
                    "operation": step.operation,
                    "parameters": step.parameters,
                    "timestamp": step.timestamp,
                }
                for step in self.steps
            ]
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ProcessingPipeline:
        """Reconstruct pipeline from serialized dict.

        Args:
            data: Dict from serialize()

        Returns:
            Reconstructed ProcessingPipeline

        Raises:
            KeyError: If any operation is not registered
        """
        pipeline = cls()
        for step_data in data.get("steps", []):
            operation = step_data["operation"]
            # Validate operation exists
            get_operation(operation)
            step = ProcessingStep(
                operation=operation,
                parameters=step_data.get("parameters", {}),
                timestamp=step_data.get("timestamp"),
            )
            pipeline.steps.append(step)

        logger.info(f"Deserialized pipeline with {len(pipeline.steps)} steps")
        return pipeline

    @property
    def num_steps(self) -> int:
        """Number of steps in the pipeline."""
        return len(self.steps)

    @property
    def is_empty(self) -> bool:
        """Whether the pipeline has no steps."""
        return len(self.steps) == 0

    def __repr__(self) -> str:
        steps_str = ", ".join(s.operation for s in self.steps)
        return f"ProcessingPipeline([{steps_str}])"
