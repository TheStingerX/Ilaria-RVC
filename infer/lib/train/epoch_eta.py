"""Measured epoch ETA, independent of the training/GPU libraries."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from time import perf_counter
from typing import Optional


def local_timezone():
    # Read the operating system timezone, including its daylight-saving rules.
    # tzwinlocal reads Windows settings directly, without an IANA database.
    import sys
    if sys.platform == "win32":
        from dateutil.tz import tzwinlocal
        return tzwinlocal()
    from dateutil.tz import tzlocal
    return tzlocal()


def validate_counts(total_epochs, completed_epochs):
    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int) or total_epochs <= 0:
        raise ValueError("total_epochs must be a positive integer")
    if isinstance(completed_epochs, bool) or not isinstance(completed_epochs, int):
        raise ValueError("completed_epochs must be an integer")
    if not 0 <= completed_epochs <= total_epochs:
        raise ValueError("completed_epochs must be between zero and total_epochs")


def format_duration(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Duration must be finite and nonnegative")
    days, remainder = divmod(int(round(seconds)), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days} days, {clock}" if days else clock


@dataclass(frozen=True)
class TrainingEstimate:
    completed_epochs: int
    remaining_epochs: int
    progress_percentage: float
    seconds_per_epoch: float
    remaining_seconds: float
    estimated_end_time: Optional[datetime]
    is_training_active: bool


def estimate_training(total_epochs, completed_epochs, seconds_per_epoch,
                      observation_timestamp, elapsed_in_current_epoch=0.0,
                      is_training_active=True):
    validate_counts(total_epochs, completed_epochs)
    seconds_per_epoch = float(seconds_per_epoch)
    elapsed = 0.0 if elapsed_in_current_epoch is None else float(elapsed_in_current_epoch)
    if not math.isfinite(seconds_per_epoch) or seconds_per_epoch <= 0:
        raise ValueError("seconds_per_epoch must be finite and positive")
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed_in_current_epoch must be finite and nonnegative")
    if not isinstance(observation_timestamp, datetime) or observation_timestamp.utcoffset() is None:
        raise ValueError("observation_timestamp must be timezone-aware")
    remaining_epochs = total_epochs - completed_epochs
    remaining = max(0.0, remaining_epochs * seconds_per_epoch - elapsed) if remaining_epochs else 0.0
    # UTC arithmetic measures actual elapsed seconds even across a DST boundary.
    estimated_end = None
    if is_training_active:
        estimated_end = (observation_timestamp.astimezone(timezone.utc)
                         + timedelta(seconds=remaining)).astimezone(local_timezone())
    return TrainingEstimate(completed_epochs, remaining_epochs,
                            completed_epochs / total_epochs * 100.0,
                            seconds_per_epoch, remaining, estimated_end,
                            bool(is_training_active))


class EpochETA:
    def __init__(self, total_epochs, completed_epochs=0, history_size=5, clock=perf_counter):
        validate_counts(total_epochs, completed_epochs)
        if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size <= 0:
            raise ValueError("history_size must be a positive integer")
        self.total_epochs = total_epochs
        self.completed_epochs = completed_epochs
        self.durations = deque(maxlen=history_size)
        self.last_epoch_seconds = None
        self._clock = clock
        self._epoch_start = None
        self._paused_at = None
        self._paused_seconds = 0.0
        self.is_training_active = True

    @property
    def seconds_per_epoch(self):
        if not self.durations:
            return None
        weights = [2 ** index for index in range(len(self.durations))]
        return sum(value * weight for value, weight in zip(self.durations, weights)) / sum(weights)

    def start_epoch(self):
        if self._epoch_start is not None:
            raise RuntimeError("An epoch is already running")
        self._epoch_start = self._clock()
        self._paused_seconds = 0.0
        self._paused_at = None
        self.is_training_active = True

    def pause(self):
        if self.is_training_active:
            self._paused_at = self._clock()
            self.is_training_active = False

    def resume(self):
        if not self.is_training_active:
            if self._paused_at is not None and self._epoch_start is not None:
                self._paused_seconds += self._clock() - self._paused_at
            self._paused_at = None
            self.is_training_active = True

    def elapsed_in_current_epoch(self):
        if self._epoch_start is None:
            return 0.0
        now = self._clock() if self.is_training_active else self._paused_at
        return max(0.0, now - self._epoch_start - self._paused_seconds)

    def finish_epoch(self, completed_epochs):
        validate_counts(self.total_epochs, completed_epochs)
        if self._epoch_start is None or not self.is_training_active:
            raise RuntimeError("No active epoch to finish")
        if completed_epochs < self.completed_epochs or (completed_epochs == self.completed_epochs and completed_epochs != self.total_epochs):
            raise ValueError("A newly completed epoch must advance progress")
        duration = self.elapsed_in_current_epoch()
        if duration <= 0 or not math.isfinite(duration):
            raise ValueError("Measured epoch duration must be finite and positive")
        self.durations.append(duration)
        self.last_epoch_seconds = duration
        self.completed_epochs = completed_epochs
        self._epoch_start = None
        self._paused_at = None
        self._paused_seconds = 0.0

    def snapshot(self, observation_timestamp=None):
        if self.seconds_per_epoch is None:
            return None
        return estimate_training(self.total_epochs, self.completed_epochs,
                                 self.seconds_per_epoch,
                                 observation_timestamp or datetime.now(timezone.utc),
                                 self.elapsed_in_current_epoch(), self.is_training_active)

    def summary(self, include_epoch_time=True, observation_timestamp=None):
        observed = observation_timestamp or datetime.now(timezone.utc)
        prefix = "Epoch" if include_epoch_time else "Completed"
        parts = [f"{prefix}: {self.completed_epochs}/{self.total_epochs}"]
        if include_epoch_time:
            parts[0] += " [" + observed.astimezone(local_timezone()).strftime("%d-%m-%Y %H:%M:%S") + "]"
        if include_epoch_time and self.last_epoch_seconds is not None:
            parts.append(f"Time: {format_duration(self.last_epoch_seconds)} ({self.last_epoch_seconds:.2f}s)")
        estimate = self.snapshot(observed)
        if estimate is None:
            parts.extend([f"Left: {self.total_epochs - self.completed_epochs}",
                          f"Progress: {self.completed_epochs / self.total_epochs * 100:.1f}%",
                          "ETA(est): measuring first epoch"])
        else:
            parts.extend([f"Avg({len(self.durations)}): {estimate.seconds_per_epoch:.2f}s",
                          f"Left: {estimate.remaining_epochs}",
                          f"Progress: {estimate.progress_percentage:.1f}%",
                          f"ETA(est): {format_duration(estimate.remaining_seconds)}"])
            if estimate.estimated_end_time is None:
                parts.append("Paused; ETA is active work time; End(est): unknown")
            else:
                parts.append("End(est): " + estimate.estimated_end_time.strftime("%d-%m-%Y %H:%M:%S"))
        return " | ".join(parts)
