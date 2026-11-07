"""Exact batch-step targets and measured ETA, independent of neural libraries."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
from datetime import datetime, timedelta, timezone
import math
from time import perf_counter


def step_target(hps):
    mode = getattr(hps.train, 'training_target', 'Epochs')
    if mode not in ('Epochs', 'Steps'):
        raise ValueError('Training Target must be Epochs or Steps.')
    if mode == 'Epochs':
        return None
    target = getattr(hps.train, 'total_steps', 0)
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError('Total Steps must be a positive whole number.')
    if hps.if_cache_data_in_gpu:
        raise ValueError('Cache data to GPU must be No when training by Total Steps, so partial epochs can resume correctly.')
    return target


def target_epoch_limit(hps, completed_steps, first_epoch, batches_per_epoch, batch_offset=0):
    target = step_target(hps)
    if batches_per_epoch < 1:
        raise ValueError('No training batches are available.')
    if batch_offset < 0 or batch_offset >= batches_per_epoch:
        raise ValueError('The saved partial-epoch batch cursor is incompatible with this dataset.')
    if target is None:
        if first_epoch > hps.total_epoch:
            raise ValueError('This model has already reached Total Epochs. Increase the target to continue.')
        return hps.total_epoch
    if completed_steps >= target:
        raise ValueError('This model has already reached Total Steps. Increase the target to continue.')
    return first_epoch - 1 + math.ceil((target - completed_steps + batch_offset) / batches_per_epoch)


def _duration(value):
    days, remainder = divmod(max(0, round(value)), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    result = '%02d:%02d:%02d' % (hours, minutes, seconds)
    return '%s days, %s' % (days, result) if days else result


class StepETA:
    def __init__(self, target, completed=0, clock=perf_counter):
        self.target, self.completed = target, completed
        self.clock = clock
        self.previous = clock()
        self.last_report = -math.inf
        self.seconds_per_step = None

    def update(self, completed, epoch, completed_epochs, force=False):
        now = self.clock()
        count = completed - self.completed
        if count > 0:
            measured = max(1e-9, now - self.previous) / count
            self.seconds_per_step = measured if self.seconds_per_step is None else .2 * measured + .8 * self.seconds_per_step
            self.completed, self.previous = completed, now
        if not force and completed < self.target and now - self.last_report < 1:
            return None
        self.last_report = now
        remaining = max(0, self.target - completed) * (self.seconds_per_step or 0)
        from infer.lib.train.epoch_eta import local_timezone
        finish = (datetime.now(timezone.utc) + timedelta(seconds=remaining)).astimezone(local_timezone())
        return ('STEP_PROGRESS! Total Steps: %s/%s | Epoch: %s | Completed Epochs: %s | Remaining: %s | Estimated finish: %s'
                % (completed, self.target, epoch, completed_epochs, _duration(remaining), finish.strftime('%d-%m-%Y %H:%M:%S')))
