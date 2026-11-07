"""Persistent, sequential training jobs owned by the local UI backend."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
import copy
import html
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import uuid

from ilaria_dataset_upload import AUDIO, validate_dataset_path
from ilaria_training_tools import experiment_path

TERMINAL = {'Completed', 'Failed', 'Stopped', 'Interrupted'}


class TrainingQueue:
    def __init__(self, root, runner, gate=None):
        self.root = Path(root).resolve()
        self.folder = self.root / '.ilaria-queue'
        self.path = self.folder / 'queue.json'
        self.runner, self.gate = runner, gate
        self.lock = threading.RLock()
        self.jobs = []
        self.thread = None
        self.pause_requested = False
        self.message = 'Add a training job, then click Start queue.'
        self.load_error = None
        self.last_save = 0.0
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding='utf-8'))
                if saved.get('schema') != 1 or not isinstance(saved.get('jobs'), list):
                    raise ValueError('Invalid queue format')
                self.jobs = saved['jobs']
                for saved_job in self.jobs:
                    if isinstance(saved_job, dict) and isinstance(saved_job.get('args'), list) and len(saved_job['args']) == 23:
                        saved_job['args'].extend(['HiFi-GAN','contentvec',2,'Epochs',10000,'Auto', 'Custom paths' if saved_job['args'][8] or saved_job['args'][9] else 'None (train from scratch)'])
                for job in self.jobs:
                    if (not isinstance(job, dict) or len(job.get('id', '')) != 32
                            or any(c not in '0123456789abcdef' for c in job['id'])
                            or not isinstance(job.get('args'), list) or len(job['args']) != 30
                            or job.get('status') not in TERMINAL | {'Pending', 'Running'}):
                        raise ValueError('Invalid queue job')
                    if job['status'] == 'Running':
                        job['status'] = 'Interrupted'
                        job['log'] = 'The backend closed during this job. Check the saved checkpoint, then Retry selected.'
                self.message = 'Saved queue loaded. Review jobs before starting; interrupted jobs are not restarted automatically.'
                self._save()
            except Exception as error:
                self.jobs = []
                self.load_error = 'Cannot read the saved queue. Keep .ilaria-queue/queue.json for recovery. ' + str(error)
                self.message = self.load_error

    def _save(self):
        self.folder.mkdir(exist_ok=True)
        temporary = self.folder / ('queue-' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                json.dump({'schema': 1, 'jobs': self.jobs}, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(temporary), str(self.path))
            self.last_save = time.monotonic()
        finally:
            if temporary.exists():
                temporary.unlink()

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def blocks_manual(self):
        return self.is_running() and threading.current_thread() is not self.thread

    def add(self, args, preparation, auto_prepare=True):
        """Freeze controls, source audio and optional preview before enqueuing."""
        with self.lock:
            if self.load_error:
                raise ValueError(self.load_error)
            args = copy.deepcopy(list(args))
            if len(args) != 30:
                raise ValueError('Invalid training settings.')
            name = str(args[0] or '').strip()
            experiment_path(self.root, name)
            if any(job['name'].casefold() == name.casefold() and job['status'] in ('Pending', 'Running') for job in self.jobs):
                raise ValueError('This Model Name is already queued. Use a different name or remove the pending job.')
            source = Path(validate_dataset_path(args[22], self.root))
            self.folder.mkdir(exist_ok=True)
            identity = uuid.uuid4().hex
            snapshot = self.folder / identity
            snapshot.mkdir()
            try:
                audio = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in AUDIO and p.stat().st_size > 0), key=lambda p: p.name.casefold())
                total = sum(p.stat().st_size for p in audio)
                if shutil.disk_usage(self.folder).free < total + 64 * 1024**2:
                    raise ValueError('Not enough disk space to save a separate dataset snapshot for this job.')
                destination = snapshot / 'dataset'
                destination.mkdir()
                for path in audio:
                    before = path.stat()
                    with path.open('rb') as reader, (destination / path.name).open('xb') as writer:
                        shutil.copyfileobj(reader, writer, 4 * 1024**2)
                    after = path.stat()
                    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                        raise ValueError('The dataset changed while saving the queue job. Retry after the import finishes.')
                args[0], args[22] = name, str(destination)
                if args[15]:
                    preview = Path(str(args[16] or ''))
                    if not preview.is_file():
                        raise ValueError('Upload the preview clip before adding this job.')
                    target = snapshot / ('preview' + (preview.suffix or '.wav'))
                    shutil.copyfile(str(preview), str(target))
                    args[16] = str(target)
                else:
                    args[16] = None
                job = {'id': identity, 'name': name, 'args': args,
                       'preparation': copy.deepcopy(preparation), 'auto_prepare': bool(auto_prepare),
                       'status': 'Pending', 'log': 'Dataset and training settings saved.',
                       'created_at': time.strftime('%d-%m-%Y %H:%M:%S')}
                self.jobs.append(job)
                try:
                    self._save()
                except Exception:
                    self.jobs.remove(job)
                    raise
                self.message = 'Added %s. Its dataset and controls are saved separately.' % name
                return job['id']
            except Exception:
                if snapshot.resolve().parent == self.folder.resolve() and snapshot.exists():
                    shutil.rmtree(snapshot)
                raise

    def start(self):
        with self.lock:
            if self.load_error:
                raise ValueError(self.load_error)
            if self.is_running():
                return 'The queue is already running.'
            if not any(job['status'] == 'Pending' for job in self.jobs):
                return 'No pending jobs. Add a job or use Retry selected.'
            self.pause_requested = False
            if self.gate:
                self.gate.enter('training queue')
            self.thread = threading.Thread(target=self._run, name='Ilaria training queue', daemon=True)
            try:
                self.thread.start()
            except Exception:
                if self.gate:
                    self.gate.leave('training queue')
                self.thread = None
                raise
            return 'Queue started. Keep the backend open; the browser may be closed.'

    def _run(self):
        try:
            while True:
                with self.lock:
                    if self.pause_requested:
                        self.message = 'Queue paused. Pending jobs are kept.'
                        break
                    job = next((j for j in self.jobs if j['status'] == 'Pending'), None)
                    if job is None:
                        self.message = 'Queue finished. No pending jobs remain.'
                        break
                    job['status'], job['log'] = 'Running', 'Starting job...'
                    self._save()
                try:
                    iterator = iter(self.runner(copy.deepcopy(job)))
                    while True:
                        try:
                            line = next(iterator)
                        except StopIteration as finished:
                            outcome = finished.value
                            break
                        with self.lock:
                            job['log'] = '\n'.join(str(line).splitlines()[-4:])[-2000:]
                            if time.monotonic() - self.last_save > 2:
                                self._save()
                    if outcome not in ('Completed', 'Stopped'):
                        raise ValueError('The job did not confirm successful training completion. Read the job log.')
                    with self.lock:
                        job['status'] = outcome
                        if outcome == 'Stopped':
                            self.pause_requested = True
                            self.message = 'Training saved. Queue paused; Retry selected resumes this model.'
                        self._save()
                except Exception as error:
                    with self.lock:
                        job['status'] = 'Failed'
                        job['log'] = str(error)[-2000:]
                        self.message = 'Queue paused after an error in %s. Fix the cause, then Retry selected.' % job['name']
                        self.pause_requested = True
                        self._save()
                with self.lock:
                    if self.pause_requested:
                        if job['status'] == 'Completed':
                            self.message = 'Queue paused. Pending jobs are kept.'
                        break
        except Exception as error:
            with self.lock:
                self.message = 'Queue stopped: ' + str(error)
        finally:
            if self.gate:
                self.gate.leave('training queue')

    def pause(self):
        with self.lock:
            self.pause_requested = True
            self.message = 'Pause requested: the current model will finish; no next job will start.' if self.is_running() else 'Queue is paused.'
            return self.message

    def change(self, identity, action):
        with self.lock:
            if self.load_error:
                raise ValueError(self.load_error)
            job = next((job for job in self.jobs if job['id'] == identity), None)
            if not job:
                raise ValueError('Select a queue job first.')
            if job['status'] == 'Running':
                raise ValueError('This job is running. Use Stop and save or Pause after current model.')
            if action == 'retry':
                if job['status'] not in {'Failed', 'Stopped', 'Interrupted'}:
                    raise ValueError('Only a failed, stopped or interrupted job can be retried.')
                if any(other is not job and other['name'].casefold() == job['name'].casefold()
                       and other['status'] in {'Pending', 'Running'} for other in self.jobs):
                    raise ValueError('This Model Name already has a pending or running job.')
                job['status'], job['log'] = 'Pending', 'Ready to retry using the saved settings and dataset.'
            elif action == 'remove':
                self.jobs.remove(job)
            else:
                raise ValueError('Unknown queue action.')
            self._save()
            if action == 'remove':
                snapshot = self.folder / identity
                if snapshot.exists() and snapshot.resolve().parent == self.folder.resolve():
                    shutil.rmtree(snapshot)
            self.message = 'Job %s: %s.' % (action, job['name'])
            return self.message

    def display(self, selected=None):
        with self.lock:
            def escape(value):
                return html.escape(str(value)).replace('|', '&#124;').replace('\n', ' ')
            rows = ['| Order | Model | Status | Target | Batch | Precision |', '| --- | --- | --- | --- | --- | --- |']
            choices = []
            for index, job in enumerate(self.jobs, 1):
                label = '%s | %s | %s' % (job['id'], job['name'], job['status'])
                choices.append(label)
                rows.append('| %s | %s | %s | %s | %s | %s |' % (index, escape(job['name']), job['status'], ('%s steps' % job['args'][27] if job['args'][26] == 'Steps' else '%s epochs' % job['args'][5]), job['args'][6], job['args'][14]))
            active = next((j for j in self.jobs if j['status'] == 'Running'), None)
            chosen_id = str(selected or '').split(' | ', 1)[0]
            chosen = next((j for j in self.jobs if j['id'] == chosen_id), None)
            job = chosen or active or (self.jobs[-1] if self.jobs else None)
            log = job['log'] if job else self.message
            selected_label = next((label for label in choices if label.startswith(chosen_id + ' | ')), None) if chosen_id else None
            return '\n'.join(rows), self.message + ('\n' + log if log != self.message else ''), choices, selected_label
