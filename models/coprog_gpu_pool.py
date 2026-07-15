"""Process-based multi-GPU training pool for the COPROG co-training algorithm.

COPROG performs many *independent* from-scratch trainings (initial fit of the two
models, one temporary model per candidate suspension unit during the search, and the
end-of-iteration retrain). This module lets those independent trainings run
concurrently, each pinned to a specific physical GPU, using one persistent worker
process per GPU.

Why processes (and not threads): running several PyTorch-Lightning ``Trainer.fit``
calls at once on different GPUs is not thread-safe, so we isolate each GPU in its own
process. Each worker sets ``CUDA_VISIBLE_DEVICES`` to its assigned physical GPU as its
very first action (before any CUDA context is created), so inside the worker the GPU is
always seen as ``cuda:0``.

The single training primitive is :func:`run_training_job`, a module-level (hence
picklable) function used both by the workers *and* by the inline (single-GPU / auto)
path in :class:`~models.Coprog.Coprog`, so there is exactly one training code path.
"""

import os
import queue as _queue
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.multiprocessing as mp
from lightning import LightningModule, Trainer
from lightning.pytorch import callbacks
from torch.utils.data import DataLoader, TensorDataset

# Sentinel used to tell a worker to exit its loop.
_SHUTDOWN = "__shutdown__"


@dataclass
class TrainingSpec:
    """Fully self-contained description of one training job.

    All fields are picklable (CPU tensors / plain data / a module-level callable or
    ``functools.partial``) so a spec can cross a process boundary.

    :param module_builder: Picklable callable returning a *fresh* ``LightningModule``
        (a module-level function or ``functools.partial`` — never a lambda/closure).
    :param initial_state_dict: CPU ``state_dict`` loaded into the freshly built module
        before training, so every from-scratch training starts from identical weights.
    :param max_epochs: Maximum training epochs.
    :param patience: ``EarlyStopping`` patience (monitors ``val_loss``).
    :param batch_size: Training/validation batch size.
    :param shuffle: Whether to shuffle the training ``DataLoader``.
    :param train_x: Training features (CPU tensor).
    :param train_y: Training targets (CPU tensor).
    :param val_x: Optional validation features for early stopping / checkpointing.
    :param val_y: Optional validation targets.
    :param eval_x: Optional evaluation features; if given, the trained model's
        summed squared error on ``(eval_x, eval_y)`` is returned as ``"sse"``.
    :param eval_y: Optional evaluation targets (required iff ``eval_x`` is given).
    :param return_state: If ``True``, the trained CPU ``state_dict`` is returned as
        ``"state_dict"``.
    :param accelerator: ``Trainer`` accelerator (``"gpu"`` in workers, ``"auto"`` or
        ``"gpu"`` for the inline path).
    :param devices: ``Trainer`` devices argument (``1`` in workers, ``[gpu_id]`` for a
        pinned inline single-GPU run, ``None`` to let Lightning decide in auto mode).
    """

    module_builder: Callable[[], LightningModule]
    initial_state_dict: dict[str, torch.Tensor]
    max_epochs: int
    patience: int
    batch_size: int
    shuffle: bool
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor | None = None
    val_y: torch.Tensor | None = None
    eval_x: torch.Tensor | None = None
    eval_y: torch.Tensor | None = None
    return_state: bool = False
    accelerator: str = "auto"
    devices: Any = None


@dataclass
class CandidateContext:
    """Static, per-(model, iteration) data cached on a worker for candidate scoring.

    During the candidate search a model trains one temporary model per candidate
    suspension unit; every one of those trainings shares the same labelled set ``L``,
    validation set, builder and trainer config. Sending that (potentially large) data
    once per iteration — instead of once per candidate — avoids re-serialising it for
    every candidate. Each candidate job then only carries the small ``xu`` +
    pseudo-label ``lu_p`` (see :meth:`GpuTrainingPool.submit_candidate`).
    """

    module_builder: Callable[[], LightningModule]
    initial_state_dict: dict[str, torch.Tensor]
    max_epochs: int
    patience: int
    batch_size: int
    shuffle: bool
    labelled_x: torch.Tensor
    labelled_y: torch.Tensor
    val_x: torch.Tensor | None = None
    val_y: torch.Tensor | None = None


def _augment_labels(labelled_y: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
    """Reshape ``pseudo_labels`` to match ``labelled_y`` and concatenate the two.

    Mirrors the label reshaping used in :meth:`models.Coprog.Coprog.train` so the
    augmented target set is built identically whether the training runs inline or in a
    worker.

    :param labelled_y: Labelled targets ``L_y`` (shape ``(N,)`` or ``(N, 1)``).
    :param pseudo_labels: Model predictions for the candidate unit's sequences.
    :return: Concatenation of ``labelled_y`` and the reshaped pseudo-labels.
    """
    if labelled_y.dim() > 1:
        reshaped = pseudo_labels.view(-1, labelled_y.shape[1])
    else:
        reshaped = pseudo_labels.view(-1)
    return torch.cat([labelled_y, reshaped], dim=0)


def _summed_squared_error(model: LightningModule, x: torch.Tensor, y: torch.Tensor) -> float:
    """Summed squared error of ``model`` on ``(x, y)``, matching COPROG's confidence measure.

    :param model: Trained ``LightningModule`` (``forward`` returns real-unit predictions).
    :param x: Evaluation features.
    :param y: Evaluation targets.
    :return: ``sum((y - pred) ** 2)`` as a Python float.
    """
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        pred = model(x.to(device)).view(-1)
        y_flat = y.view(-1).to(device)
        return ((y_flat - pred) ** 2).sum().item()


def run_training_job(spec: TrainingSpec) -> dict[str, Any]:
    """Train one model from scratch and return the requested artefacts.

    Builds a fresh module via ``spec.module_builder()``, loads ``spec.initial_state_dict``
    (so every training starts from the same initial weights), fits it with a fresh
    ``Trainer`` (``EarlyStopping`` + ``ModelCheckpoint`` on ``val_loss`` in a throwaway
    temp directory), reloads the best checkpoint, and optionally evaluates / exports
    weights. The temp checkpoint directory is always removed.

    This function is module-level and picklable so it runs unchanged both in a worker
    process and inline in the main process.

    :param spec: The job description.
    :return: A dict possibly containing ``"sse"`` (if ``spec.eval_x`` was given) and
        ``"state_dict"`` (CPU tensors, if ``spec.return_state`` was set).
    """
    model = spec.module_builder()
    model.load_state_dict(spec.initial_state_dict)

    train_loader = DataLoader(
        TensorDataset(spec.train_x, spec.train_y),
        batch_size=spec.batch_size,
        shuffle=spec.shuffle,
    )
    val_loader = None
    if spec.val_x is not None and spec.val_y is not None:
        val_loader = DataLoader(TensorDataset(spec.val_x, spec.val_y), batch_size=spec.batch_size)

    ckpt_dir = tempfile.mkdtemp(prefix="coprog_ckpt_")
    try:
        early_stop_callback = callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=spec.patience,
            verbose=False,
            mode="min",
        )
        checkpoint_callback = callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_loss",
            filename="best-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
        )

        trainer_kwargs: dict[str, Any] = dict(
            default_root_dir=ckpt_dir,
            accelerator=spec.accelerator,
            max_epochs=spec.max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        if spec.devices is not None:
            trainer_kwargs["devices"] = spec.devices

        trainer = Trainer(**trainer_kwargs)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Reload the best checkpoint (best monitored val_loss) instead of last-epoch weights.
        best_model_path = getattr(checkpoint_callback, "best_model_path", "")
        if best_model_path:
            device = next(model.parameters()).device
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])

        result: dict[str, Any] = {}
        if spec.eval_x is not None and spec.eval_y is not None:
            result["sse"] = _summed_squared_error(model, spec.eval_x, spec.eval_y)
        if spec.return_state:
            result["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return result
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def _worker_loop(
    gpu_id: int,
    num_threads: int,
    in_queue: "mp.Queue",
    out_queue: "mp.Queue",
) -> None:
    """Entry point of a per-GPU worker process.

    Pins the process to ``gpu_id`` via ``CUDA_VISIBLE_DEVICES`` (as the very first
    action, before any CUDA context is created), then serves training messages until it
    receives the shutdown sentinel. Any exception while handling a job is captured and
    reported on ``out_queue`` so the main process can re-raise it instead of hanging.

    Message protocol (tuples read from ``in_queue``):
      * ``("job", job_id, spec)`` — run :func:`run_training_job` on ``spec``.
      * ``("ctx", ctx_id, context)`` — cache ``context`` (a :class:`CandidateContext`).
      * ``("candidate", job_id, ctx_id, xu, lu_p)`` — build the augmented set from the
        cached context + ``xu``/``lu_p`` and score the resulting model (returns ``sse``).
      * ``("clear_ctx", ctx_id)`` — drop a cached context to free memory.
      * ``_SHUTDOWN`` — exit.

    :param gpu_id: Physical GPU id this worker is pinned to.
    :param num_threads: CPU thread cap for this worker (avoids oversubscription).
    :param in_queue: Queue this worker reads messages from.
    :param out_queue: Shared queue results/errors are written to.
    """
    # Pin to the assigned physical GPU BEFORE any CUDA op. Importing torch does not
    # initialise CUDA; the first tensor/Trainer on GPU does, and by then only this
    # device is visible (as cuda:0).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass

    contexts: dict[str, CandidateContext] = {}

    while True:
        message = in_queue.get()
        if message == _SHUTDOWN:
            break

        kind = message[0]
        if kind == "ctx":
            _, ctx_id, context = message
            contexts[ctx_id] = context
            continue
        if kind == "clear_ctx":
            _, ctx_id = message
            contexts.pop(ctx_id, None)
            continue

        # Training messages carry a job_id we must always answer (result or error).
        job_id = message[1]
        try:
            if kind == "job":
                _, _, spec = message
                result = run_training_job(spec)
            elif kind == "candidate":
                _, _, ctx_id, xu, lu_p = message
                ctx = contexts[ctx_id]
                spec = TrainingSpec(
                    module_builder=ctx.module_builder,
                    initial_state_dict=ctx.initial_state_dict,
                    max_epochs=ctx.max_epochs,
                    patience=ctx.patience,
                    batch_size=ctx.batch_size,
                    shuffle=ctx.shuffle,
                    train_x=torch.cat([ctx.labelled_x, xu], dim=0),
                    train_y=_augment_labels(ctx.labelled_y, lu_p),
                    val_x=ctx.val_x,
                    val_y=ctx.val_y,
                    eval_x=ctx.labelled_x,
                    eval_y=ctx.labelled_y,
                    return_state=False,
                    accelerator="gpu",
                    devices=1,
                )
                result = run_training_job(spec)
            else:
                raise ValueError(f"Unknown worker message kind: {kind!r}")
            out_queue.put(("result", job_id, result))
        except Exception:  # noqa: BLE001 - report any failure back to the parent
            out_queue.put(("error", job_id, traceback.format_exc()))


class GpuTrainingPool:
    """A persistent pool of one worker process per GPU for COPROG training jobs.

    Each GPU id in ``gpu_ids`` gets its own worker (pinned via ``CUDA_VISIBLE_DEVICES``)
    and its own input queue, so callers control exactly which GPU runs which job. Results
    are collected from a single shared queue keyed by job id.

    Typical use::

        pool = GpuTrainingPool([0, 1])
        pool.start()
        try:
            jid = pool.submit_job(gpu_id=0, spec=spec)
            results = pool.gather([jid])
        finally:
            pool.shutdown()

    :param gpu_ids: Physical GPU ids to spawn workers for.
    """

    def __init__(self, gpu_ids: list[int]) -> None:
        if not gpu_ids:
            raise ValueError("GpuTrainingPool requires at least one GPU id.")
        self.gpu_ids = list(gpu_ids)
        # CUDA requires the 'spawn' start method; it also works on Windows and Linux.
        self._ctx = mp.get_context("spawn")
        self._in_queues: dict[int, "mp.Queue"] = {}
        self._out_queue: "mp.Queue" = self._ctx.Queue()
        self._processes: dict[int, "mp.Process"] = {}
        self._job_counter = 0
        self._started = False

        cpu_count = os.cpu_count() or 4
        self._num_threads = max(1, cpu_count // len(self.gpu_ids))

    def start(self) -> None:
        """Spawn one worker process per GPU. Idempotent."""
        if self._started:
            return
        for gpu_id in self.gpu_ids:
            in_queue: "mp.Queue" = self._ctx.Queue()
            process = self._ctx.Process(
                target=_worker_loop,
                args=(gpu_id, self._num_threads, in_queue, self._out_queue),
                daemon=True,
            )
            process.start()
            self._in_queues[gpu_id] = in_queue
            self._processes[gpu_id] = process
        self._started = True

    def _next_job_id(self) -> int:
        self._job_counter += 1
        return self._job_counter

    def submit_job(self, gpu_id: int, spec: TrainingSpec) -> int:
        """Queue a full :class:`TrainingSpec` on ``gpu_id``'s worker.

        :param gpu_id: Target GPU id (must be in ``gpu_ids``).
        :param spec: The self-contained job description.
        :return: A unique job id to pass to :meth:`gather`.
        """
        job_id = self._next_job_id()
        self._in_queues[gpu_id].put(("job", job_id, spec))
        return job_id

    def set_context(self, gpu_ids: list[int], ctx_id: str, context: CandidateContext) -> None:
        """Cache ``context`` on every worker in ``gpu_ids`` for candidate scoring.

        :param gpu_ids: GPUs whose workers will run candidates against this context.
        :param ctx_id: Identifier referenced by :meth:`submit_candidate`.
        :param context: The static per-(model, iteration) data.
        """
        for gpu_id in gpu_ids:
            self._in_queues[gpu_id].put(("ctx", ctx_id, context))

    def clear_context(self, gpu_ids: list[int], ctx_id: str) -> None:
        """Drop a cached context from the given workers to free memory.

        :param gpu_ids: GPUs to clear the context from.
        :param ctx_id: The context identifier previously passed to :meth:`set_context`.
        """
        for gpu_id in gpu_ids:
            self._in_queues[gpu_id].put(("clear_ctx", ctx_id))

    def submit_candidate(
        self,
        gpu_id: int,
        ctx_id: str,
        xu: torch.Tensor,
        lu_p: torch.Tensor,
    ) -> int:
        """Queue a candidate-scoring job referencing a cached context.

        Only the small ``xu`` + pseudo-label ``lu_p`` cross the boundary; the labelled
        set and other static data come from the cached :class:`CandidateContext`.

        :param gpu_id: Target GPU id (its worker must already hold ``ctx_id``).
        :param ctx_id: The cached context identifier.
        :param xu: The candidate unit's suspension sequences (CPU tensor).
        :param lu_p: Pseudo-labels for ``xu`` produced by the current model (CPU tensor).
        :return: A unique job id to pass to :meth:`gather`.
        """
        job_id = self._next_job_id()
        self._in_queues[gpu_id].put(("candidate", job_id, ctx_id, xu, lu_p))
        return job_id

    def gather(self, job_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Block until all ``job_ids`` have results, returning them keyed by job id.

        :param job_ids: Job ids returned by :meth:`submit_job` / :meth:`submit_candidate`.
        :return: Mapping ``job_id -> result dict``.
        :raises RuntimeError: If any worker reported an exception (the worker traceback
            is included) or a worker died before answering.
        """
        pending = set(job_ids)
        results: dict[int, dict[str, Any]] = {}
        while pending:
            try:
                message = self._out_queue.get(timeout=1.0)
            except _queue.Empty:
                # Detect a dead worker so we fail fast instead of blocking forever.
                if any(not p.is_alive() for p in self._processes.values()):
                    raise RuntimeError(
                        "A GPU worker process died before returning all results "
                        f"(still waiting on job ids {sorted(pending)})."
                    )
                continue

            status, job_id, payload = message
            if job_id not in pending:
                continue
            if status == "error":
                raise RuntimeError(f"GPU worker job {job_id} failed:\n{payload}")
            results[job_id] = payload
            pending.discard(job_id)
        return results

    def shutdown(self) -> None:
        """Stop all workers and join them. Safe to call more than once."""
        if not self._started:
            return
        for gpu_id in self.gpu_ids:
            try:
                self._in_queues[gpu_id].put(_SHUTDOWN)
            except Exception:
                pass
        for process in self._processes.values():
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        self._processes.clear()
        self._in_queues.clear()
        self._started = False
