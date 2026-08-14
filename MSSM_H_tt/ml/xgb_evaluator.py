# coding: utf-8
from __future__ import annotations
import os
from dataclasses import dataclass
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection, wait

from columnflow.types import Any


STOP_SIGNAL = "STOP"
ERROR_SIGNAL = "__XGB_EVALUATOR_ERROR__"


class XGBEvaluator:
    """
    XGBoost model evaluator running in a separate process.

    Multiple models can be registered. The subprocess loads all registered
    models once and then waits for evaluation requests through multiprocessing
    pipes.
    """

    @dataclass
    class Model:
        name: str
        path: str
        pipe: Connection | None = None
        signature_key: str = ""

    def __init__(self) -> None:
        self._models: dict[str, XGBEvaluator.Model] = {}
        self._p: Process | None = None
        self.silent = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.evaluate(*args, **kwargs)

    @property
    def running(self) -> bool:
        return self._p is not None and self._p.is_alive()

    def add_model(
        self,
        name: str,
        path: str,
        signature_key: str = "",
    ) -> None:
        if self._p is not None:
            raise ValueError("cannot add models while evaluator process exists")

        if name in self._models:
            raise ValueError(
                f"model with name '{name}' already exists",
            )

        # Normalize path.
        path = str(path)
        path = os.path.expandvars(
            os.path.expanduser(path),
        )
        path = os.path.abspath(path)

        self._models[name] = XGBEvaluator.Model(
            name=name,
            path=path,
            signature_key=signature_key,
        )

    def start(self) -> None:
        if self._p is not None:
            if self._p.is_alive():
                raise ValueError("process already started")

            # Clean up a stale process object.
            self.stop()

        config = []
        child_pipes = []

        for model in self._models.values():
            parent_pipe, child_pipe = Pipe()

            model.pipe = parent_pipe
            child_pipes.append(child_pipe)

            config.append(
                {
                    "name": model.name,
                    "path": model.path,
                    "pipe": child_pipe,
                    "signature_key": model.signature_key,
                },
            )

        self._p = Process(
            target=_xgb_evaluate,
            args=(config,),
            kwargs={
                "silent": self.silent,
            },
        )

        self._p.start()

        # The parent process only uses the parent ends of the pipes.
        # Closing these copies is important so that EOF is propagated
        # correctly if the child process exits.
        for child_pipe in child_pipes:
            child_pipe.close()

    def evaluate(
        self,
        name: str,
        *args,
        **kwargs,
    ) -> Any:
        if self._p is None:
            raise RuntimeError(
                "XGB evaluator process has not been started",
            )

        if not self._p.is_alive():
            exitcode = self._p.exitcode

            raise RuntimeError(
                "XGB evaluator subprocess is not alive "
                f"(exit code {exitcode})",
            )

        if name not in self._models:
            raise ValueError(
                f"model with name '{name}' does not exist",
            )

        model = self._models[name]

        if model.pipe is None:
            raise RuntimeError(
                f"pipe for model '{name}' is not available",
            )

        try:
            model.pipe.send(
                (
                    args,
                    kwargs,
                ),
            )

            result = model.pipe.recv()

        except (
            BrokenPipeError,
            EOFError,
            OSError,
        ) as exc:
            exitcode = (
                self._p.exitcode
                if self._p is not None
                else None
            )

            raise RuntimeError(
                f"communication with XGB evaluator failed for "
                f"model '{name}' "
                f"(subprocess exit code {exitcode})",
            ) from exc

        # Propagate exceptions raised inside the evaluator subprocess.
        if (
            isinstance(result, tuple)
            and len(result) == 3
            and result[0] == ERROR_SIGNAL
        ):
            _, error_message, error_traceback = result

            raise RuntimeError(
                f"XGB evaluation failed for model '{name}':\n"
                f"{error_message}\n\n"
                f"{error_traceback}",
            )

        return result

    def stop(
        self,
        timeout: float = 5,
    ) -> None:
        process = self._p

        # Signal all models to stop when the subprocess is alive.
        if process is not None and process.is_alive():
            for model in self._models.values():
                if model.pipe is None:
                    continue

                try:
                    model.pipe.send(STOP_SIGNAL)

                except (
                    BrokenPipeError,
                    EOFError,
                    OSError,
                ):
                    # The subprocess might already have closed this pipe.
                    pass

        # Close all parent-side pipes.
        for model in self._models.values():
            if model.pipe is None:
                continue

            try:
                model.pipe.close()

            except OSError:
                pass

            model.pipe = None

        if process is None:
            return

        # Wait for graceful termination.
        if process.is_alive():
            process.join(timeout)

        # Force termination when necessary.
        if process.is_alive():
            process.kill()
            process.join()

        self._p = None


def _xgb_evaluate(
    config: list[dict[str, Any]],
    *,
    silent: bool = False,
) -> None:
    """
    Worker process for XGBEvaluator.

    Models are loaded exactly once when this process starts. The worker then
    blocks in multiprocessing.connection.wait() until one of the registered
    models receives either an evaluation request or a stop signal.
    """

    import traceback

    import numpy as np
    import xgboost as xgb

    _print = (
        (lambda *args, **kwargs: None)
        if silent
        else print
    )

    @dataclass
    class Model:
        name: str
        path: str
        pipe: Connection
        signature_key: str = ""
        model: Any = None

        @classmethod
        def from_config(
            cls,
            model_config: dict[str, Any],
        ):
            for attr in (
                "name",
                "path",
                "pipe",
            ):
                if attr not in model_config:
                    raise ValueError(
                        f"missing field '{attr}' in model config",
                    )

            path = model_config["path"]

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"model file '{path}' does not exist",
                )

            pipe = model_config["pipe"]

            if not isinstance(pipe, Connection):
                raise TypeError(
                    f"'pipe' {pipe} is not of type "
                    f"'{Connection}'",
                )

            return cls(
                name=model_config["name"],
                path=path,
                pipe=pipe,
                signature_key=model_config.get(
                    "signature_key",
                    "",
                ),
            )

        def load(self) -> None:
            signature_message = (
                f" (signature '{self.signature_key}')"
                if self.signature_key
                else ""
            )

            _print(
                f"loading model '{self.name}'"
                f"{signature_message} "
                f"from {self.path} ...",
            )

            self.model = xgb.XGBClassifier()
            self.model.load_model(self.path)

            _print("done")

        def evaluate(
            self,
            *args,
            **kwargs,
        ) -> np.ndarray:
            return self.model.predict_proba(
                *args,
                **kwargs,
            )

        def clear(self) -> None:
            _print(
                f"clearing model '{self.name}'",
            )

            self.model = None

            try:
                self.pipe.close()

            except OSError:
                pass

    # ------------------------------------------------------------------
    # Build and load models
    # ------------------------------------------------------------------

    models = [
        Model.from_config(model_config)
        for model_config in config
    ]

    try:
        for model in models:
            model.load()

    except Exception:
        # Make sure every pipe is closed if model loading fails.
        for model in models:
            try:
                model.pipe.close()
            except OSError:
                pass

        raise

    # Map each pipe directly to its model. This allows wait() to block until
    # any model has work available without polling or sleeping.
    active_models = {
        model.pipe: model
        for model in models
    }

    def remove_model(
        pipe: Connection,
    ) -> None:
        model = active_models.pop(
            pipe,
            None,
        )

        if model is not None:
            model.clear()

    def shutdown() -> None:
        for pipe in list(active_models):
            remove_model(pipe)

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    try:
        while active_models:
            ready_pipes = wait(
                list(active_models),
            )

            for pipe in ready_pipes:
                if pipe not in active_models:
                    continue

                model = active_models[pipe]

                try:
                    data = pipe.recv()

                except (
                    EOFError,
                    OSError,
                ):
                    remove_model(pipe)
                    continue

                # ------------------------------------------------------
                # Evaluation request
                # ------------------------------------------------------

                if (
                    isinstance(data, tuple)
                    and len(data) == 2
                ):
                    args, kwargs = data

                    try:
                        result = model.evaluate(
                            *args,
                            **kwargs,
                        )

                    except Exception as exc:
                        error_traceback = traceback.format_exc()

                        try:
                            pipe.send(
                                (
                                    ERROR_SIGNAL,
                                    repr(exc),
                                    error_traceback,
                                ),
                            )

                        except (
                            BrokenPipeError,
                            EOFError,
                            OSError,
                        ):
                            pass

                        # Evaluation errors are treated as fatal since the
                        # corresponding ProduceColumns task must fail.
                        raise

                    try:
                        pipe.send(result)

                    except (
                        BrokenPipeError,
                        EOFError,
                        OSError,
                    ):
                        remove_model(pipe)

                # ------------------------------------------------------
                # Stop request
                # ------------------------------------------------------

                elif data == STOP_SIGNAL:
                    remove_model(pipe)

                # ------------------------------------------------------
                # Invalid request
                # ------------------------------------------------------

                else:
                    raise ValueError(
                        f"received invalid data for model "
                        f"'{model.name}': {data}",
                    )

    finally:
        shutdown()