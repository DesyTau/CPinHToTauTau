# coding: utf-8

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from columnflow.types import Any


STOP_SIGNAL = "STOP"
EVALUATE_SIGNAL = "__XGB_EVALUATE__"
EVALUATE_MANY_SIGNAL = "__XGB_EVALUATE_MANY__"
ERROR_SIGNAL = "__XGB_EVALUATOR_ERROR__"


class XGBEvaluator:
    """
    XGBoost model evaluator running in a separate process.

    Multiple models can be registered. The subprocess loads all registered
    models once and then waits for evaluation requests through a single
    multiprocessing pipe.

    Two evaluation modes are supported:

      - evaluate(name, features):
          evaluate a single model

      - evaluate_many(names, features):
          serialize the features only once and evaluate all requested models
          on the same deserialized object in the worker process
    """

    @dataclass
    class Model:
        name: str
        path: str
        signature_key: str = ""

    def __init__(self) -> None:
        self._models: dict[str, XGBEvaluator.Model] = {}
        self._p: Process | None = None
        self._pipe: Connection | None = None
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
            raise ValueError(
                "cannot add models while evaluator process exists",
            )

        if name in self._models:
            raise ValueError(
                f"model with name '{name}' already exists",
            )

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

            self.stop()

        parent_pipe, child_pipe = Pipe()

        self._pipe = parent_pipe

        config = [
            {
                "name": model.name,
                "path": model.path,
                "signature_key": model.signature_key,
            }
            for model in self._models.values()
        ]

        self._p = Process(
            target=_xgb_evaluate,
            args=(
                config,
                child_pipe,
            ),
            kwargs={
                "silent": self.silent,
            },
        )

        self._p.start()

        # Only the subprocess uses the child side.
        child_pipe.close()

    def _check_running(self) -> None:
        if self._p is None:
            raise RuntimeError(
                "XGB evaluator process has not been started",
            )

        if not self._p.is_alive():
            raise RuntimeError(
                "XGB evaluator subprocess is not alive "
                f"(exit code {self._p.exitcode})",
            )

        if self._pipe is None:
            raise RuntimeError(
                "XGB evaluator communication pipe is not available",
            )

    def _request(
        self,
        request: Any,
        description: str,
    ) -> Any:
        self._check_running()

        try:
            self._pipe.send(request)
            result = self._pipe.recv()

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
                f"{description} "
                f"(subprocess exit code {exitcode})",
            ) from exc

        # Propagate exceptions raised in the evaluator subprocess.
        if (
            isinstance(result, tuple)
            and len(result) == 3
            and result[0] == ERROR_SIGNAL
        ):
            _, error_message, error_traceback = result

            raise RuntimeError(
                f"XGB evaluation failed for {description}:\n"
                f"{error_message}\n\n"
                f"{error_traceback}",
            )

        return result

    def evaluate(
        self,
        name: str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Evaluate a single registered model.
        """
        if name not in self._models:
            raise ValueError(
                f"model with name '{name}' does not exist",
            )

        return self._request(
            (
                EVALUATE_SIGNAL,
                name,
                args,
                kwargs,
            ),
            description=f"model '{name}'",
        )

    def evaluate_many(
        self,
        model_names: Iterable[str],
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate multiple registered models on the same input.

        The input arguments are serialized through the multiprocessing pipe
        exactly once. In the worker process, the same deserialized input object
        is passed sequentially to every requested model.
        """
        model_names = tuple(model_names)

        if not model_names:
            return {}

        if len(set(model_names)) != len(model_names):
            raise ValueError(
                "duplicate model names passed to evaluate_many",
            )

        missing = [
            name
            for name in model_names
            if name not in self._models
        ]

        if missing:
            raise ValueError(
                "unknown model names passed to evaluate_many: "
                + ", ".join(missing),
            )

        result = self._request(
            (
                EVALUATE_MANY_SIGNAL,
                model_names,
                args,
                kwargs,
            ),
            description=(
                f"{len(model_names)} models "
                f"[{', '.join(model_names)}]"
            ),
        )

        if not isinstance(result, dict):
            raise RuntimeError(
                "invalid result returned by evaluate_many: "
                f"expected dict, got {type(result)}",
            )

        return result

    def stop(
        self,
        timeout: float = 5,
    ) -> None:
        process = self._p
        pipe = self._pipe

        if (
            process is not None
            and process.is_alive()
            and pipe is not None
        ):
            try:
                pipe.send(STOP_SIGNAL)

            except (
                BrokenPipeError,
                EOFError,
                OSError,
            ):
                pass

        if pipe is not None:
            try:
                pipe.close()

            except OSError:
                pass

        self._pipe = None

        if process is None:
            return

        if process.is_alive():
            process.join(timeout)

        if process.is_alive():
            process.kill()
            process.join()

        self._p = None


def _xgb_evaluate(
    config: list[dict[str, Any]],
    pipe: Connection,
    *,
    silent: bool = False,
) -> None:
    """
    Worker process for XGBEvaluator.

    All models are loaded exactly once. Evaluation requests are received
    through one control pipe.

    For evaluate_many requests, the feature dataframe is deserialized once
    and reused for all requested models.
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

            return cls(
                name=model_config["name"],
                path=path,
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

            self.model = xgb.XGBClassifier(n_jobs=1)
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

    models = [
        Model.from_config(model_config)
        for model_config in config
    ]

    models_by_name = {
        model.name: model
        for model in models
    }

    try:
        # Load each model exactly once.
        for model in models:
            model.load()

        while True:
            try:
                data = pipe.recv()

            except (
                EOFError,
                OSError,
            ):
                break

            if data == STOP_SIGNAL:
                break

            try:
                if (
                    isinstance(data, tuple)
                    and len(data) == 4
                    and data[0] == EVALUATE_SIGNAL
                ):
                    _, model_name, args, kwargs = data

                    if model_name not in models_by_name:
                        raise ValueError(
                            f"unknown model '{model_name}'",
                        )

                    result = models_by_name[
                        model_name
                    ].evaluate(
                        *args,
                        **kwargs,
                    )

                elif (
                    isinstance(data, tuple)
                    and len(data) == 4
                    and data[0] == EVALUATE_MANY_SIGNAL
                ):
                    _, model_names, args, kwargs = data

                    missing = [
                        name
                        for name in model_names
                        if name not in models_by_name
                    ]

                    if missing:
                        raise ValueError(
                            "unknown models in batch request: "
                            + ", ".join(missing),
                        )

                    # IMPORTANT:
                    #
                    # args/kwargs have been deserialized only once above.
                    # Every model evaluates the exact same input object.
                    result = {
                        model_name: models_by_name[
                            model_name
                        ].evaluate(
                            *args,
                            **kwargs,
                        )
                        for model_name in model_names
                    }

                else:
                    raise ValueError(
                        f"received invalid XGB evaluator request: {data}",
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

                # A prediction error should fail ProduceColumns.
                raise

            try:
                pipe.send(result)

            except (
                BrokenPipeError,
                EOFError,
                OSError,
            ):
                break

    finally:
        for model in models:
            model.clear()

        try:
            pipe.close()

        except OSError:
            pass