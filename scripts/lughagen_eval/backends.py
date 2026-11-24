"""Model backends.

A ``ModelBackend`` answers two questions for the runner: "what does the
model generate for this prompt" (``generate``) and, for multiple-choice
tasks, "which of these candidate continuations does the model prefer"
(``loglikelihood``). Everything else in the evaluation pipeline (prompt
building, JSONL logging, resumability, scoring, aggregation) is written
against this interface and does not care which backend is in use.

Two implementations are provided:

* ``APIBackend`` -- calls the existing LughaGen HTTP ``/generate`` endpoint.
  This is what the original script did. It supports ``generate`` today; it
  can support ``loglikelihood`` too, but only if the server exposes a
  logprob-returning endpoint (see ``score_endpoint`` below). Without one,
  ``loglik_choice`` tasks cannot be scored fairly through the API and the
  runner will raise a clear error telling you to add that endpoint or use
  ``LocalBackend`` instead.

* ``LocalBackend`` -- loads the model directly (via ``transformers``) on the
  machine it runs on. It can compute true per-token log-likelihoods, which
  is what makes fair multiple-choice scoring (Belebele, AfriMMLU, AfriXNLI)
  possible without depending on the model's ability to follow an "only
  output the letter" instruction.

Both raise ``BackendError`` on failure; the runner catches this and records
a ``status: "error"`` row rather than crashing the whole job.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

import requests

from .local_scoring import score_pairs, split_context_continuation
from .model_specs import parse_model_specs, resolve_model_spec
from .core_types import GenerationConfig, ModelOutput


class BackendError(RuntimeError):
    """Raised when a backend cannot produce a usable result."""


@runtime_checkable
class ModelBackend(Protocol):
    def generate(self, model: str, prompt: str, config: GenerationConfig) -> ModelOutput: ...

    def loglikelihood(
        self, model: str, prompt: str, continuations: list[str], config: GenerationConfig
    ) -> ModelOutput: ...


class APIBackend:
    """Calls LughaGen's HTTP ``/generate`` endpoint (and optionally a score endpoint)."""

    def __init__(self, endpoint: str, score_endpoint: str | None = None):
        self.endpoint = endpoint
        self.score_endpoint = score_endpoint

    def generate(self, model: str, prompt: str, config: GenerationConfig) -> ModelOutput:
        payload = {
            "model": model,
            "prompt": prompt,
            "language": config.api_language,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
        }
        last_error: Exception | None = None
        for attempt in range(config.retries + 1):
            try:
                start = time.monotonic()
                response = requests.post(self.endpoint, json=payload, timeout=config.timeout)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict) or not isinstance(data.get("generated_text"), str):
                    raise BackendError("API response did not contain a string 'generated_text'.")
                latency_ms = data.get("latency_ms")
                if not isinstance(latency_ms, (int, float)):
                    latency_ms = (time.monotonic() - start) * 1000
                return ModelOutput(generated_text=data["generated_text"], latency_ms=latency_ms, raw=data)
            except (requests.RequestException, ValueError, BackendError) as error:
                last_error = error
                if attempt == config.retries:
                    break
                time.sleep(min(2**attempt, 8))
        return ModelOutput(error=f"API request failed after {config.retries + 1} attempts: {last_error}")

    def loglikelihood(
        self, model: str, prompt: str, continuations: list[str], config: GenerationConfig
    ) -> ModelOutput:
        if self.score_endpoint is None:
            raise BackendError(
                "APIBackend has no score_endpoint configured, so it cannot compute "
                "log-likelihoods for multiple-choice tasks (Belebele / AfriMMLU / "
                "AfriXNLI). Either add a logprob-returning endpoint on the LughaGen "
                "server and pass --score-endpoint, or run with --backend local."
            )
        payload = {"model": model, "prompt": prompt, "continuations": continuations}
        try:
            response = requests.post(self.score_endpoint, json=payload, timeout=config.timeout)
            response.raise_for_status()
            data = response.json()
            logprobs = data.get("logprobs")
            if not isinstance(logprobs, list) or len(logprobs) != len(continuations):
                raise BackendError("Score endpoint did not return one logprob per continuation.")
            return ModelOutput(choice_logprobs=[float(v) for v in logprobs], raw=data)
        except (requests.RequestException, ValueError, BackendError) as error:
            return ModelOutput(error=f"Score endpoint request failed: {error}")


class LocalBackend:
    """Runs the model directly on this machine's GPU via ``transformers``.

    Built for a Google Colab session: checkpoints are read straight from the
    mounted Google Drive, loaded onto the GPU (optionally 8-bit/4-bit quantized
    so an 8B model fits a 15 GB T4), and only ONE model is kept in GPU memory
    at a time -- switching to the next model frees the previous one first.

    ``torch``/``transformers`` are imported in ``__init__`` so the rest of the
    package and its unit tests work without them installed.

    Choice scoring (Belebele / AfriMMLU / AfriXNLI) uses true log-likelihood.
    The choice is tokenized together with the prompt (with a leading space, as
    lm-evaluation-harness does) so the scored token is the one the model would
    really emit after "Answer:". When every choice is a single token (the A/B/C/D
    letters) a single forward pass scores all of them.
    """

    def __init__(self, model_paths: dict[str, Any], device: str = "cuda",
                 quantization: str = "none", dtype: str = "float16",
                 choice_delimiter: str = " "):
        if quantization not in ("none", "8bit", "4bit"):
            raise BackendError(f"quantization must be none, 8bit or 4bit, got {quantization!r}")
        try:
            import torch
            import transformers
        except ImportError as error:
            raise BackendError(
                "LocalBackend requires 'torch' and 'transformers' (plus 'accelerate', and "
                "'bitsandbytes' for 8bit/4bit). In Colab run: "
                "pip install -q transformers accelerate bitsandbytes peft"
            ) from error
        if quantization != "none" and not device.startswith("cuda"):
            raise BackendError("8bit/4bit quantization needs a CUDA GPU; use --quantization none on CPU.")
        try:
            self._specs = parse_model_specs(model_paths)
        except ValueError as error:
            raise BackendError(str(error)) from error
        self._torch = torch
        self._transformers = transformers
        self._device = device
        self._quantization = quantization
        self._dtype_name = dtype
        self._delimiter = choice_delimiter
        self._current_name: str | None = None
        self._net: Any = None
        self._tokenizer: Any = None
        self._load_errors: dict[str, str] = {}

    # -- model lifecycle -------------------------------------------------
    def _unload(self) -> None:
        import gc

        self._net = None
        self._tokenizer = None
        self._current_name = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _load(self, model: str) -> tuple[Any, Any]:
        if self._current_name == model:
            return self._net, self._tokenizer
        if model in self._load_errors:  # don't retry a multi-minute Drive load per example
            raise BackendError(self._load_errors[model])
        self._unload()
        try:
            if model not in self._specs:
                raise ValueError(f"no entry for model {model!r} in the models file")
            spec = resolve_model_spec(model, self._specs[model])
            tf, torch = self._transformers, self._torch
            dtype = getattr(torch, self._dtype_name) if self._dtype_name != "auto" else "auto"
            kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
            if self._device.startswith("cuda"):
                kwargs["device_map"] = {"": 0}
            if self._quantization == "4bit":
                kwargs["quantization_config"] = tf.BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16 if dtype == "auto" else dtype)
            elif self._quantization == "8bit":
                kwargs["quantization_config"] = tf.BitsAndBytesConfig(load_in_8bit=True)
            if "gemma" in model.lower():
                kwargs["attn_implementation"] = "eager"  # recommended for Gemma 2
            base_path = spec.base_path or spec.path
            try:
                tokenizer = tf.AutoTokenizer.from_pretrained(spec.path)
            except OSError:  # adapter folders often ship without a tokenizer
                tokenizer = tf.AutoTokenizer.from_pretrained(base_path)
            net = tf.AutoModelForCausalLM.from_pretrained(base_path, **kwargs)
            if spec.is_adapter:
                try:
                    from peft import PeftModel
                except ImportError as error:
                    raise RuntimeError("this model is a LoRA adapter; run: pip install -q peft") from error
                net = PeftModel.from_pretrained(net, spec.path)
            if not self._device.startswith("cuda"):
                net = net.to(self._device)
            net.eval()
        except Exception as error:  # noqa: BLE001 - any load failure becomes a clear, cached BackendError
            self._unload()
            message = f"Could not load model {model!r}: {error}"
            self._load_errors[model] = message
            raise BackendError(message) from error
        self._net, self._tokenizer, self._current_name = net, tokenizer, model
        return net, tokenizer

    def _free_after_error(self) -> None:
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # -- generation ------------------------------------------------------
    def generate(self, model: str, prompt: str, config: GenerationConfig) -> ModelOutput:
        torch = self._torch
        net, tokenizer = self._load(model)  # BackendError propagates; the runner records it
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(net.device)
            # Temperatures this low are "near greedy" requests (the API refuses 0);
            # locally we can simply be greedy, which is reproducible.
            greedy = config.temperature <= 0.05
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": config.max_new_tokens,
                "repetition_penalty": config.repetition_penalty,
                "do_sample": not greedy,
                "pad_token_id": pad_id,
            }
            if not greedy:
                gen_kwargs.update(temperature=config.temperature, top_p=config.top_p)
            start = time.monotonic()
            with torch.no_grad():
                output_ids = net.generate(**inputs, **gen_kwargs)
            latency_ms = (time.monotonic() - start) * 1000
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return ModelOutput(generated_text=text, latency_ms=latency_ms, raw={"backend": "local"})
        except Exception as error:  # noqa: BLE001 - e.g. CUDA OOM on one long prompt
            self._free_after_error()
            return ModelOutput(error=f"Local generation failed: {error}")

    # -- multiple-choice scoring -----------------------------------------
    def loglikelihood(
        self, model: str, prompt: str, continuations: list[str], config: GenerationConfig
    ) -> ModelOutput:
        torch = self._torch
        net, tokenizer = self._load(model)
        try:
            context_ids = tokenizer(prompt).input_ids
            pairs = [
                split_context_continuation(context_ids, tokenizer(prompt + self._delimiter + choice).input_ids)
                for choice in continuations
            ]

            def row_fn(ids: list[int]):
                batch = torch.tensor([ids], device=net.device)
                return torch.log_softmax(net(batch).logits[0].float(), dim=-1)

            start = time.monotonic()
            with torch.no_grad():
                scores = score_pairs(pairs, row_fn)
            latency_ms = (time.monotonic() - start) * 1000
            return ModelOutput(choice_logprobs=scores, latency_ms=latency_ms, raw={"backend": "local"})
        except Exception as error:  # noqa: BLE001
            self._free_after_error()
            return ModelOutput(error=f"Local log-likelihood scoring failed: {error}")


class FakeBackend:
    """Deterministic backend for tests: returns scripted responses.

    ``responses`` maps prompt substrings to either a generated string
    (for ``generate``) or a list of floats (for ``loglikelihood``). This
    lets tests exercise the runner/task/scoring pipeline end-to-end without
    any network or GPU dependency.
    """

    def __init__(self, generate_responses: dict[str, str] | None = None,
                 loglik_responses: dict[str, list[float]] | None = None,
                 default_generate: str = "",
                 fail_on: set[str] | None = None):
        self.generate_responses = generate_responses or {}
        self.loglik_responses = loglik_responses or {}
        self.default_generate = default_generate
        self.fail_on = fail_on or set()
        self.calls: list[tuple[str, str]] = []
        self.model_calls: list[str] = []  # model name per call, to check load ordering

    def _match(self, table: dict, prompt: str, default=None):
        for key, value in table.items():
            if key in prompt:
                return value
        return default

    def generate(self, model: str, prompt: str, config: GenerationConfig) -> ModelOutput:
        self.calls.append(("generate", prompt))
        self.model_calls.append(model)
        for needle in self.fail_on:
            if needle in prompt:
                return ModelOutput(error=f"simulated failure for prompt containing {needle!r}")
        text = self._match(self.generate_responses, prompt, self.default_generate)
        return ModelOutput(generated_text=text, latency_ms=1.0, raw={"backend": "fake"})

    def loglikelihood(
        self, model: str, prompt: str, continuations: list[str], config: GenerationConfig
    ) -> ModelOutput:
        self.calls.append(("loglikelihood", prompt))
        self.model_calls.append(model)
        for needle in self.fail_on:
            if needle in prompt:
                return ModelOutput(error=f"simulated failure for prompt containing {needle!r}")
        scores = self._match(self.loglik_responses, prompt)
        if scores is None:
            scores = [0.0] * len(continuations)
        return ModelOutput(choice_logprobs=list(scores), latency_ms=1.0, raw={"backend": "fake"})
