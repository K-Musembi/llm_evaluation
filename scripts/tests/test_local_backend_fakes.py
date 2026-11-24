"""Exercise LocalBackend's real code paths with tiny fake torch/transformers/peft modules.

This does NOT prove the numerics of a real model (that needs a GPU; see the README's
manual smoke test). It does prove the logic around the model: kwargs passed when loading,
quantization, adapter handling, one-model-at-a-time memory, load-error caching, greedy vs
sampled generation, and choice scoring through the real tokenizer-splitting code.
"""
import contextlib
import re
import sys
import types

import pytest

from lughagen_eval.backends import BackendError, LocalBackend
from lughagen_eval.core_types import GenerationConfig


class FakeTensor:
    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        return (len(self.data), len(self.data[0]))

    def __getitem__(self, index):
        return self.data[index]

    def to(self, device):
        return self


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """GPT-style: each non-space character, with its preceding space attached, is one token."""

    pad_token_id = None
    eos_token_id = 2

    def __init__(self):
        self.vocab, self.reverse = {}, {}

    def _id(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab) + 10
            self.reverse[self.vocab[token]] = token
        return self.vocab[token]

    def __call__(self, text, return_tensors=None):
        ids = [1] + [self._id(t) for t in re.findall(r"\s?\S", text)]
        if return_tensors == "pt":
            return FakeBatch(input_ids=FakeTensor([ids]))
        return types.SimpleNamespace(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.reverse.get(i, "?") for i in ids)


class Row:
    def __init__(self, net, ids):
        self.net, self.ids = net, ids

    def __getitem__(self, token_id):
        last = self.net.tokenizer.reverse.get(self.ids[-1])
        wanted = self.net.favorite.get(last)
        return -0.1 if self.net.tokenizer.reverse.get(token_id) == wanted else -5.0


class Rows:
    def __init__(self, net, ids):
        self.net, self.ids = net, ids

    def float(self):
        return self

    def __getitem__(self, position):
        return Row(self.net, self.ids[: position + 1])


class FakeNet:
    device = "cuda:0"

    def __init__(self, tokenizer, favorite):
        self.tokenizer, self.favorite = tokenizer, favorite
        self.forward_calls = 0
        self.generate_kwargs = None

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, batch):
        self.forward_calls += 1
        return types.SimpleNamespace(logits=[Rows(self, batch.data[0])])

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        ids = kwargs["input_ids"].data[0]
        return FakeTensor([ids + [self.tokenizer._id("ok")]])


@pytest.fixture
def fakes(monkeypatch):
    state = types.SimpleNamespace(
        loads=[], tokenizer=FakeTokenizer(), favorite={":": " B"}, missing_models=set(),
        missing_tokenizers=set(), empty_cache_calls=0, peft_calls=[], nets=[])

    torch = types.ModuleType("torch")
    torch.float16, torch.bfloat16 = "float16", "bfloat16"
    torch.no_grad = lambda: contextlib.nullcontext()
    torch.tensor = lambda data, device=None: FakeTensor(data)
    torch.log_softmax = lambda rows, dim=-1: rows
    torch.cuda = types.SimpleNamespace(is_available=lambda: True,
                                       empty_cache=lambda: setattr(state, "empty_cache_calls", state.empty_cache_calls + 1))

    transformers = types.ModuleType("transformers")
    transformers.BitsAndBytesConfig = lambda **kw: types.SimpleNamespace(**kw)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            if path in state.missing_tokenizers:
                raise OSError("no tokenizer here")
            return state.tokenizer

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kwargs):
            state.loads.append((path, kwargs))
            if path in state.missing_models:
                raise OSError(f"cannot read {path}")
            net = FakeNet(state.tokenizer, state.favorite)
            state.nets.append(net)
            return net

    transformers.AutoTokenizer, transformers.AutoModelForCausalLM = AutoTokenizer, AutoModelForCausalLM

    peft = types.ModuleType("peft")

    class PeftModel:
        @staticmethod
        def from_pretrained(net, path):
            state.peft_calls.append(path)
            return net

    peft.PeftModel = PeftModel
    for name, module in (("torch", torch), ("transformers", transformers), ("peft", peft)):
        monkeypatch.setitem(sys.modules, name, module)
    return state


def make_backend(models=None, **kwargs):
    return LocalBackend(models or {"Llama-8B-FFT": "/drive/llama", "Gemma2-27B-CPT": "/drive/gemma"}, **kwargs)


def test_choice_scoring_picks_the_models_favourite_letter_with_one_forward_pass(fakes):
    backend = make_backend()
    output = backend.loglikelihood("Llama-8B-FFT", "Q: pick one\nAnswer:", ["A", "B", "C", "D"], GenerationConfig())
    assert output.ok
    scores = output.choice_logprobs
    assert scores.index(max(scores)) == 1  # " B" is favoured after "Answer:"
    assert scores[1] == pytest.approx(-0.1) and scores[0] == pytest.approx(-5.0)
    assert fakes.nets[0].forward_calls == 1  # all four letters scored from one pass over the prompt


def test_multi_token_choices_use_one_pass_per_choice(fakes):
    fakes.favorite = {":": " e"}
    backend = make_backend()
    output = backend.loglikelihood("Llama-8B-FFT", "Answer:", ["entailment", "neutral", "contradiction"], GenerationConfig())
    assert output.ok and len(output.choice_logprobs) == 3
    assert fakes.nets[0].forward_calls == 3


def test_greedy_generation_for_near_zero_temperature(fakes):
    backend = make_backend()
    output = backend.generate("Llama-8B-FFT", "Say ok:", GenerationConfig(temperature=0.01))
    assert output.ok and output.generated_text == "ok"
    kwargs = fakes.nets[0].generate_kwargs
    assert kwargs["do_sample"] is False and "temperature" not in kwargs
    assert kwargs["pad_token_id"] == 2  # falls back to eos when the tokenizer has no pad token


def test_sampling_when_a_real_temperature_is_requested(fakes):
    backend = make_backend()
    backend.generate("Llama-8B-FFT", "Say ok:", GenerationConfig(temperature=0.7, top_p=0.9))
    kwargs = fakes.nets[0].generate_kwargs
    assert kwargs["do_sample"] is True and kwargs["temperature"] == 0.7 and kwargs["top_p"] == 0.9


def test_loading_passes_quantization_dtype_and_device_map(fakes):
    backend = make_backend(quantization="4bit", dtype="float16")
    backend.generate("Llama-8B-FFT", "hi", GenerationConfig())
    path, kwargs = fakes.loads[0]
    assert path == "/drive/llama"
    assert kwargs["device_map"] == {"": 0} and kwargs["torch_dtype"] == "float16"
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["quantization_config"].load_in_4bit is True
    assert "attn_implementation" not in kwargs


def test_eight_bit_and_unquantized_loading(fakes):
    make_backend(quantization="8bit").generate("Llama-8B-FFT", "hi", GenerationConfig())
    assert fakes.loads[0][1]["quantization_config"].load_in_8bit is True
    make_backend(quantization="none").generate("Llama-8B-FFT", "hi", GenerationConfig())
    assert "quantization_config" not in fakes.loads[1][1]


def test_gemma_models_load_with_eager_attention(fakes):
    make_backend().generate("Gemma2-27B-CPT", "hi", GenerationConfig())
    assert fakes.loads[0][1]["attn_implementation"] == "eager"


def test_only_one_model_is_kept_loaded_and_the_previous_is_freed(fakes):
    backend = make_backend()
    backend.generate("Llama-8B-FFT", "hi", GenerationConfig())
    backend.generate("Llama-8B-FFT", "again", GenerationConfig())
    assert len(fakes.loads) == 1  # same model: no reload
    before = fakes.empty_cache_calls
    backend.generate("Gemma2-27B-CPT", "hi", GenerationConfig())
    assert [p for p, _ in fakes.loads] == ["/drive/llama", "/drive/gemma"]
    assert backend._current_name == "Gemma2-27B-CPT"
    assert fakes.empty_cache_calls > before  # GPU memory released before loading the next one


def test_a_failed_load_is_a_clear_error_and_is_not_retried_for_every_example(fakes):
    fakes.missing_models.add("/drive/llama")
    backend = make_backend()
    for _ in range(3):
        with pytest.raises(BackendError, match="Could not load model 'Llama-8B-FFT'"):
            backend.generate("Llama-8B-FFT", "hi", GenerationConfig())
    assert len(fakes.loads) == 1  # a multi-minute Drive read is attempted once, not once per example


def test_unknown_model_name_is_a_clear_error(fakes):
    with pytest.raises(BackendError, match="no entry for model 'Mystery'"):
        make_backend().generate("Mystery", "hi", GenerationConfig())


def test_lora_adapter_is_applied_on_top_of_its_base_model(fakes, tmp_path):
    adapter = tmp_path / "qlora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path": "/drive/base-8b"}')
    fakes.missing_tokenizers.add(str(adapter))  # adapter folders often have no tokenizer
    backend = make_backend({"Llama-8B-QLoRA": str(adapter)})
    output = backend.generate("Llama-8B-QLoRA", "hi", GenerationConfig())
    assert output.ok
    assert fakes.loads[0][0] == "/drive/base-8b"   # the base weights are what get loaded
    assert fakes.peft_calls == [str(adapter)]       # then the adapter is applied


def test_explicit_base_model_path_is_used_for_an_adapter(fakes, tmp_path):
    adapter = tmp_path / "qlora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")  # config doesn't name a base
    backend = make_backend({"Q": {"path": str(adapter), "base_model_path": "/drive/explicit-base"}})
    assert backend.generate("Q", "hi", GenerationConfig()).ok
    assert fakes.loads[0][0] == "/drive/explicit-base"


def test_adapter_with_unknown_base_fails_clearly(fakes, tmp_path):
    adapter = tmp_path / "qlora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    with pytest.raises(BackendError, match="base_model_path"):
        make_backend({"Q": str(adapter)}).generate("Q", "hi", GenerationConfig())


def test_invalid_configuration_is_rejected_up_front(fakes):
    with pytest.raises(BackendError, match="quantization must be"):
        make_backend(quantization="2bit")
    with pytest.raises(BackendError, match="needs a CUDA GPU"):
        make_backend(quantization="4bit", device="cpu")
    with pytest.raises(BackendError):
        LocalBackend({"M": 123})  # malformed models.json entry


def test_missing_torch_gives_install_instructions(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    with pytest.raises(BackendError, match="pip install"):
        LocalBackend({"M": "/x"})


def test_runner_integration_loglik_job_through_the_local_backend(fakes, tmp_path, monkeypatch):
    """The full path: Task -> runner -> LocalBackend -> scoring -> JSONL, using the fake model."""
    from lughagen_eval.runner import evaluate_job
    from lughagen_eval.tasks import belebele as belebele_module
    from lughagen_eval.tasks.belebele import BelebeleTask

    def rows(language_code, limit, cache_dir):
        for i, correct in enumerate(["2", "2", "3"]):  # letters B, B, C
            yield {"index": i, "passage": "p", "question": f"q{i}", "choices": list("wxyz"),
                   "correct_answer_num": correct}

    monkeypatch.setattr(belebele_module, "_load_raw_rows", rows)
    fakes.favorite = {":": " B"}  # the fake model always answers B
    path = evaluate_job(task=BelebeleTask(), model="Llama-8B-FFT", language="swh_Latn", output_dir=tmp_path,
                        backend=make_backend(), config=GenerationConfig())
    import json
    metrics = json.loads(path.with_suffix(".metrics.json").read_text())
    assert metrics["examples"] == 3 and metrics["complete"] is True
    assert metrics["score"] == pytest.approx(2 / 3)  # B is right for two of the three
    assert metrics["random_baseline"] == 0.25
