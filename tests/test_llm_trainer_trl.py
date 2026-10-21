"""The custom SFTTrainer against the real trl, on a tiny random model on CPU.

Skipped unless the gpu extra is installed (``uv sync --extra gpu``).
"""

import pytest

pytest.importorskip("trl")


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["</s>", "<unk>", "a", "b", "c", "=", "+", "-", *map(str, range(10))]
    vocab = {w: i for i, w in enumerate(words)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))  # noqa: S106
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation(behavior="isolated")]
    )
    # Special tokens, not passwords.
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="</s>", pad_token="</s>", unk_token="<unk>"  # noqa: S106
    )


def _trainer(tmp_path):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import LlamaConfig, LlamaForCausalLM
    from trl import SFTConfig

    from src.ml.llm.finetune import build_trainer_class

    # On macOS, torch's OpenMP and the one lightgbm loaded in an earlier test
    # segfault together in a multithreaded forward. One thread is plenty here.
    torch.set_num_threads(1)
    torch.manual_seed(0)
    tokenizer = _tokenizer()
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )
    data = Dataset.from_dict(
        {
            "prompt": ["a b =", "b c =", "c a =", "a a =", "b b =", "c c ="],
            "completion": ["+7", "-3", "+1", "-9", "+4", "-2"],
        }
    )
    args = SFTConfig(
        output_dir=str(tmp_path),
        max_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        bf16=False,
        completion_only_loss=True,
        max_length=16,
    )
    return build_trainer_class()(
        model=model,
        args=args,
        train_dataset=data,
        processing_class=tokenizer,
        peft_config=LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )


def test_first_token_accuracy_costs_no_extra_forward(tmp_path):
    trainer = _trainer(tmp_path)
    forwards = []
    trainer.model.register_forward_hook(lambda *_: forwards.append(1))

    trainer.train()

    # One forward per micro-batch. Asking trl for return_outputs would add a
    # second, full-logits forward on every step.
    assert len(forwards) == trainer.state.global_step == 2
    logged = [h["first_token_accuracy"] for h in trainer.state.log_history if "first_token_accuracy" in h]
    assert len(logged) == 2
    assert all(0.0 <= value <= 1.0 for value in logged)


def test_first_token_accuracy_is_logged_on_eval(tmp_path):
    trainer = _trainer(tmp_path)

    trainer.evaluate(trainer.train_dataset)

    assert any("eval_first_token_accuracy" in h for h in trainer.state.log_history)
