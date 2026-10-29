# LLM Fine-tuning Resources

## Knowledge

- [Article: "LoRA Without Regret" by John Schulman et al. (Thinking Machines, Sept 2025)](https://thinkingmachines.ai/blog/lora/)
  The most careful public study of LoRA hyperparameters. Covers target modules (all layers, especially MLP), learning rate (about 10× full fine-tuning), α/r scaling and batch-size sensitivity. Use for: any LoRA knob question.
- [TRL docs: LoRA Without Regret reproduction](https://huggingface.co/docs/trl/main/en/lora_without_regret)
  Hugging Face's reproduction in TRL, the library this repo trains with. Use for: translating the findings into `SFTConfig`/`LoraConfig`.
- [Paper: QLoRA, Dettmers et al. (2023)](https://arxiv.org/abs/2305.14314)
  NF4, double quantization, paged optimizers. Use for: the `quantization` block.
- [Paper: LoRA, Hu et al. (2021)](https://arxiv.org/abs/2106.09685)
  The original low-rank adaptation method. Use for: what `r` and ΔW = BA mean.
- [Paper: TabLLM, Hegselmann et al. (AISTATS 2023)](https://proceedings.mlr.press/v206/hegselmann23a.html)
  Fine-tuning LLMs on rows of a table written out as text. Finds LLMs strongest when labels are scarce and gradient-boosted trees competitive with many rows. Use for: why the LLM ties XGBoost here.
- [Docs: PEFT LoraConfig](https://huggingface.co/docs/peft/package_reference/lora)
  Authoritative field definitions and defaults (r=8, alpha=8).
- [Docs: scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
  Platt/sigmoid calibration and reliability curves. Use for: log-loss vs AUC, recalibration.
- In-repo: `plans/llm_finetune_handoff.md`, `project_knowledge/llm_training/*.md`
  The user's own experiment results. Ground every lesson in these numbers.

## Wisdom (Communities)

- [Hugging Face Forums](https://discuss.huggingface.co/)
  Maintainers of TRL/PEFT answer there. Use for: trainer behaviour and config questions.
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
  Large, active practitioner community for fine-tuning small open models. Quality varies, so cross-check claims against Knowledge sources.

## Gaps

- No high-trust source yet on LLMs for *sports* outcome prediction specifically.
- No primary source yet on combining LLM and gradient-boosted tree predictions (stacking or blending) for tabular tasks. Needed for N2.
