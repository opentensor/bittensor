from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM
from ..base import LM


class AutoLM(LM):
    """
    Auto language model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (AutoConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the low-rank approximation.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[AutoConfig] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        if pretrained is not None:
            model = AutoModelForCausalLM.from_pretrained(pretrained)
        elif config is not None:
            model = AutoModelForCausalLM(config)
        else:
            model = AutoModelForCausalLM(AutoConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        super().__init__(model, lora_rank, lora_train_bias)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
