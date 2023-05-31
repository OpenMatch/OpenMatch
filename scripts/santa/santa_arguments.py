from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments, Seq2SeqTrainingArguments

@dataclass
class SantaArguments:
    l_max_len: int = field(
        default=35,
        metadata={
            "help": "The max length of label"}
    )
    use_generate: bool = field(
        default=True,
        metadata={"help": "Whether to get generate loss"}
    )
