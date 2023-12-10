from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding, DefaultDataCollator


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    max_q_len: int = 32
    max_p_len: int = 128
    max_l_len: int = 32

    def __call__(self, features):
        qq = [f["query"] for f in features]
        dd = [f["passages"] for f in features]

        ll = [f["labels"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        if isinstance(ll[0], list):
            ll = sum(ll[0], [])
        q_collated = self.tokenizer.pad(
            qq,
            padding="max_length",
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding="max_length",
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        if ll[0] != None:
            l_collated = self.tokenizer.pad(
                ll,
                padding="max_length",
                max_length=self.max_l_len,
                return_tensors="pt",
            )

            l_collated = l_collated.input_ids
            l_collated[l_collated == self.tokenizer.pad_token_id] = -100
        else:
            l_collated = torch.tensor([0])
        return q_collated, d_collated, l_collated
