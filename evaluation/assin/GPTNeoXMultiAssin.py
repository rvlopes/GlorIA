from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import GPTNeoXPreTrainedModel, GPTNeoXModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


class GPTNeoXMultiAssin(GPTNeoXPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"lm_head.weight",
        r"h\.\d+\.attn\.attention\.bias",
    ]
    _keys_to_ignore_on_save = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTNeoXModel(config)
        self.sts_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.rte_head = nn.Linear(config.hidden_size, 2, bias=False)
        self.dropout = nn.Dropout(p=0.1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] | Tuple[torch.tensor, torch.tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # rte_logits = self.rte_head(hidden_states)
        rte_logits = self.rte_head(self.dropout(hidden_states))

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(rte_logits.device)
            else:
                sequence_lengths = -1

        loss = None
        aux_labels = [[], []]
        for label_tuple in labels:
            aux_labels[0] += [label_tuple[0]]
            aux_labels[1] += [label_tuple[1]]

        # labels 0 is entailment, labels 1 is sim
        rte_pooled_logits = rte_logits[torch.arange(batch_size, device=rte_logits.device), sequence_lengths]
        rte_loss_fct = CrossEntropyLoss()
        rte_loss = rte_loss_fct(rte_pooled_logits.view(-1, 2),
                                torch.tensor(aux_labels[0]).to("cuda").to(torch.int64))  # .view(-1)
        loss = rte_loss  # .to(torch.float32)

        # sts_logits = self.sts_head(hidden_states)
        sts_logits = self.sts_head(self.dropout(hidden_states))
        sts_pooled_logits = sts_logits[torch.arange(batch_size, device=sts_logits.device), sequence_lengths]
        sts_loss_fct = MSELoss()

        sts_loss = sts_loss_fct(sts_pooled_logits.squeeze(),
                                torch.tensor(aux_labels[1]).to("cuda").to(torch.float))  # .squeeze()
        loss += sts_loss  # .to(torch.float32)

        # loss = loss / 2

        if not return_dict:
            output = (rte_pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=(rte_pooled_logits, sts_pooled_logits),
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )