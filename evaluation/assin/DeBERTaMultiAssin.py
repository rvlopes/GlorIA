import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import DebertaV2ForSequenceClassification, DebertaV2Model
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout
from typing import Optional, Tuple, Union


class DeBERTaMultiAssin(DebertaV2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.sts_head = nn.Linear(output_dim, 1, bias=False)
        self.rte_head = nn.Linear(output_dim, 2, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification.forward with Deberta->DebertaV2
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)

        rte_logits = self.rte_head(self.dropout(pooled_output))

        loss = None
        aux_labels = [[], []]
        for label_tuple in labels:
            aux_labels[0] += [label_tuple[0]]
            aux_labels[1] += [label_tuple[1]]

        # labels 0 is entailment, labels 1 is sim
        rte_pooled_logits = rte_logits #rte_logits[torch.arange(batch_size, device=rte_logits.device), sequence_lengths]
        rte_loss_fct = CrossEntropyLoss()
        rte_loss = rte_loss_fct(rte_pooled_logits.view(-1, 2),
                                torch.tensor(aux_labels[0]).to("cuda").to(torch.int64))  # .view(-1)
        loss = rte_loss

        sts_logits = self.sts_head(self.dropout(pooled_output))
        sts_pooled_logits = sts_logits #sts_logits[torch.arange(batch_size, device=sts_logits.device), sequence_lengths]
        sts_loss_fct = MSELoss()

        sts_loss = sts_loss_fct(sts_pooled_logits.squeeze(),
                                torch.tensor(aux_labels[1]).to("cuda").to(torch.float))  # .squeeze()
        loss += sts_loss

        if not return_dict:
            output = (rte_pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=(rte_pooled_logits, sts_pooled_logits), hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
