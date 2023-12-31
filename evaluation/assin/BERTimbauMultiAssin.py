from torch import nn
from transformers import BertForSequenceClassification, BertModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from typing import Optional, Tuple, Union
import torch
from transformers.modeling_outputs import SequenceClassifierOutput


class BERTimbauMultiAssin(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        #self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.sts_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.rte_head = nn.Linear(config.hidden_size, 2, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        #pooled_output = self.dropout(pooled_output)
        #logits = self.classifier(pooled_output)

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
            output = (rte_pooled_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=(rte_pooled_logits, sts_pooled_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
