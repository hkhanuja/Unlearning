import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def custom_data_collator_forget(samples):
    forget_samples = [sample[0] for sample in samples]
    alternate_samples1 = [sample[1] for sample in samples]
    alternate_samples2 = [sample[2] for sample in samples]
    alternate_samples3 = [sample[3] for sample in samples]
    retain_samples = [sample[4] for sample in samples]

    def stack_data(data):
        input_ids = torch.stack([s[0] for s in data])
        labels = torch.stack([s[1] for s in data])
        attention_mask = torch.stack([s[2] for s in data])
        return input_ids, labels, attention_mask
    return (
        stack_data(forget_samples),
        stack_data(alternate_samples1),
        stack_data(alternate_samples2),
        stack_data(alternate_samples3),
        stack_data(retain_samples),
    )

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss

def compute_loss(model, inputs, base_model):

    forget_inputs, alternate_inputs1 ,alternate_inputs2, alternate_inputs3, retain_inputs = inputs
    forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
    alternate_input_ids1, alternate_labels1, alternate_attention_mask1 = alternate_inputs1
    alternate_input_ids2, alternate_labels2, alternate_attention_mask2 = alternate_inputs2
    alternate_input_ids3, alternate_labels3, alternate_attention_mask3 = alternate_inputs3
    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs

    forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
    alternate_outputs1 = model(alternate_input_ids1,labels=alternate_labels1, attention_mask=alternate_attention_mask1)
    alternate_outputs2 = model(alternate_input_ids2,labels=alternate_labels2, attention_mask=alternate_attention_mask2)
    alternate_outputs3 = model(alternate_input_ids3,labels=alternate_labels3, attention_mask=alternate_attention_mask3)
    retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)

    with torch.no_grad():
        alternate_outputs1_base = base_model(alternate_input_ids1,labels=alternate_labels1, attention_mask=alternate_attention_mask1)
        alternate_outputs2_base = base_model(alternate_input_ids2,labels=alternate_labels2, attention_mask=alternate_attention_mask2)
        alternate_outputs3_base = base_model(alternate_input_ids3,labels=alternate_labels3, attention_mask=alternate_attention_mask3)
        forget_outputs_base = base_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
        alternate_outputs1_logits = alternate_outputs1_base.logits
        alternate_outputs2_logits = alternate_outputs2_base.logits
        alternate_outputs3_logits = alternate_outputs3_base.logits
        forget_logits_base = forget_outputs_base.logits
    
    alternate1_loss_base = -1 * get_batch_loss(alternate_outputs1_logits, alternate_labels1)
    alternate2_loss_base = -1 * get_batch_loss(alternate_outputs2_logits, alternate_labels2)
    alternate3_loss_base = -1 * get_batch_loss(alternate_outputs3_logits, alternate_labels3)
    forget_loss_base = -1 * get_batch_loss(forget_logits_base, forget_labels)
        
    alternate1_loss_current = -1 * get_batch_loss(alternate_outputs1.logits, alternate_labels1)
    alternate2_loss_current = -1 * get_batch_loss(alternate_outputs2.logits, alternate_labels2)
    alternate3_loss_current = -1 * get_batch_loss(alternate_outputs3.logits, alternate_labels3)
    forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)
    
    pi_logratios = alternate1_loss_current - forget_loss_current
    ref_logratios = alternate1_loss_base - forget_loss_base

    beta = 0.1
    loss1 = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

    pi_logratios = alternate2_loss_current - forget_loss_current
    ref_logratios = alternate2_loss_base - forget_loss_base

    loss2 = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

    pi_logratios = alternate3_loss_current - forget_loss_current
    ref_logratios = alternate3_loss_base - forget_loss_base

    loss3 = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
    
    logits = retain_outputs.logits.view(-1, retain_outputs.logits.size(-1))
    labels = retain_labels.view(-1) 

    nll_loss = F.cross_entropy(logits, labels)

    w_r = 1

    loss = loss1+loss2+loss3
    # Combine the two terms
    total_loss = loss + w_r * nll_loss
    
    return total_loss

def compute_idk_loss(model, inputs):

    forget_inputs, alternate_inputs1 ,alternate_inputs2, alternate_inputs3, retain_inputs = inputs
    alternate_input_ids1, alternate_labels1, alternate_attention_mask1 = alternate_inputs1
    alternate_input_ids2, alternate_labels2, alternate_attention_mask2 = alternate_inputs2
    alternate_input_ids3, alternate_labels3, alternate_attention_mask3 = alternate_inputs3
    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs

    
    input_ids = torch.cat((alternate_input_ids1, retain_input_ids), dim=0)
    labels = torch.cat((alternate_labels1, retain_labels), dim=0)
    attention_mask = torch.cat((alternate_attention_mask1, retain_attention_mask), dim=0)
    
    output = model(input_ids,labels=labels, attention_mask=attention_mask)
    loss = output.loss
    return loss