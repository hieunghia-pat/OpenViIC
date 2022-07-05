import torch
from data_utils.types import *

class BeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

    def _expand_state(self, selected_beam, cur_beam_size):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([self.b_s, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([self.b_s, self.beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    def _expand_visual(self, tensor: TensorOrSequence, cur_beam_size: int, selected_beam: torch.Tensor):
        tensor_shape = tensor.shape # (bs, seq_len, d_model)
        tensor_exp_shape = (self.b_s, cur_beam_size) + tensor_shape[1:] # (bs, cur_beam_size, seq_len, d_model)
        tensor_red_shape = (self.b_s * self.beam_size,) + tensor_shape[1:] # (bs * beam_size, seq_len, d_model)
        selected_beam_red_size = (self.b_s, self.beam_size) + tuple(1 for _ in range(len(tensor_exp_shape) - 2)) # (bs, beam_size, 1, 1)
        selected_beam_exp_size = (self.b_s, self.beam_size) + tensor_exp_shape[2:] # (bs, beam_size, seq_len, d_model)
        tensor_exp = tensor.view(tensor_exp_shape)
        selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(selected_beam_exp_size)
        tensor = torch.gather(tensor_exp, dim=1, index=selected_beam_exp).view(tensor_red_shape)    # copy from visual_exp with shape (bs, cur_beam_size, seq_len, d_model)
                                                                                                    # to visual with shape (bs, beam_size, seq_len, d_model) along axis 1
                                                                                                    # at the beginning cur_beam_size is 1 as the beam search tree will expand from bos_idx
                                                                                                    # to new branch that has beam_size leaves
        
        return tensor # (bs * beam_size, seq_len, d_model)

    def select(self, candidate_logprob):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(self.b_s, -1), -1, descending=True)  # flatten the candicate_lobprob from (bs, beam_size, vocab_size) 
                                                                                                                # to (bs, beam_size*vocab_size) and decendingly sort it
        selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size] # then select the top-beam_size highest digits
        return selected_idx, selected_logprob

    def iter(self, t: int, outputs, return_probs, **visual_inputs):
        cur_beam_size = 1 if t == 0 else self.beam_size

        word_logprob = self.model.step(t, self.selected_words, **visual_inputs)
        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)
        candidate_logprob = self.seq_logprob + word_logprob

        # Mask sequence if it reaches <eos>
        if t > 0:
            mask = (self.selected_words.view(self.b_s, cur_beam_size) == self.eos_idx).unsqueeze(-1) # (bs, cur_beam_size, 1)
            self.seq_mask = torch.logical_or(self.seq_mask, mask)
            word_logprob = word_logprob.masked_fill(self.seq_mask.expand_as(word_logprob), value=0)
            candidate_logprob = torch.where(self.seq_mask, torch.scalar_tensor(0, device=self.device), word_logprob + self.seq_logprob)

        selected_idx, selected_logprob = self.select(candidate_logprob) # get the top-beam_size highest logits
        selected_beam = torch.div(selected_idx, candidate_logprob.shape[-1], rounding_mode="floor") # then find its appropriate beam
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1] # and get the index of them in term of vocab size

        self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size))

        # reorder the seq_logprob as well as the seq_mask and outputs to match the newest selected beam
        self.seq_logprob = selected_logprob.unsqueeze(-1)
        self.seq_mask = torch.gather(self.seq_mask, 1, selected_beam.unsqueeze(-1))
        outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
        outputs.append(selected_words.unsqueeze(-1))

        if return_probs:
            if t == 0:
                self.all_log_probs.append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
            else:
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        this_word_logprob = torch.gather(word_logprob, 1,
                                         selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                            word_logprob.shape[-1]))
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
        self.log_probs = list(
            torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size, 1)) for o in self.log_probs)
        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)

        return outputs

    def apply(self, batch_size, device, out_size=1, return_probs=False, **visual_inputs):
        self.b_s = batch_size
        self.device = device
        self.seq_mask = torch.zeros((self.b_s, self.beam_size, 1), device=self.device).bool()
        self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device)    # (bs, beam_size, 1)
                                                                                # at the beginning the beam search tree has a root of bos_idx
        self.log_probs = []
        self.selected_words = None
        if return_probs:
            self.all_log_probs = []

        outputs = []
        with self.model.statefulness(self.b_s):
            for t in range(self.max_len):
                outputs = self.iter(t, outputs, return_probs, **visual_inputs)

        # Sort result
        _, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        log_probs = torch.cat(self.log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        if return_probs:
            all_log_probs = torch.cat(self.all_log_probs, 2)
            all_log_probs = torch.gather(all_log_probs, 
                                            dim=1, 
                                            index=sort_idxs.unsqueeze(-1).expand(self.b_s, self.beam_size,
                                                                            self.max_len,
                                                                            all_log_probs.shape[-1]))

        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        if return_probs:
            return outputs, log_probs, all_log_probs
        else:
            return outputs, log_probs
