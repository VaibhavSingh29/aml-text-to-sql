from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.auto import AutoTokenizer
from transformers.generation.beam_search import BeamScorer
import torch
from typing import List, Optional, Tuple, Union

from collections import UserDict


class RABeamScorer(BeamScorer):
    def __init__(
        self, 
        batch_size: int, 
        num_beams: int, 
        device: torch.device, 
        tokenizer: Union[T5TokenizerFast, AutoTokenizer], 
        num_beam_hyps_to_keep: Optional[int] = 1, 
        num_beam_groups: Optional[int] = 1, 
        max_length: Optional[int] = None
    ):
        self.num_beams = num_beams
        self.device = device
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups =num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups # we will have one group per batch instance
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.open_brace_input_id = tokenizer.encode("(")[0]
        self.close_brace_input_id = tokenizer.encode(")")[0]

        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=1,
                early_stopping=True,
                max_length=100
            )
            for _ in range(batch_size)
        ]

        self._is_init = False
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)
        
    def process(
        self,
        input_ids: torch.LongTensor, # (batch_size * num_beams, sequence_length)
        next_scores: torch.FloatTensor, # (batch_size, 2* num_beams)
        next_tokens: torch.LongTensor, # (batch_size, 2* num_beams)
        next_indices: torch.LongTensor, 
        eos_token_id: Optional[Union[int, List[int]]],
        pad_token_id: Optional[int] = None,
        beam_indices: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor]:

        current_length = input_ids.shape[-1]
        batch_size = input_ids.shape[0] // self.num_beams 

        # if current_length == 1: # does this work
        #     input_ids = torch.zeros_like(input_ids)
        #     input_ids = self.open_brace_input_id

        device = input_ids.device

        next_beam_scores = torch.zeros((batch_size, self.num_beams), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.num_beams), dtype=next_scores.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.num_beams), dtype=next_scores.dtype, device=device)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue
            
            # generate next tokens
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])):
                batch_beam_idx = batch_idx * self.group_size + next_index
                if (eos_token_id is not None) and (next_token.item() is eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (batch_beam_idx,)
                    else:
                        beam_index = None

                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        beam_indices=beam_index,
                    )
                else:
                    # add next predicted token (not eos)
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once num_beams attained dont add more
                if beam_idx == self.group_size:
                    break
            
            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                    f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            cur_len += 1  # add up to the length which the next_scores is calculated on
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )
            
        

        return UserDict(
            {
                'next_beam_scores': next_beam_scores.view(-1),
                'next_beam_tokens': next_beam_tokens.view(-1),
                'next_beam_indices': next_beam_indices.view(-1),
            }
        )

        # - **next_beam_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Updated scores of all non-finished beams.
        # - **next_beam_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Next tokens to be added to the non-finished beam_hypotheses.
        # - **next_beam_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Beam indices indicating to which beam the next tokens shall be added.

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        next_scores: torch.FloatTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
    ) -> [torch.LongTensor]:

        batch_size = len(self._beam_hyps)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue
            
            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
                beam_hyp.add(final_tokens, final_score, beam_indices=beam_index)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_indices = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append hyp to lists
                best.append(best_hyp)

                # append indices to list
                best_indices.append(best_index)

                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        else:
            indices = None

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo

            if indices is not None:
                indices[i, : len(best_idx)] = torch.tensor(best_idx)

            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )

        #  `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated sequences.
        # The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished early
        # due to the `eos_token_id`.

    @property
    def is_done(self) -> bool:
        return self._done.all()

class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: Optional[int] = None):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

        if not isinstance(self.early_stopping, bool) and self.max_length is None:
            raise ValueError(
                "When `do_early_stopping` is set to a string, `max_length` must be defined. Ensure it is passed to the"
                " BeamScorer class instance at initialization time."
            )

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float, beam_indices: Optional[torch.LongTensor] = None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False

        # `True`: stop as soon as at least `num_beams` hypotheses are finished
        if self.early_stopping is True:
            return True
        # `False`: heuristic -- compute best possible score from `cur_len`, even though it is not entirely accurate
        #  when `length_penalty` is positive. See the discussion below for more details.
        # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        elif self.early_stopping is False:
            highest_attainable_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret
        # `"never"`: compute the best possible score, depending on the signal of `length_penalty`
        else:
            # `length_penalty` > 0.0 -> max denominator is obtaned from `max_length`, not from `cur_len` -> min
            # abs(`highest_attainable_score`) is obtained -> `highest_attainable_score` is negative, hence we obtain
            # its max this way
            if self.length_penalty > 0.0:
                highest_attainable_score = best_sum_logprobs / self.max_length**self.length_penalty
            # the opposite logic applies here (max `highest_attainable_score` from `cur_len`)
            else:
                highest_attainable_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret

