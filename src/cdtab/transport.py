import time
from .great_bridge import NativeSessionProcessor

class BatchProcessor:

    def __init__(self, masker, native, input_ids, attention_mask, pad_id):
        self.masker, self.native, self.pad_id = (masker, native, int(pad_id))
        self.prompts = input_ids.tolist()
        masks = attention_mask.tolist()
        self.width = input_ids.shape[1]
        self.processors, self.padding = ([], [])
        for ids, mask in zip(self.prompts, masks):
            if len(ids) != len(mask) or any((v not in (0, 1) for v in mask)) or 1 not in mask:
                raise ValueError('INVALID_NATIVE_ATTENTION_MASK')
            left = mask.index(1)
            if mask != [0] * left + [1] * (len(ids) - left) or any((t != self.pad_id for t in ids[:left])):
                raise ValueError('EXPECTED_CONTIGUOUS_LEFT_PADDING')
            self.padding.append(left)
            self.processors.append(NativeSessionProcessor(masker, native, ids[left:]))
        self.pending = [None] * len(self.processors)
        self.seen = [self.width] * len(self.processors)
        self.calls, self.seconds = (0, 0.0)

    def sync(self, batches):
        if len(batches) != len(self.processors):
            raise ValueError('BATCH_SIZE_CHANGED')
        for i, (ids, proc) in enumerate(zip(batches, self.processors)):
            session = proc.session
            consumed = self.prompts[i][:self.padding[i]] + list(proc.ids)
            if ids[:len(consumed)] != consumed or len(ids) < self.seen[i]:
                raise ValueError('CONSUMED_PREFIX_CHANGED')
            for token in ids[self.seen[i]:]:
                if session.finished:
                    if token != self.pad_id:
                        raise ValueError('NONPAD_AFTER_EOS')
                    continue
                transitions = self.pending[i]
                if transitions is None or token not in transitions:
                    raise ValueError('TOKEN_OUTSIDE_EXACT_MASK')
                if token == self.masker.eos_token_id:
                    if transitions[token] is not None or not self.masker.engine.can_end(session.state):
                        raise ValueError('ILLEGAL_EOS')
                    session.row = self.masker.engine.finish(session.state)
                    session.finished = True
                else:
                    if transitions[token] is None:
                        raise ValueError('MISSING_EXACT_SUCCESSOR')
                    session.state = transitions[token]
                proc.ids += (token,)
                self.pending[i] = None
            self.seen[i] = len(ids)

    def __call__(self, input_ids, scores):
        import torch
        tick = time.perf_counter()
        try:
            if input_ids.ndim != 2 or scores.ndim != 2 or scores.shape[0] != len(self.processors):
                raise ValueError('UNSUPPORTED_BATCH_SHAPE')
            self.sync(input_ids.tolist())
            row_ids, token_ids = ([], [])
            for i, proc in enumerate(self.processors):
                if proc.session.finished:
                    legal = [self.pad_id]
                else:
                    transitions = self.masker.legal_token_transitions(proc.session.state)
                    if not transitions:
                        raise ValueError('EMPTY_EXACT_MASK')
                    self.pending[i] = transitions
                    legal = list(transitions)
                    proc.calls += 1
                row_ids.extend([i] * len(legal))
                token_ids.extend(legal)
            out = torch.full_like(scores, -float('inf'))
            ri = torch.tensor(row_ids, dtype=torch.long, device=scores.device)
            ti = torch.tensor(token_ids, dtype=torch.long, device=scores.device)
            out[ri, ti] = scores[ri, ti]
            self.calls += 1
            return out
        finally:
            self.seconds += time.perf_counter() - tick

    def finish(self, token_batches):
        self.sync(token_batches)
        return [p.finish(p.ids) for p in self.processors]
