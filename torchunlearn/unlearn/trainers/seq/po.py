"""PO / IDK (Preference Optimization): the forget answer is replaced by a refusal and fine-tuned with NLL together
with the retain set.  Baseline of FIUBench (ICLR 2025), MLLMU-Bench, UMU-Bench (NeurIPS 2025), and the expert
training objective of LUMoE (ICML 2026).

    L = NLL(refusal | image, question)  +  alpha * NLL(retain)

The refusal comes from ``SeqCollator(alt_text="I cannot answer that.")`` or ``SeqSample.alt_answers``.
"""
from .base import SeqUnlearner


class PO(SeqUnlearner):
    """Preference optimisation towards a refusal answer (alt_*), plus retain NLL."""
    needs_alt = True

    def forget_loss(self, batch):
        if "alt_input_ids" not in batch:
            raise ValueError("PO needs a refusal answer: SeqCollator(alt_text=...) or SeqSample.alt_answers")
        logits = self._logits(batch, prefix="alt_")
        nll, mask = self._token_nll(logits, batch["alt_labels"])
        return nll.sum() / mask.sum().clamp(min=1)
