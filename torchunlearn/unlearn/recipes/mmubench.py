"""MMUBench (SIU, Li et al., NeurIPS 2024): 20 concepts x 50 images; 1 training image + 49 test images per concept.
Also the builder for SIU's *Multifaceted Fine-tuning Data* (four targets).

    siu_train = build_siu_samples(concept="Donald Trump", image=img, new_name="Jacob Campbell",
                                  visual_descriptions=[...], facts=[...], other_samples=[...])
    forget = [s for s in siu_train if s.meta["target"] in (1, 2)]          # image-conditioned (targets 1-2)
    retain = [s for s in siu_train if s.meta["target"] in (3, 4)]          # text facts + non-target knowledge

Target 1  aligning with an unseen concept:  (image, "Who is this?") -> substitute name   (specified_text = new name)
Target 2  new visual description:           (image, "Describe this person.") -> description without the real name
Target 3  decoupling factual knowledge:     text-only (no image) facts about the concept  (keeps textual knowledge)
Target 4  preserving non-target knowledge:  QA about other concepts
The paper rephrases every target with GPT-4; pass several phrasings through the ``*_prompts`` arguments.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from ..seq_data import SeqSample

NAME_PROMPTS = ["Who is the person in this image?", "What is the name of the person shown here?", "Can you identify this person?"]
DESC_PROMPTS = ["Describe the person in this image.", "What does the person in the image look like?"]
TEST_PROMPTS = ["Who is this?", "What is this person's name?", "Is the person in the image {name}? Answer yes or no."]


def build_siu_samples(concept: str, image, new_name: str, visual_descriptions: Sequence[str], facts: Sequence[tuple],
                      other_samples: Sequence[SeqSample] = (), name_prompts: Sequence[str] = NAME_PROMPTS,
                      desc_prompts: Sequence[str] = DESC_PROMPTS) -> List[SeqSample]:
    out = []
    for k, p in enumerate(name_prompts):
        out.append(SeqSample(prompt=p, answer=f"This is {new_name}.", images=[image], group=concept, id=f"{concept}#t1_{k}",
                             meta={"target": 1, "specified_text": new_name, "concept": concept}))
    for k, (p, d) in enumerate(zip(desc_prompts * len(visual_descriptions), visual_descriptions)):
        out.append(SeqSample(prompt=p, answer=d, images=[image], group=concept, id=f"{concept}#t2_{k}",
                             meta={"target": 2, "specified_text": None, "concept": concept}))
    for k, (q, a) in enumerate(facts):
        out.append(SeqSample(prompt=q, answer=a, images=[], group=concept, id=f"{concept}#t3_{k}", meta={"target": 3, "concept": concept}))
    for k, s in enumerate(other_samples):
        out.append(SeqSample(prompt=s.prompt, answer=s.answer, images=list(s.images), group=s.group, id=f"other#{k}",
                             meta={**s.meta, "target": 4}))
    return out


def test_samples(concept: str, images: Iterable, prompts: Sequence[str] = TEST_PROMPTS) -> List[SeqSample]:
    out = []
    for i, im in enumerate(images):
        for k, p in enumerate(prompts):
            out.append(SeqSample(prompt=p.format(name=concept), answer=concept, images=[im], group=concept, id=f"{concept}#test{i}_{k}"))
    return out


def from_folder(root: str, concepts: Optional[Sequence[str]] = None):
    """root/<concept>/*.jpg -> {concept: [paths]} (first image = training image, rest = test images)."""
    import glob
    import os
    out = {}
    for d in sorted(glob.glob(os.path.join(glob.escape(root), "*"))):
        if os.path.isdir(d):
            c = os.path.basename(d)
            if concepts and c not in concepts:
                continue
            out[c] = sorted(p for p in glob.glob(os.path.join(glob.escape(d), "*")) if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    return out
