"""A stand-in for AsyncTypeSafeClient that answers Noul, Score and Choice questions."""

import asyncio
from types import SimpleNamespace


class FakeJev:
    """`answer(state, questions)` returns {question_id: value}: a float for a Noul or
    Score; for a Choice, a label or {"choice", "confidence", "probabilities"}.
    Unanswered questions get 0.0 / the first Choice label at confidence 1.0, or are
    left out entirely when fill=False (a partial answer). `fail(state)` raising
    simulates an API error."""

    def __init__(self, answer=None, fail=None, delay=0.0, fill=True):
        self.answer = answer or (lambda state, questions: {})
        self.fail = fail or (lambda state: False)
        self.delay, self.fill, self.calls = delay, fill, []

    async def system_one(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail(state):
            raise RuntimeError("jev unavailable")
        got = self.answer(state, questions)
        nouls, scores, choices = {}, {}, {}
        for qid, q in questions.items():
            if qid not in got and not self.fill:
                continue
            value = got.get(qid)
            kind = type(q).__name__
            if kind == "Noul":
                nouls[qid] = SimpleNamespace(noul=0.0 if value is None else value)
            elif kind == "Score":
                scores[qid] = SimpleNamespace(score=0.0 if value is None else value)
            else:
                if value is None:
                    value = next(iter(q.criteria))
                if isinstance(value, str):
                    value = {"choice": value, "confidence": 1.0, "probabilities": {value: 1.0}}
                choices[qid] = SimpleNamespace(**value)
        return SimpleNamespace(nouls=nouls, scores=scores, choices=choices)
