"""Внутреннее представление токена и разобранного предложения.

Это промежуточный формат парсера. Не кладём его в `domain/`, т.к. он не
является частью модели знаний — только интерфейсом между UD-парсером и
верхними стадиями pipeline.

Инварианты:
- `id_in_sent` нумеруется с 1 (как в CoNLL-U);
- `head == 0` означает «корень предложения» (обычно root-предикат);
- `start`/`end` — смещения внутри **предложения** (не документа);
- `feats` хранит UD-морфологию: `{"Case": "Nom", "Number": "Sing", ...}`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Token(BaseModel):
    id_in_sent: int = Field(ge=1)
    text: str
    lemma: str
    pos: str = Field(description="Universal POS tag: NOUN, VERB, ADJ, ...")
    feats: dict[str, str] = Field(default_factory=dict)
    head: int = Field(ge=0, description="id_in_sent головы; 0 для root.")
    deprel: str = Field(description="UD dependency relation: nsubj, obj, root, ...")
    start: int = Field(ge=0, description="Смещение в тексте предложения.")
    end: int = Field(ge=0)


class ParsedSentence(BaseModel):
    text: str
    tokens: list[Token] = Field(default_factory=list)

    def root(self) -> Token | None:
        """Вернуть токен-корень (head == 0), если есть."""
        for t in self.tokens:
            if t.head == 0:
                return t
        return None

    def children_of(self, token_id: int) -> list[Token]:
        return [t for t in self.tokens if t.head == token_id]

    def by_id(self, token_id: int) -> Token | None:
        for t in self.tokens:
            if t.id_in_sent == token_id:
                return t
        return None

    def subtree_ids(self, root_id: int, stop_deprels: set[str] | None = None) -> set[int]:
        """Собрать id всех токенов поддерева, начиная с root_id.

        Если задан `stop_deprels`, ребёнок, у которого `deprel` в этом
        множестве, не включается в результат вместе со своим поддеревом.
        Это используется для «вырезания» подчинённых клауз (ccomp, xcomp,
        advcl, acl, acl:relcl).
        """
        stop = stop_deprels or set()
        result: set[int] = {root_id}
        frontier = [root_id]
        while frontier:
            nxt: list[int] = []
            for tid in frontier:
                for child in self.children_of(tid):
                    if child.deprel in stop:
                        continue
                    if child.id_in_sent in result:
                        continue
                    result.add(child.id_in_sent)
                    nxt.append(child.id_in_sent)
            frontier = nxt
        return result
