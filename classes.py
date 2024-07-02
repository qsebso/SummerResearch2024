class Entity:
    def __init__(self, unused: str, span: str):
        # vestigial first positional argument; it used to be "etype", but that
        # behavior has been moved to TypedEntity
        self.span = span

    def __repr__(self) -> str:
        repr: str = f'{self.span}'
        return repr

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other) -> bool:
        return self.span == other.span

    def __hash__(self) -> int:
        return hash(self.span)

    def to_dict(self) -> dict:
        return {'span': self.span}

class TypedEntity:
    def __init__(self, etype: str, span: str):
        self.etype = etype
        self.span = span

    def __repr__(self) -> str:
        repr: str = f'<{self.etype}> {self.span}'
        return repr

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other) -> bool:
        return self.etype == other.etype and self.span == other.span

    def __hash__(self) -> int:
        return hash(self.etype) ^ hash(self.span)

    def to_dict(self) -> dict:
        return {'etype': self.etype, 'span': self.span}

class Relation:
    def __init__(self, rtype: str, entities: list[Entity], slots: list[str], evidence=list[int]):
        assert len(entities) == len(slots)
        self.rtype = rtype
        self.entities = entities
        self.slots = slots
        self.evidence = evidence

    def __str__(self) -> str:
        repr: str = ' | '.join([f'<<{self.rtype}>>'] + [f'{slot}: {r}' for slot, r in zip(self.slots, self.entities)])
        return repr
    
    def __repr__(self) -> str:
        return f'<<{self.rtype}>>' + ''.join([f'<{slot}>' for slot in self.slots])

    def __eq__(self, other) -> bool:
        return self.rtype == other.rtype and self.entities == other.entities

    def __hash__(self) -> int:
        return hash(self.rtype) ^ hash(tuple(self.entities))

    def to_dict(self) -> dict:
        ents = { slot: e.to_dict() for slot, e in zip(self.slots, self.entities) }
        return { 
            'rtype': self.rtype,
            'entities': ents
        }

    def from_json(json_data):
        ...

class Article:
    def __init__(self, text: str, relations: list[Relation]):
        self.text: str = text
        self.relations: list[Relation] = relations
        self.target: str = ""

    def __repr__(self) -> str:
        return f'<{len(self.text)=}|{len(self.relations)=}>'

    def __str__(self) -> str:
        repr: str = f'TEXT:\n{self.text}\n\nRELATIONS:\n\t' + '\n\t'.join(map(str, self.relations))
        return repr

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'relations': [r.to_dict() for r in self.relations]
        }

    def get_entities(self):
        out = []
        for rel in self.relations:
            for ent in rel.entities:
                if ent not in out:
                    out.append(ent)
        return out