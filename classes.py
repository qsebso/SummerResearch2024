class Entity:
    def __init__(self, etype: str, span: str):
        self.etype = etype
        self.span = span

    def __repr__(self) -> str:
        repr: str = f'<{self.etype}> {self.span}'
        return repr

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self) -> dict:
        return {'etype': self.etype, 'span': self.span}

class Relation:
    def __init__(self, rtype: str, entities: list[Entity], slots: list[str]):
        assert len(entities) == len(slots)
        self.rtype = rtype
        self.entities = entities
        self.slots = slots

    def __repr__(self) -> str:
        repr: str = ' | '.join([f'<<{self.rtype}>>'] + [f'{slot}: {r}' for slot, r in zip(self.slots, self.entities)])
        return repr
    
    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self) -> dict:
        ents = { slot: e.to_dict() for slot, e in zip(self.slots, self.entities) }
        return { 
            'rtype': self.rtype,
            'entities': ents
        }

class Article:
    def __init__(self, text: str, relations: list[Relation]):
        self.text = text
        self.relations = relations

    def __repr__(self) -> str:
        repr: str = f'TEXT:\n{self.text}\n\nRELATIONS:\n\t' + '\n\t'.join(map(str, self.relations))
        return repr

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'relations': [r.to_dict() for r in self.relations]
        }