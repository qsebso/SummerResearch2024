class Entity:
    def __init__(self, etype: str, span: str):
        self.etype = etype
        self.span = span

class Relation:
    def __init__(self, rtype: str, entities: list[Entity], slots: list[str]):
        self.rtype = rtype
        self.entities = entities
        self.slots = slots

class Article:
    def __init__(self, text: str, relations: list[Relation]):
        self.text = text
        self.relations = relations