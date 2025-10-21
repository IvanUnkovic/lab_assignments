from dataclasses import dataclass

@dataclass
class Instance:
    text: list
    label: str
    def __init__(self, text, label):
        self.text = text
        self.label = label