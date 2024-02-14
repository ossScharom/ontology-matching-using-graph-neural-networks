class LabelEncoder:
    def __init__(self, elements):
        self._mapping = dict(zip(elements, range(len(elements))))
        self._mapping_inverse = {v: k for k, v in self._mapping.items()}

    def transform(self, elements):
        return [self._mapping[x] for x in elements]

    def inverse_transform(self, elements):
        return [self._mapping_inverse[x] for x in elements]
