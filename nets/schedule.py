import hashlib


class Epoch:
    def __init__(self, grow, prune, regularization_penalty, regularization_method):
        self.grow = grow
        self.prune = prune
        self.regularization_penalty = regularization_penalty
        self.regularization_method = regularization_method

    def __str__(self):
        return f'{int(self.grow)}{int(self.prune)}{self.regularization_penalty}{self.regularization_method}'

    def __repr__(self):
        return self.__str__()


class DynamicEpoch(Epoch):
    def __init__(self, regularization_penalty, regularization_method):
        super().__init__(True, True, regularization_penalty, regularization_method)


class StaticEpoch(Epoch):
    def __init__(self, regularization_penalty, regularization_method):
        super().__init__(False, False, regularization_penalty, regularization_method)


class StaticEpochNoRegularization(StaticEpoch):
    def __init__(self):
        super().__init__(0., None)


class Schedule:
    def __init__(self, epochs):
        self.epochs = epochs

    def __iter__(self):
        return self.epochs.__iter__()

    def __len__(self):
        return len(self.epochs)

    def __str__(self):
        text = ''.join([str(epoch) for epoch in self.epochs])
        _hash = hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]
        return f'{_hash}({self.epochs[0].regularization_penalty})'

    def __repr__(self):
        return self.__str__()
