from itertools import islice


def window(seq, n):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class EWM:
    def __init__(self, span):
        self.alpha = 2 / (span + 1)
        self.mean = 0

    def add_value(self, value):
        self.mean = value * self.alpha + self.mean * (1 - self.alpha)

    def __str__(self):
        return f"{self.mean:.3f}"
