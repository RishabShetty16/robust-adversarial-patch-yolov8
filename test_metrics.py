from evaluation.metrics import compute_metrics


class DummyBox:

    def __init__(self, conf):
        self.conf = conf


class DummyTensor:

    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class DummyResult:

    def __init__(self):

        self.boxes = [
            DummyBox(DummyTensor(0.9)),
            DummyBox(DummyTensor(0.8)),
            DummyBox(DummyTensor(0.7)),
        ]


result = DummyResult()

metrics = compute_metrics(result)

print(metrics)