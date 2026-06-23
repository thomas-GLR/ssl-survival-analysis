from pyclus.models import ClusModelClassification, ClusModelTree, ClusModelEnsemble, \
    ClusModelRelief


class ClassificationTree(ClusModelClassification, ClusModelTree):
    def __init__(self, **kwargs):
        ClusModelTree.__init__(self, **kwargs)


class ClassificationEnsemble(ClusModelClassification, ClusModelEnsemble):
    def __init__(self, **kwargs):
        ClusModelEnsemble.__init__(self, **kwargs)


class ClassificationRelief(ClusModelClassification, ClusModelRelief):
    def __init__(self, **kwargs):
        ClusModelRelief.__init__(self, **kwargs)
