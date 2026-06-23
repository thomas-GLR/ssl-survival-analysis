from pyclus.models import ClusModelMultiLabelClassification, ClusModelTree, ClusModelEnsemble, \
    ClusModelRelief


class MLCTree(ClusModelMultiLabelClassification, ClusModelTree):
    def __init__(self, **kwargs):
        ClusModelTree.__init__(self, **kwargs)


class MLCEnsemble(ClusModelMultiLabelClassification, ClusModelEnsemble):
    def __init__(self, **kwargs):
        ClusModelEnsemble.__init__(self, **kwargs)


class MLCRelief(ClusModelMultiLabelClassification, ClusModelRelief):
    def __init__(self, **kwargs):
        ClusModelRelief.__init__(self, **kwargs)
