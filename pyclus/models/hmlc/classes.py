from pyclus.models import ClusModelHierarchicalMultiLabelClassification, \
    ClusModelTree, ClusModelEnsemble, ClusModelRelief


class HMLCTree(ClusModelHierarchicalMultiLabelClassification, ClusModelTree):
    def __init__(self, **kwargs):
        ClusModelTree.__init__(self, **kwargs)


class HMLCEnsemble(ClusModelHierarchicalMultiLabelClassification, ClusModelEnsemble):
    def __init__(self, **kwargs):
        ClusModelEnsemble.__init__(self, **kwargs)


class HMLCRelief(ClusModelHierarchicalMultiLabelClassification, ClusModelRelief):
    def __init__(self, **kwargs):
        ClusModelRelief.__init__(self, **kwargs)

