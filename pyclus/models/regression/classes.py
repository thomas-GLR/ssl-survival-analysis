from pyclus.models import ClusModelRegression, ClusModelTree, ClusModelEnsemble, ClusModelRelief


class RegressionTree(ClusModelRegression, ClusModelTree):
    def __init__(self, **kwargs):
        ClusModelTree.__init__(self, **kwargs)


class RegressionEnsemble(ClusModelRegression, ClusModelEnsemble):
    def __init__(self, **kwargs):
        ClusModelEnsemble.__init__(self, **kwargs)


class RegressionRelief(ClusModelRegression, ClusModelRelief):
    def __init__(self, **kwargs):
        ClusModelRelief.__init__(self, **kwargs)
