from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.util.misc import crossover_mask
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.repair.rounding import RoundingRepair
from utils.debug import *
import numpy as np


class DummyCrossover(Crossover):
    def __init__(self, args):
        super().__init__(2, 2, prob=args.crossover_prob)
    
    def _do(self, problem, X, **kwargs):
        return X
    

###################################################################
#  Grid Guide crossover
###################################################################

class GuidGuideSBXCrossover(SBX):
    def __init__(self, args):
        super().__init__(
            repair=RoundingRepair(),
            prob=args.crossover_prob,
            prob_var=args.sbx_prob_var,
            eta=args.sbx_eta,
            prob_exch=args.sbx_prob_exch,
            prob_bin=args.sbx_prob_bin,
            n_offsprings=2,
        )

class MaskGuidedOptimizationUniformCrossover(UX):
    def __init__(self, args):
        super(MaskGuidedOptimizationUniformCrossover, self).__init__(
            prob=args.crossover_prob
        )
        self.args = args
        self.prob_var = float(args.uniform_prob_var)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        mask = np.random.random((n_matings, n_var)) < self.prob_var
        return crossover_mask(X, mask)

###################################################################
#  SP crossover
###################################################################

class SPOrderCrossover(OrderCrossover):
    def __init__(self, args):
        super(SPOrderCrossover, self).__init__(shift=False)
        self.args = args
    
    def _do(self, problem, X, **kwargs):
        _, _, n_var = X.shape
        node_cnt = n_var // 2
        X1 = X[:, :, :node_cnt]
        X2 = X[:, :, node_cnt:]
        
        X1 = super(SPOrderCrossover, self)._do(problem=problem, X=X1)
        X2 = super(SPOrderCrossover, self)._do(problem=problem, X=X2)

        X = np.concatenate([X1, X2], axis=-1)
        return X

###################################################################
#  Hyperparameter crossover
###################################################################

class HyperparameterUniformCrossover(UX):
    def __init__(self, args):
        super(
            HyperparameterUniformCrossover, self
        ).__init__()
        self.args = args
    
