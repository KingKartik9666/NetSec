"""Recovery must extend the budget without weakening numerical convergence."""
from experiment.train_hybrids_optimizer_compat import bounded_minimize


def test_only_classwise_iteration_budget_changes():
    calls=[]
    def optimizer(fun,initial,**kwargs):
        calls.append(kwargs); return 'result'
    options={'maxiter':150,'ftol':1e-9,'gtol':1e-6}
    assert bounded_minimize(optimizer,None,None,method='L-BFGS-B',options=options)=='result'
    assert calls[0]['options']=={'maxiter':500,'ftol':1e-9,'gtol':1e-6}
    assert options['maxiter']==150
    bounded_minimize(optimizer,None,None,method='SLSQP',options={'maxiter':200})
    assert calls[1]['options']['maxiter']==200
