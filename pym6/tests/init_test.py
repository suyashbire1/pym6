from pym6 import Variable, Domain, Plotter
gv = Variable.GridVariable
Initializer = Domain.Initializer

def test_addition():
    a = 1
    b = 2
    assert a+b == 3
