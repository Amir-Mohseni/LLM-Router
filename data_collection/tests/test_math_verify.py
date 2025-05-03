import pytest
from math_verify import verify, parse

def verify_answers(x, y):
    """Verify answers in both directions."""
    return verify(x, y) or verify(y, x)

def test_verify_normal():
    assert verify_answers(parse("x=2"), parse("2")) == True
    assert verify_answers(parse("2"), parse("x=3")) == False
    assert verify_answers(parse("x=2"), parse("m=3")) == False
    assert verify_answers(parse("x=2"), parse("m=2")) == True
    
    assert verify_answers(parse("\\boxed{Bob}"), parse("\\boxed{\\text{Bob}}")) == True
    assert verify_answers(parse("\\boxed{Bob}"), parse("\\boxed{\\text{Alice}}")) == False

def test_verify_basic():
    assert verify_answers(parse("1 + 1"), parse("2")) == True
    assert verify_answers(parse("1 + 1"), parse("3")) == False

def test_verify_with_multiple_equations():
    assert verify_answers(parse("x + y = 3\ny = 1"), parse("x = 2\ny = 1")) == True
    assert verify_answers(parse("x + y = 3\ny = 1"), parse("x = 1\ny = 2")) == False
    
def test_verify_with_mcq():
    assert verify_answers(parse("\\boxed{G}"), parse("\\boxed{G}")) == True
    assert verify_answers(parse("\\boxed{G}"), parse("\\boxed{B}")) == False
    
if __name__ == "__main__":
    pytest.main()