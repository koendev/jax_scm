import jax
import jax.numpy as jnp

from scm.interfaces import ProgVars, DiagVars
import pytest


@pytest.fixture
def v() -> ProgVars:
    """ProgVar Test Fixture"""
    return ProgVars(
        u=jnp.array([1.0, 2.0, 3.0, 4.0]),
        v=jnp.array([4.0, 5.0, 6.0, 7.0]),
        th=jnp.array([7.0, 8.0, 9.0, 10.0]),
        q=jnp.array([0.1, 0.2, 0.3, 0.4]),
    )


@pytest.fixture
def d() -> DiagVars:
    """DiagVar Test Fixture"""
    return DiagVars(
        u_w=jnp.array([1.0, 2.0, 3.0, 4.0]),
        v_w=jnp.array([4.0, 5.0, 6.0, 7.0]),
        w_th=jnp.array([7.0, 8.0, 9.0, 10.0]),
        w_q=jnp.array([0.1, 0.2, 0.3, 0.4]),
    )


def test_jit(v: ProgVars, d: DiagVars):
    """Test JIT compilation of ProgVars and DiagVars. Should not raise an error."""

    def _test_prog(v: ProgVars):
        a = v.u + v.v + v.th
        return a

    def _test_diag(v: DiagVars):
        b = v.u_w + v.v_w + v.w_th
        return b

    jax.jit(_test_prog)(v)
    jax.jit(_test_diag)(d)


def test_as_tensor(v: ProgVars, d: DiagVars):
    """Test as_tensor methods of ProgVars and DiagVars."""
    tensor = v.as_tensor()
    assert tensor.shape == (4, 4)  # 4 variables, 4 elements each
    assert jnp.all(tensor[0] == v.u)
    assert jnp.all(tensor[1] == v.v)
    assert jnp.all(tensor[2] == v.th)

    tensor_diag = d.as_tensor()
    assert tensor_diag.shape == (4, 4)  # 4 variables, 4 elements each
    assert jnp.all(tensor_diag[0] == d.u_w)
    assert jnp.all(tensor_diag[1] == d.v_w)
    assert jnp.all(tensor_diag[2] == d.w_th)


def test_from_tensor(v: ProgVars, d: DiagVars):
    """Test from_tensor methods of ProgVars and DiagVars."""
    tensor = v.as_tensor()
    new_v = ProgVars.from_tensor(tensor)
    assert isinstance(new_v, ProgVars)
    assert jnp.all(new_v.u == v.u)
    assert jnp.all(new_v.v == v.v)
    assert jnp.all(new_v.th == v.th)

    tensor_diag = d.as_tensor()
    new_d = DiagVars.from_tensor(tensor_diag)
    assert isinstance(new_d, DiagVars)
    assert jnp.all(new_d.u_w == d.u_w)
    assert jnp.all(new_d.v_w == d.v_w)
    assert jnp.all(new_d.w_th == d.w_th)


def test_as_tensor_compile(v: ProgVars):
    """Test that as_tensor method can be compiled."""

    @jax.jit
    def _test_as_tensor(v: ProgVars) -> jnp.ndarray:
        return v.as_tensor()

    tensor = _test_as_tensor(v)
    assert tensor.shape == (4, 4)  # 4 variables, 4 elements each
    assert jnp.all(tensor[0] == v.u)
    assert jnp.all(tensor[1] == v.v)
    assert jnp.all(tensor[2] == v.th)


def test_from_tensor_compile(v: ProgVars):
    """Test that from_tensor method can be compiled."""

    @jax.jit
    def _test_from_tensor(tensor: jnp.ndarray) -> ProgVars:
        return ProgVars.from_tensor(tensor)

    tensor = v.as_tensor()
    new_v = _test_from_tensor(tensor)
    assert isinstance(new_v, ProgVars)
    assert jnp.all(new_v.u == v.u)
    assert jnp.all(new_v.v == v.v)
    assert jnp.all(new_v.th == v.th)
