"""SDE utilities"""

import jax
import jax.numpy as jnp
import equinox as eqx

from diffrax import (
    diffeqsolve,
    ControlTerm,
    Euler,
    ItoMilstein,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
)

from onsagernet._utils import default_floating_dtype

# ------------------------- Typing imports ------------------------- #

from onsagernet.dynamics import SDE
from typing import Optional, Callable
from jax.typing import ArrayLike, DTypeLike
from jax import Array
from jax.random import PRNGKey
from diffrax._solution import Solution

METHOD_ALIASES = {
    "euler": Euler,
    "milstein": ItoMilstein,
}


class SDEIntegrator:
    """SDE solver wrapper of diffrax."""

    def __init__(
        self,
        model: SDE,
        state_dim: int,
        bm_dim: Optional[int] = None,
        method: Optional[str] = "euler",
    ):
        """SDE solver wrapper of diffrax.

        Args:
            model (SDE): SDE to be solved
            state_dim (int): dimension of the state of the SDE
            bm_dim (Optional[int], optional): dimension of the Brownian motion. Defaults to None, meaning that the dimension is the same as the state dimension.
        """
        self.model = model
        self.state_dim = state_dim
        self.bm_dim = bm_dim or state_dim
        self.solver = METHOD_ALIASES.get(method)()  # Get class and instantiate in one line

    def _build_paralle_solver(
        self,
        t0: float,
        t1: float,
        dt: float,
        args: ArrayLike,
        dt_rtol: float,
        max_steps: int,
        dtype: Optional[DTypeLike],
        num_steps: Optional[int] = None,
    ) -> Callable[[Array, Array], Array]:
        dtype = dtype or default_floating_dtype()
        bm_shape = jax.ShapeDtypeStruct(shape=(self.bm_dim,), dtype=dtype)

        # Use num_steps for precise control if provided, otherwise use t1
        if num_steps is not None:
            t1_precise = t0 + num_steps * dt
            ts = jnp.linspace(t0, t1_precise, num_steps + 1)  # +1 to include both endpoints
        else:
            ts = jnp.arange(t0, t1, dt)
            t1_precise = t1

        saveat = SaveAt(ts=ts)

        @eqx.filter_jit
        @jax.vmap
        def parallel_solve(init: Array, key: Array) -> Array:
            brownian_motion = VirtualBrownianTree(
                t0, t1_precise, tol=dt_rtol * dt, shape=bm_shape, key=key
            )
            terms = MultiTerm(
                ODETerm(self.model.drift),
                ControlTerm(self.model.diffusion, brownian_motion),
            )
            sol = diffeqsolve(
                terms,
                self.solver,  # Use pre-created solver
                t0,
                t1_precise,
                dt0=dt,
                y0=init,
                saveat=saveat,
                max_steps=max_steps,
                args=args,
            )
            return sol

        return parallel_solve

    def parallel_solve(
        self,
        initial_conditions: ArrayLike,
        key: PRNGKey,
        t0: float,
        t1: float,
        dt: float,
        args: ArrayLike,
        dt_rtol: float = 0.1,
        max_steps: int = 10000,
        dtype: Optional[DTypeLike] = None,
        num_steps: Optional[int] = None,
    ) -> Solution:
        """Solve the SDE in parallel using `jax.vmap`.

        Args:
            initial_conditions (ArrayLike): initial conditions of size (num_runs, state_dim)
            key (PRNGKey): random key
            t0 (float): initial time
            t1 (float): final time (ignored if num_steps is provided)
            dt (float): time step size
            args (ArrayLike): arguments to pass to drift and diffusion terms
            dt_rtol (float, optional): relative tolerance of Brownian motion. Defaults to 0.1.
            max_steps (int, optional): maximum number of solver steps. Defaults to 10000.
            dtype (Optional[DTypeLike], optional): data type. Defaults to None.
            num_steps (Optional[int], optional): number of time steps. If provided, t1 will be computed as t0 + num_steps * dt. Defaults to None.

        Returns:
            Solution: the solution of the SDE
        """
        solver = self._build_paralle_solver(t0, t1, dt, args, dt_rtol, max_steps, dtype, num_steps)
        return solver(initial_conditions, key)
