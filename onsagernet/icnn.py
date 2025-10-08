# icnn_equinox.py
from __future__ import annotations
from typing import Sequence, List
import jax
import jax.numpy as jnp
import equinox as eqx


class PositiveLinear(eqx.Module):
    """Linear layer with elementwise-nonnegative weight via softplus."""

    weight_raw: jnp.ndarray  # (out, in)
    bias: jnp.ndarray  # (out,)

    def __init__(self, in_size: int, out_size: int, key: jax.Array):
        k1, k2 = jax.random.split(key)
        # small init so softplus(weight_raw) ~ positive small numbers
        self.weight_raw = jax.random.normal(k1, (out_size, in_size)) * 0.02
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        W = jax.nn.softplus(self.weight_raw)  # >= 0 elementwise
        return W @ x + self.bias


class Affine(eqx.Module):
    """Standard affine layer (unconstrained)."""

    weight: jnp.ndarray  # (out, in)
    bias: jnp.ndarray  # (out,)

    def __init__(self, in_size: int, out_size: int, key: jax.Array):
        k1, k2 = jax.random.split(key)
        # Xavier-like small init
        lim = jnp.sqrt(6.0 / (in_size + out_size))
        self.weight = jax.random.uniform(
            k1, (out_size, in_size), minval=-lim, maxval=lim
        )
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.weight @ x + self.bias


class ICNN(eqx.Module):
    """Input-Convex Neural Network: convex in x; output is scalar.

    z_{k+1} = phi( W_k z_k + U_k x + b_k ), with W_k >= 0 elementwise.
    Final readout is a >=0 linear form on z_L plus optional (mu/2)||x||^2.
    """

    # Layer lists
    W_layers: List[PositiveLinear]  # acts on z (constrained >=0)
    U_layers: List[Affine]  # acts on x (unconstrained)
    # Readout on z_L with nonnegative weights
    readout_pos: PositiveLinear  # maps z_L -> 1 (nonnegative weights)
    # Optional strong convexity margin
    mu: float

    # activation is convex & nondecreasing
    def _phi(self, u: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.softplus(u)  # smooth, convex, nondecreasing

    def __init__(
        self,
        d_in: int,
        widths: Sequence[int],  # e.g. [64, 64, 64]
        key: jax.Array,
        mu: float = 0.0,  # set >0 for strong convexity
    ):
        self.mu = float(mu)
        keys = jax.random.split(key, num=2 * len(widths) + 2)
        # Build hidden stacks
        W_layers: List[PositiveLinear] = []
        U_layers: List[Affine] = []

        # z_0 := 0 vector (implicit). First layer takes z_0 and x.
        # W_0 acts on z_0; we still include it for biasing capacity.
        # in_z = 0  # conceptual; layer handles shapes internally
        # prev_width = 0

        # For implementation simplicity, we treat W_k as acting on current z_k
        # with known width; for k=0 we define z_0 as zero vector of 'widths[0]'.
        # That’s equivalent to having only U_0 x + b_0 at the first layer.
        # We just set W_0 as a PositiveLinear on that width; it won't harm convexity.

        # First layer: z_1 = phi(U_0 x + b_0)       (W_0 z_0 term is effectively bias)
        U_layers.append(Affine(d_in, widths[0], keys[0]))
        W_layers.append(PositiveLinear(widths[0], widths[0], keys[1]))

        # Hidden layers
        for i in range(1, len(widths)):
            U_layers.append(Affine(d_in, widths[i], keys[2 * i + 0]))
            W_layers.append(PositiveLinear(widths[i - 1], widths[i], keys[2 * i + 1]))

        self.W_layers = W_layers
        self.U_layers = U_layers

        # Readout: a^T z_L + b, with a >= 0 to preserve convexity
        self.readout_pos = PositiveLinear(widths[-1], 1, keys[-1])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: shape (d_in,) -> returns scalar shape ()"""
        # Forward hidden stack
        z = jnp.zeros((self.U_layers[0].bias.shape[0],))  # width_0
        # Layer 0: z = phi(U0 x + b0)   (+ W0 z0 is just bias-like)
        z = self._phi(self.U_layers[0](x) + self.W_layers[0](z))

        for U, W in zip(self.U_layers[1:], self.W_layers[1:]):
            z = self._phi(W(z) + U(x))

        # Nonnegative readout on z
        out = self.readout_pos(z).squeeze()  # shape ()

        # Optional strong convexity margin
        if self.mu > 0.0:
            out = out + 0.5 * self.mu * jnp.vdot(x, x)

        return out
