from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Quaternion:
    """Class for manipulating quaternion tensors with JAX.

    A quaternion is represented by [w, x, y, z] where w is the scalar part
    and (x, y, z) is the vector part.
    """

    def __init__(
        self,
        w: float | jnp.ndarray,
        x: float | jnp.ndarray,
        y: float | jnp.ndarray,
        z: float | jnp.ndarray,
    ) -> None:
        """Initialize a tensor of quaternions.

        Args:
            w, x, y, z: components of the quaternions.
        """
        w, x, y, z = jnp.broadcast_arrays(w, x, y, z)
        self.wxyz = jnp.stack([w, x, y, z], axis=-1)

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        """Flatten the Quaternion PyTree."""
        return (self.wxyz,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Quaternion:
        """Unflatten The Quaternion PyTree"""
        # Create an instance directly without going through from_array to avoid tracer issues
        instance = cls.__new__(cls)
        instance.wxyz = children[0]
        return instance

    @classmethod
    def from_array(cls, array: Array) -> Quaternion:
        """Create a Quaternion array from a numeric array of shape (..., 4).

        Args:
            array: array of shape (..., 4) where the last dimension is [w, x, y, z]
        """
        # Handle JAX tracers and arrays properly
        if not isinstance(array, jnp.ndarray):
            array = jnp.asarray(array)

        if array.shape[-1:] != (4,):
            raise ValueError(f'Array must have shape (..., 4), got {array.shape}')

        instance = cls.__new__(cls)
        instance.wxyz = array
        return instance

    @classmethod
    def from_scalar_vector(cls, scalar: Array, vector: Array) -> Quaternion:
        """Create a quaternion from scalar and vector parts.

        Args:
            scalar: Array of shape (...,) for the scalar part.
            vector: Array of shape (..., 3) for the vector part.

        Returns:
            Quaternion
        """
        if vector.shape[-1:] != (3,):
            raise ValueError(f'Vector must have shape (..., 3), got {vector.shape}')
        scalar = jnp.expand_dims(scalar, axis=-1)
        return cls.from_array(jnp.concatenate([scalar, vector], axis=-1))

    @classmethod
    def from_rotation_matrix(cls, rot: Array) -> Quaternion:
        """Create the quaternion associated to a rotation matrix.

        Args:
            rot: Array of shape (..., 3, 3) representing the rotation matrix

        Returns:
            The normalized Quaternion tensor representing the rotation matrix.
        """
        if rot.shape[-2:] != (3, 3):
            raise ValueError(f'Rotation matrix must have shape (..., 3, 3), got {rot.shape}')

        # Implémentation de la conversion matrice -> quaternion
        trace = jnp.trace(rot, axis1=-2, axis2=-1)

        # Cas où trace > 0
        s = jnp.sqrt(trace + 1.0) * 2  # s = 4 * w
        w = 0.25 * s
        x = (rot[..., 2, 1] - rot[..., 1, 2]) / s
        y = (rot[..., 0, 2] - rot[..., 2, 0]) / s
        z = (rot[..., 1, 0] - rot[..., 0, 1]) / s

        return cls.from_array(jnp.stack([w, x, y, z], axis=-1))

    @classmethod
    def zeros(cls, shape: tuple[int, ...] = (), dtype: jnp.dtype = jnp.float32) -> Quaternion:
        """Create quaternions with all components set to 0.

        Args:
            shape: Shape of the tensor (without the last dimension).
            dtype: Data type of the quaternion components.

        Returns:
            Quaternion with all components equal to 0.
        """
        data = jnp.zeros(shape + (4,), dtype=dtype)
        return cls.from_array(data)

    @classmethod
    def ones(cls, shape: tuple[int, ...] = (), dtype: jnp.dtype = jnp.float32) -> Quaternion:
        """Create quaternions with scalar component set to 1 and vector components set to 0.

        Args:
            shape: Shape of the tensor (without the last dimension).
            dtype: Data type of the quaternion components.

        Returns:
            Quaternions with w=1 and x=y=z=0.
        """
        data = jnp.zeros(shape + (4,), dtype=dtype)
        data = data.at[..., 0].set(1.0)
        return cls.from_array(data)

    @classmethod
    def full(
        cls, shape: tuple[int, ...], fill_value: float, dtype: jnp.dtype = jnp.float32
    ) -> Quaternion:
        """Create quaternions with scalar component set to a value and vector components set to 0.

        Args:
            shape: Shape of the tensor (without the last dimension).
            fill_value: Value to fill the scalar component with.
            dtype: Data type of the quaternion components.

        Returns:
            Quaternions with w=fill_value and x=y=z=0.
        """
        data = jnp.zeros(shape + (4,), dtype=dtype)
        data = data.at[..., 0].set(fill_value)
        return cls.from_array(data)

    @classmethod
    def random(cls, key: jax.random.PRNGKey, shape: tuple[int, ...] = ()) -> Quaternion:
        """Generate normalized random quaternions.

        Args:
            key: Key PRNG.
            shape: Shape of the tensor (without the last dimension).

        Returns:
            Normalized Quaternion.
        """
        data = jax.random.normal(key, shape + (4,))
        return Quaternion.from_array(data).normalize()

    @property
    def w(self) -> Array:
        return self.wxyz[..., 0]

    @property
    def x(self) -> Array:
        return self.wxyz[..., 1]

    @property
    def y(self) -> Array:
        return self.wxyz[..., 2]

    @property
    def z(self) -> Array:
        return self.wxyz[..., 3]

    @property
    def vector(self) -> Array:
        """Vector part (..., 3)"""
        return self.wxyz[..., 1:]

    def norm(self) -> Array:
        """Quaternion norm."""
        return jnp.sqrt(jnp.sum(self.wxyz**2, axis=-1))

    def normalize(self) -> Quaternion:
        """Normalize the quaternion.

        Returns the normalized quaternion. If the quaternion has zero norm,
        returns the zero quaternion [0, 0, 0, 0].
        """
        norm = self.norm()
        # Avoid division by zero
        safe_norm = jnp.where(norm == 0, 1.0, norm)
        return Quaternion.from_array(self.wxyz / jnp.expand_dims(safe_norm, axis=-1))

    def _inverse(self) -> Quaternion:
        """Quaternion inverse (private method - use 1/q instead)."""
        conj = self.conj()
        norm_sq = self.norm() ** 2
        return Quaternion.from_array(conj.wxyz / jnp.expand_dims(norm_sq, axis=-1))

    def to_components(self) -> tuple[Array, Array, Array, Array]:
        return self.w, self.x, self.y, self.z

    def to_rotation_matrix(self) -> Array:
        """Convert quaternion to rotation matrix.

        Returns:
            Array of shape (..., 3, 3)
        """
        # Normalize the quaternion
        q = self.normalize()
        w, x, y, z = q.to_components()

        # Calculate matrix elements
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot = jnp.stack(
            [
                jnp.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
                jnp.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
                jnp.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
            ],
            axis=-2,
        )

        return rot

    def rotate_vector(self, v: Array) -> Array:
        """Apply quaternion rotation to a vector.

        Args:
            v: Array of shape (..., 3) representing vectors

        Returns:
            Array of shape (..., 3) representing rotated vectors
        """
        # Convert vector to pure quaternion
        v_quat = Quaternion(0, v[..., 0], v[..., 1], v[..., 2])
        # Apply rotation: q * v * q^-1
        result = self * v_quat * self._inverse()

        return result.vector

    def __repr__(self) -> str:
        if self.shape == ():
            w, x, y, z = self.wxyz
            return f'{w} + {x}i + {y}j + {z}k'
        return f'Quaternion(shape={self.shape}, dtype={self.dtype})'

    #######################
    # JAX array interface #
    #######################

    def __len__(self):
        """Length of the first axis."""
        if self.ndim == 0:
            raise TypeError('len() of unsized object')
        return self.shape[0]

    def __iter__(self):
        """Iterate over the first axis."""
        if self.ndim == 0:
            raise TypeError('iteration over a 0-d quaternion')
        for i in range(self.shape[0]):
            yield Quaternion.from_array(self.wxyz[i])

    def __neg__(self) -> Quaternion:
        """Quaternion negation."""
        return Quaternion.from_array(-self.wxyz)

    def __add__(self, other: Any) -> Quaternion:
        """Quaternion addition."""
        if isinstance(other, Quaternion):
            return Quaternion.from_array(self.wxyz + other.wxyz)

        if isinstance(other, int | float | jnp.ndarray):
            return Quaternion.from_scalar_vector(self.w + other, self.vector)

        raise NotImplementedError

    def __radd__(self, other: Any) -> Quaternion:
        """Quaternion addition."""
        return self.__add__(other)

    def __sub__(self, other: Any) -> Quaternion:
        """Quaternion subtraction."""
        if isinstance(other, Quaternion):
            return Quaternion.from_array(self.wxyz - other.wxyz)

        if isinstance(other, int | float | jnp.ndarray):
            return Quaternion.from_scalar_vector(self.w - other, self.vector)

        raise NotImplementedError

    def __rsub__(self, other: Any) -> Quaternion:
        """Quaternion subtraction."""
        if isinstance(other, int | float | jnp.ndarray):
            return Quaternion.from_scalar_vector(other - self.w, -self.vector)

        raise NotImplementedError

    def __mul__(self, other: Any) -> Quaternion:
        """Quaternion multiplication."""
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.to_components()
            w2, x2, y2, z2 = other.to_components()

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return Quaternion(w, x, y, z)

        if isinstance(other, int | float):
            return Quaternion.from_array(self.wxyz * other)

        if isinstance(other, jnp.ndarray):
            return Quaternion.from_array(self.wxyz * jnp.expand_dims(other, axis=-1))

        return NotImplemented

    def __rmul__(self, other: Any) -> Quaternion:
        """Quaternion multiplication."""
        if isinstance(other, int | float):
            return Quaternion.from_array(other * self.wxyz)

        if isinstance(other, jnp.ndarray):
            return Quaternion.from_array(jnp.expand_dims(other, axis=-1) * self.wxyz)

        return NotImplemented

    def __truediv__(self, other: Any) -> Quaternion:
        """Quaternion division."""
        if isinstance(other, Quaternion):
            return self * other._inverse()

        if isinstance(other, int | float):
            return Quaternion.from_array(self.wxyz / other)

        if isinstance(other, jnp.ndarray):
            return Quaternion.from_array(self.wxyz / jnp.expand_dims(other, axis=-1))

        return NotImplemented

    def __rtruediv__(self, other: Any) -> Quaternion:
        """Quaternion division."""
        if isinstance(other, int | float) and other == 1:
            return self._inverse()

        if isinstance(other, int | float | jnp.ndarray):
            return other * self._inverse()

        return NotImplemented

    def __pow__(self, exponent: float | int | Array) -> Quaternion:
        """Quaternion exponentiation q^n.

        For integer exponents, uses optimized special cases.
        For non-integer exponents, uses the general formula: q^n = exp(n * log(q))

        Args:
            exponent: The exponent (scalar or array)

        Returns:
            The quaternion raised to the given power
        """
        # Handle special cases for integer exponents only
        if isinstance(exponent, int | float):
            if exponent == -2:
                q_inv = self._inverse()
                return q_inv * q_inv
            elif exponent == -1:
                return self._inverse()
            elif exponent == 0:
                return Quaternion.ones(self.shape, self.dtype)
            elif exponent == 1:
                return self
            elif exponent == 2:
                return self * self
            return (exponent * self.log()).exp()

        # General case: q^n = exp(n * log(q))
        result = (exponent * self.log()).exp().wxyz
        return Quaternion.from_array(
            jnp.where(
                exponent[..., None] == 0, jnp.array([1.0, 0.0, 0.0, 0.0], dtype=self.dtype), result
            )
        )

    def log(self) -> Quaternion:
        """Compute quaternion logarithm.

        For a quaternion q = ‖q‖ * (cos(θ) + sin(θ)v), the logarithm is:
        log(q) = log(‖q‖) + θ * v

        Returns:
            The logarithm of the quaternion
        """
        # Get norm and handle zero quaternion
        q_norm = self.norm()

        # Normalize to get unit quaternion (handles zero quaternions safely)
        # Note that the norm is not computed again for jitted functions.
        unit_q = self.normalize()

        # For unit quaternion q = cos(θ) + sin(θ)v, compute θ and v
        # θ = arccos(w) and v = vector/|vector|
        theta = jnp.arccos(jnp.clip(unit_q.w, -1.0, 1.0))
        vector_norm = jnp.linalg.norm(unit_q.vector, axis=-1)

        # Handle case where vector is zero (real quaternion)
        inv_vector_norm = jnp.where(vector_norm == 0, 0.0, 1 / vector_norm)
        unit_vector = unit_q.vector * inv_vector_norm[..., None]

        # log(q) = log(|q|) + θ * v
        log_norm = jnp.log(q_norm)
        log_q_vector = theta[..., None] * unit_vector

        return Quaternion.from_scalar_vector(log_norm, log_q_vector)

    def exp(self) -> Quaternion:
        """Compute quaternion exponential.

        For a quaternion q = s + v, the exponential is:
        exp(q) = exp(s) * (cos(‖v‖) + sin(‖v‖) * v/‖v‖)

        Returns:
            The exponential of the quaternion
        """
        scalar_part = self.w
        vector_part = self.vector
        vector_norm = jnp.linalg.norm(vector_part, axis=-1)

        # exp(s + v) = exp(s) * (cos(|v|) + sin(|v|) * v/|v|)
        exp_scalar = jnp.exp(scalar_part)
        cos_vnorm = jnp.cos(vector_norm)
        sin_vnorm = jnp.sin(vector_norm)

        # Handle case where |v| = 0 (real quaternion)
        inv_vector_norm = jnp.where(vector_norm == 0, 0.0, 1 / vector_norm)
        unit_v = vector_part * jnp.expand_dims(inv_vector_norm, -1)

        result_w = exp_scalar * cos_vnorm
        result_vector = exp_scalar * jnp.expand_dims(sin_vnorm, -1) * unit_v

        return Quaternion.from_scalar_vector(result_w, result_vector)

    @property
    def nbytes(self) -> int:
        """Number of bytes in the tensor."""
        return self.wxyz.nbytes

    @property
    def itemsize(self) -> int:
        """Size of one quaternion element in bytes."""
        return self.wxyz.itemsize * 4

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the tensor."""
        return self.wxyz.shape[:-1]

    @property
    def ndim(self):
        """Number of dimensions of the quaternion tensor (without the quaternion dimension)."""
        return self.wxyz.ndim - 1

    @property
    def size(self):
        """Total number of quaternions."""
        return self.wxyz.size >> 2

    @property
    def dtype(self) -> jnp.dtype:
        """Data type."""
        return self.wxyz.dtype

    def reshape(self, *shape) -> Quaternion:
        """Redimensionne le tableau de quaternions"""
        if len(shape) == 0:
            raise ValueError('Must specify at least one dimension')
        if isinstance(shape[0], tuple):
            if len(shape) > 1:
                raise ValueError('Cannot specify more than one shape')
            shape = shape[0]
        new_shape = shape + (4,)
        return self.from_array(self.wxyz.reshape(new_shape))

    def flatten(self) -> Quaternion:
        """Aplatis le tableau de quaternions"""
        return self.from_array(self.wxyz.reshape(-1, 4))

    def ravel(self) -> Quaternion:
        """Aplatis le tableau de quaternions"""
        return self.flatten()

    def squeeze(self, axis=None) -> Quaternion:
        """Supprime les dimensions de taille 1"""
        return Quaternion.from_array(jnp.squeeze(self.wxyz, axis=axis))

    def conjugate(self) -> Quaternion:
        """Quaternion conjugate."""
        sign = jnp.array([1, -1, -1, -1])
        return Quaternion.from_array(self.wxyz * sign)

    def conj(self) -> Quaternion:
        """Quaternion conjugate."""
        return self.conjugate()

    def block_until_ready(self) -> None:
        """Block until all pending computations are done."""
        self.wxyz.block_until_ready()

    @property
    def device(self) -> jax.Device[Any]:
        return self.wxyz.device

    def devices(self) -> set[jax.Device[Any]]:
        return self.wxyz.devices()

    def slerp(self, other: Quaternion, t: float | Array) -> Quaternion:
        """Spherical linear interpolation between two quaternions.

        Args:
            other: Target quaternion to interpolate towards
            t: Interpolation parameter in [0, 1]. t=0 returns self, t=1 returns other

        Returns:
            Interpolated quaternion
        """
        # Ensure both quaternions are normalized
        q1 = self.normalize()
        q2 = other.normalize()

        # Compute dot product
        dot = jnp.sum(q1.wxyz * q2.wxyz, axis=-1)

        # If dot product is negative, slerp won't take the shorter path.
        # Note that this is necessary to handle the double cover of SO(3)
        # by unit quaternions: q and -q represent the same rotation.
        q2_corrected = jnp.where(jnp.expand_dims(dot < 0, -1), -q2.wxyz, q2.wxyz)
        dot = jnp.abs(dot)

        # If quaternions are very close, use linear interpolation to avoid numerical issues
        threshold = 0.9995
        use_linear = dot > threshold

        # Linear interpolation case
        result_linear = q1.wxyz + jnp.expand_dims(t * (1 - t), -1) * (q2_corrected - q1.wxyz)
        result_linear = Quaternion.from_array(result_linear).normalize()

        # Spherical interpolation case
        theta = jnp.arccos(jnp.clip(dot, 0.0, 1.0))
        sin_theta = jnp.sin(theta)

        # Avoid division by zero
        safe_sin_theta = jnp.where(sin_theta == 0, 1.0, sin_theta)

        factor1 = jnp.sin((1 - t) * theta) / safe_sin_theta
        factor2 = jnp.sin(t * theta) / safe_sin_theta

        result_slerp = (
            jnp.expand_dims(factor1, -1) * q1.wxyz + jnp.expand_dims(factor2, -1) * q2_corrected
        )
        result_slerp = Quaternion.from_array(result_slerp)

        # Choose between linear and spherical interpolation
        result = jnp.where(jnp.expand_dims(use_linear, -1), result_linear.wxyz, result_slerp.wxyz)

        return Quaternion.from_array(result)
