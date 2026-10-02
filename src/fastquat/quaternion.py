import operator
from collections.abc import Sequence
from typing import Any, Self, SupportsIndex

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.spatial.transform import Rotation
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike, DTypeLike

ShapeLike = SupportsIndex | Sequence[SupportsIndex]


def _to_int_tuple(value: ShapeLike) -> tuple[int, ...]:
    """Convert an int or a sequence of ints, such as a shape or axes, to a tuple of ints."""
    if isinstance(value, Sequence):
        return tuple(operator.index(item) for item in value)
    return (operator.index(value),)


@register_pytree_node_class
class Quaternion:
    """Class for manipulating quaternion tensors with JAX.

    A quaternion is represented by [w, x, y, z] where w is the scalar part
    and (x, y, z) is the vector part.
    """

    # Prevent NumPy from iterating over the array and calling __rmul__ element-wise
    __array_ufunc__ = None

    def __init__(
        self,
        w: ArrayLike = 0,
        x: ArrayLike = 0,
        y: ArrayLike = 0,
        z: ArrayLike = 0,
        dtype: DTypeLike | None = None,
    ) -> None:
        """Initialize a tensor of quaternions.

        Args:
            w, x, y, z: components of the quaternions.
            dtype: Data type of the quaternion components (inferred by default).
        """
        w = jnp.asarray(w, dtype=dtype)
        x = jnp.asarray(x, dtype=dtype)
        y = jnp.asarray(y, dtype=dtype)
        z = jnp.asarray(z, dtype=dtype)
        w, x, y, z = jnp.broadcast_arrays(w, x, y, z)
        self.wxyz = jnp.stack([w, x, y, z], axis=-1)

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        """Flatten the Quaternion PyTree."""
        return (self.wxyz,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        """Unflatten The Quaternion PyTree"""
        # Create an instance directly without going through from_array to avoid tracer issues
        instance = cls.__new__(cls)
        instance.wxyz = children[0]
        return instance

    @classmethod
    def from_array(cls, array: ArrayLike) -> Self:
        """Create a Quaternion array from a numeric array of shape (..., 4).

        Args:
            array: array of shape (..., 4) where the last dimension is [w, x, y, z]
        """
        array = jnp.asarray(array)

        if array.shape[-1:] != (4,):
            raise ValueError(f'Array must have shape (..., 4), got {array.shape}')

        instance = cls.__new__(cls)
        instance.wxyz = array
        return instance

    @classmethod
    def from_scalar_vector(cls, scalar: ArrayLike, vector: ArrayLike) -> Self:
        """Create a quaternion from scalar and vector parts.

        The scalar part and the batch dimensions of the vector part are broadcast together.

        Args:
            scalar: Array of shape (...,) for the scalar part.
            vector: Array of shape (..., 3) for the vector part.

        Returns:
            Quaternion
        """
        scalar = jnp.asarray(scalar)
        vector = jnp.asarray(vector)
        if vector.shape[-1:] != (3,):
            raise ValueError(f'Vector must have shape (..., 3), got {vector.shape}')
        shape = jnp.broadcast_shapes(scalar.shape, vector.shape[:-1])
        scalar = jnp.broadcast_to(scalar, shape)[..., None]
        vector = jnp.broadcast_to(vector, shape + (3,))
        return cls.from_array(jnp.concatenate([scalar, vector], axis=-1))

    @classmethod
    def from_rotation_matrix(cls, rot: ArrayLike) -> Self:
        """Create the quaternion associated to a rotation matrix.

        Args:
            rot: Array of shape (..., 3, 3) representing the rotation matrix

        Returns:
            The normalized Quaternion tensor representing the rotation matrix, with w >= 0.
        """
        rot = jnp.asarray(rot)
        if rot.shape[-2:] != (3, 3):
            raise ValueError(f'Rotation matrix must have shape (..., 3, 3), got {rot.shape}')
        rot = rot.astype(jnp.result_type(rot, float))

        m00, m01, m02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
        m10, m11, m12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
        m20, m21, m22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]
        trace = m00 + m11 + m22

        # Each row is 4 q_k q for k = w, x, y, z: proportional to q, with 4 q_k² as its k-th
        # component. Picking the row with the largest q_k² avoids the cancellation of the
        # trace-based formula near rotations by π, and its norm 4 |q_k| >= 2 is safe to divide by.
        candidates = jnp.stack(
            [
                jnp.stack([1 + trace, m21 - m12, m02 - m20, m10 - m01], axis=-1),
                jnp.stack([m21 - m12, 1 + 2 * m00 - trace, m01 + m10, m02 + m20], axis=-1),
                jnp.stack([m02 - m20, m01 + m10, 1 + 2 * m11 - trace, m12 + m21], axis=-1),
                jnp.stack([m10 - m01, m02 + m20, m12 + m21, 1 + 2 * m22 - trace], axis=-1),
            ],
            axis=-2,
        )
        best = jnp.argmax(jnp.stack([trace, m00, m11, m22], axis=-1), axis=-1)
        q = jnp.take_along_axis(candidates, best[..., None, None], axis=-2)[..., 0, :]
        # q and -q are the same rotation: return the one with w >= 0
        q = jnp.where(q[..., :1] < 0, -q, q)
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)

        return cls.from_array(q)

    @classmethod
    def from_axis_angle(cls, axis: ArrayLike, angle: ArrayLike) -> Self:
        """Create the unit quaternion of a rotation about an axis.

        The rotation follows the right-hand rule: a positive angle rotates counterclockwise when
        looking from the tip of the axis towards the origin.

        Args:
            axis: Array of shape (..., 3) for the rotation axis. It does not need to be normalized,
                but must be non-zero. Use `from_rotation_vector` for rotations that can be zero.
            angle: Array of shape (...) for the rotation angle, in radians.

        Returns:
            Quaternion of shape broadcast(axis.shape[:-1], angle.shape).
        """
        axis = jnp.asarray(axis)
        angle = jnp.asarray(angle)
        if axis.shape[-1:] != (3,):
            raise ValueError(f'Axis must have shape (..., 3), got {axis.shape}')
        dtype = jnp.result_type(axis, angle, float)
        unit_axis = axis / jnp.linalg.norm(axis, axis=-1, keepdims=True)
        half_angle = 0.5 * angle.astype(dtype)
        scalar = jnp.cos(half_angle)
        vector = jnp.sin(half_angle)[..., None] * unit_axis.astype(dtype)
        scalar, vector = jnp.broadcast_arrays(scalar[..., None], vector)
        return cls.from_scalar_vector(scalar[..., 0], vector)

    @classmethod
    def from_rotation_vector(cls, rotvec: ArrayLike) -> Self:
        """Create the unit quaternion of a rotation vector.

        The rotation vector is the rotation axis scaled by the rotation angle in radians. The
        rotation follows the right-hand rule, as in `from_axis_angle`. The zero vector gives the
        identity, with exact derivatives.

        Args:
            rotvec: Array of shape (..., 3) for the rotation vectors.

        Returns:
            Quaternion of shape rotvec.shape[:-1].
        """
        rotvec = jnp.asarray(rotvec)
        if rotvec.shape[-1:] != (3,):
            raise ValueError(f'Rotation vector must have shape (..., 3), got {rotvec.shape}')
        rotvec = rotvec.astype(jnp.result_type(rotvec, float))
        # q = exp(rotvec / 2), whose implementation is safe at the zero vector
        half_rotvec = 0.5 * rotvec
        return cls.from_scalar_vector(jnp.zeros_like(half_rotvec[..., 0]), half_rotvec).exp()

    @classmethod
    def from_scipy_rotation(cls, rotation: Any) -> Self:
        """Create the unit quaternion of a rotation object.

        Args:
            rotation: A `jax.scipy.spatial.transform.Rotation`, or any object with the same
                `as_quat` method returning scalar-last quaternions, such as a
                `scipy.spatial.transform.Rotation`.

        Returns:
            Quaternion of shape rotation.as_quat().shape[:-1].
        """
        xyzw = jnp.asarray(rotation.as_quat())
        return cls.from_scalar_vector(xyzw[..., 3], xyzw[..., :3])

    @classmethod
    def from_euler(cls, seq: str, angles: ArrayLike, degrees: bool = False) -> Self:
        """Create the unit quaternion of a sequence of rotations about the coordinate axes.

        This wraps `jax.scipy.spatial.transform.Rotation.from_euler`, which has the same
        convention as scipy.

        Args:
            seq: Sequence of 1 to 3 axes among 'x', 'y' and 'z'. Lowercase letters are extrinsic
                rotations (about the fixed frame axes), uppercase letters are intrinsic rotations
                (about the rotating frame axes). Extrinsic and intrinsic rotations cannot be mixed.
            angles: Array of shape (..., len(seq)) for the rotation angles, in radians unless
                `degrees` is True.
            degrees: Whether the angles are in degrees.

        Returns:
            Quaternion of shape angles.shape[:-1].
        """
        angles = jnp.asarray(angles)
        if angles.shape[-1:] != (len(seq),):
            raise ValueError(
                f'Angles must have shape (..., {len(seq)}) for sequence {seq!r}, got {angles.shape}'
            )
        angles = angles.astype(jnp.result_type(angles, float))
        return cls.from_scipy_rotation(Rotation.from_euler(seq, angles, degrees=degrees))

    @classmethod
    def zeros(cls, shape: ShapeLike, dtype: DTypeLike | None = None) -> Self:
        """Create quaternions with all components set to 0.

        Args:
            shape: Shape of the tensor (without the last dimension).
            dtype: Data type of the quaternion components.

        Returns:
            Quaternion with all components equal to 0.
        """
        data = jnp.zeros(_to_int_tuple(shape) + (4,), dtype=dtype)
        return cls.from_array(data)

    @classmethod
    def ones(cls, shape: ShapeLike, dtype: DTypeLike | None = None) -> Self:
        """Create quaternions with scalar component set to 1 and vector components set to 0.

        Args:
            shape: Shape of the tensor (without the last dimension).
            dtype: Data type of the quaternion components.

        Returns:
            Quaternions with w=1 and x=y=z=0.
        """
        data = jnp.zeros(_to_int_tuple(shape) + (4,), dtype=dtype)
        data = data.at[..., 0].set(1.0)
        return cls.from_array(data)

    @classmethod
    def full(cls, shape: ShapeLike, fill_value: float, dtype: DTypeLike | None = None) -> Self:
        """Create quaternions with scalar component set to a value and vector components set to 0.

        Args:
            shape: Shape of the tensor (without the last dimension).
            fill_value: Value to fill the scalar component with.
            dtype: Data type of the quaternion components.

        Returns:
            Quaternions with w=fill_value and x=y=z=0.
        """
        data = jnp.zeros(_to_int_tuple(shape) + (4,), dtype=dtype)
        data = data.at[..., 0].set(fill_value)
        return cls.from_array(data)

    @classmethod
    def random(cls, key: Array, shape: ShapeLike = (), dtype: DTypeLike | None = None) -> Self:
        """Generate normalized random quaternions.

        Args:
            key: Key PRNG.
            shape: Shape of the tensor (without the last dimension).
            dtype: Data type of the quaternion components.

        Returns:
            Normalized Quaternion.
        """
        data = jax.random.normal(key, _to_int_tuple(shape) + (4,), dtype=dtype)
        return cls.from_array(data).normalize()

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

    def __abs__(self) -> Array:
        """Quaternion norm."""
        return jnp.sqrt(jnp.sum(self.wxyz**2, axis=-1))

    def normalize(self) -> Self:
        """Normalize the quaternion.

        Returns the normalized quaternion. If the quaternion has zero norm,
        returns the quaternion [NaN, NaN, NaN, NaN].
        """
        norm = abs(self)
        return self.from_array(self.wxyz / jnp.expand_dims(norm, axis=-1))

    def _inverse(self) -> Self:
        """Quaternion inverse (private method - use 1/q instead)."""
        conj = self.conj()
        norm_sq = jnp.sum(self.wxyz**2, axis=-1)
        return self.from_array(conj.wxyz / jnp.expand_dims(norm_sq, axis=-1))

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

    def to_rotation_vector(self) -> Array:
        """Convert quaternion to rotation vector.

        The rotation vector is the rotation axis scaled by the rotation angle in radians, with the
        angle in [0, π]. q and -q give the same vector, except for rotations by π, where either of
        the two opposite vectors may be returned. The rotation vector is discontinuous there.
        Non-unit quaternions are treated as their normalized counterpart.

        Returns:
            Array of shape (..., 3)
        """
        # q and -q are the same rotation: pick the one with w >= 0 so that the angle is in [0, π]
        sign = jnp.where(self.w < 0, -1, 1).astype(self.dtype)
        # rotvec = 2 log(q), whose vector part does not depend on |q| and is safe at the identity
        return 2 * (sign * self).log().vector

    def to_scipy_rotation(self) -> Rotation:
        """Convert quaternion to a `jax.scipy.spatial.transform.Rotation`.

        Non-unit quaternions are normalized.

        Returns:
            Rotation of the same shape.
        """
        return Rotation.from_quat(jnp.concatenate([self.vector, self.w[..., None]], axis=-1))

    def to_euler(self, seq: str, degrees: bool = False) -> Array:
        """Convert quaternion to Euler angles.

        This wraps `jax.scipy.spatial.transform.Rotation.as_euler`, which has the same convention
        as scipy. The first and third angles are in [-π, π]. The second angle is in [0, π] if the
        first and third axes are the same (proper Euler angles), and in [-π/2, π/2] otherwise
        (Tait-Bryan angles).

        In gimbal lock, when the second angle is at a bound of its range, only the sum or
        difference of the first and third angles is defined: the third angle is then set to 0.
        The angles are discontinuous there, and their gradients are meaningless (possibly NaN).
        Non-unit quaternions are treated as their normalized counterpart.

        Args:
            seq: Sequence of 3 axes among 'x', 'y' and 'z', with no two consecutive axes the same.
                Lowercase letters are extrinsic rotations, uppercase letters are intrinsic ones.
            degrees: Whether to return the angles in degrees.

        Returns:
            Array of shape (..., 3)
        """
        return self.to_scipy_rotation().as_euler(seq, degrees=degrees)

    def rotate_vector(self, v: ArrayLike) -> Array:
        """Apply quaternion rotation to a vector.

        Args:
            v: Array of shape (..., 3) representing vectors

        Returns:
            Array of shape (..., 3) representing rotated vectors
        """
        v = jnp.asarray(v)

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
            yield self.from_array(self.wxyz[i])

    def __getitem__(self, idx: Any) -> Self:
        """Index or slice the tensor of quaternions."""
        if not isinstance(idx, tuple):
            idx = (idx,)
        return self.from_array(self.wxyz[(*idx, slice(None))])

    def __eq__(self, other: Any) -> Array:  # ty: ignore[invalid-method-override]
        """Element-wise quaternion equality.

        Real scalars and arrays are compared as quaternions with a zero vector part.

        Returns:
            Boolean array of the broadcast shape, True where all four components are equal.
        """
        if isinstance(other, Quaternion):
            return jnp.all(self.wxyz == other.wxyz, axis=-1)

        try:
            other = jnp.asarray(other)
        except (TypeError, ValueError):  # jnp.asarray(None) raises ValueError
            return NotImplemented

        if jnp.iscomplexobj(other):
            raise NotImplementedError('Quaternion and complex comparison is not implemented.')

        return (self.w == other) & jnp.all(self.vector == 0, axis=-1)

    def __ne__(self, other: Any) -> Array:  # ty: ignore[invalid-method-override]
        """Element-wise quaternion inequality."""
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return NotImplemented
        return ~equal

    def __pos__(self) -> Self:
        """Quaternion positive."""
        return self

    def __neg__(self) -> Self:
        """Quaternion negation."""
        return self.from_array(-self.wxyz)

    def __add__(self, other: Any) -> Self:
        """Quaternion addition."""
        if isinstance(other, Quaternion):
            return self.from_array(self.wxyz + other.wxyz)

        try:
            other = jnp.asarray(other)
        except TypeError:
            return NotImplemented

        if jnp.iscomplexobj(other):
            raise NotImplementedError('Quaternion and complex addition is not implemented.')

        return self.from_scalar_vector(self.w + other, self.vector)

    def __radd__(self, other: Any) -> Self:
        """Quaternion addition."""
        return self.__add__(other)

    def __sub__(self, other: Any) -> Self:
        """Quaternion subtraction."""
        if isinstance(other, Quaternion):
            return self.from_array(self.wxyz - other.wxyz)

        try:
            other = jnp.asarray(other)
        except TypeError:
            return NotImplemented

        if jnp.iscomplexobj(other):
            raise NotImplementedError('Quaternion and complex subtraction is not implemented.')

        return self.from_scalar_vector(self.w - other, self.vector)

    def __rsub__(self, other: Any) -> Self:
        """Quaternion subtraction."""
        try:
            other = jnp.asarray(other)
        except TypeError:
            return NotImplemented

        if jnp.iscomplexobj(other):
            raise NotImplementedError('Quaternion and complex subtraction is not implemented.')

        return self.from_scalar_vector(other - self.w, -self.vector)

    def __mul__(self, other: Any) -> Self:
        """Quaternion multiplication."""
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.to_components()
            w2, x2, y2, z2 = other.to_components()

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return self.from_array(jnp.stack([w, x, y, z], axis=-1))

        try:
            other = jnp.asarray(other)
        except TypeError:
            return NotImplemented

        if jnp.iscomplexobj(other):
            raise NotImplementedError('Quaternion and complex multiplication is not implemented.')

        return self.from_array(self.wxyz * jnp.expand_dims(other, axis=-1))

    def __rmul__(self, other: Any) -> Self:
        """Quaternion multiplication."""
        try:
            other = jnp.asarray(other)
        except TypeError:
            return NotImplemented

        if jnp.iscomplexobj(other):
            raise NotImplementedError('Quaternion and complex multiplication is not implemented.')

        return self.from_array(jnp.expand_dims(other, axis=-1) * self.wxyz)

    def __truediv__(self, other: Any) -> Self:
        """Quaternion division."""
        if isinstance(other, Quaternion):
            return self * other._inverse()

        try:
            other = jnp.asarray(other)
        except TypeError:
            return NotImplemented

        if jnp.iscomplexobj(other):
            raise NotImplementedError('Quaternion and complex division is not implemented.')

        return self.from_array(self.wxyz / jnp.expand_dims(other, axis=-1))

    def __rtruediv__(self, other: Any) -> Self:
        """Quaternion division."""
        try:
            other = jnp.asarray(other)
        except TypeError:
            return NotImplemented

        if jnp.iscomplexobj(other):
            raise NotImplementedError('Quaternion and complex division is not implemented.')

        return other * self._inverse()

    def __pow__(self, exponent: ArrayLike) -> Self:
        """Quaternion exponentiation q^n.

        For integer exponents, uses optimized special cases.
        For non-integer exponents, uses the general formula: q^n = exp(n * log(q))

        Args:
            exponent: The exponent (scalar or array)

        Returns:
            The quaternion raised to the given power
        """
        if jnp.iscomplexobj(exponent):
            raise NotImplementedError('Quaternion and complex exponentiation is not implemented.')

        # Handle special cases for static integer exponents only
        if isinstance(exponent, int | float | np.number):
            if exponent == -2:
                q_inv = self._inverse()
                return q_inv * q_inv
            elif exponent == -1:
                return self._inverse()
            elif exponent == 0:
                return self.ones(self.shape, self.dtype)
            elif exponent == 1:
                return self
            elif exponent == 2:
                return self * self
            return (exponent * self.log()).exp()

        # General case: q^n = exp(n * log(q))
        exponent = jnp.asarray(exponent)
        result = (exponent * self.log()).exp().wxyz
        return self.from_array(
            jnp.where(
                exponent[..., None] == 0, jnp.array([1.0, 0.0, 0.0, 0.0], dtype=self.dtype), result
            )
        )

    def log(self) -> Self:
        """Compute quaternion logarithm.

        For a quaternion q = |q| * (cos(θ) + sin(θ)v), the logarithm is:
        log(q) = log(|q|) + θ * v

        For a real quaternion, the axis v is undefined. This is harmless when q = a > 0 (θ = 0),
        but for q = -a, we must choose v because the result depends on it. By convention, the axis
        i is used: log(-a) = log(a) + π * i.

        For the zero quaternion, returns (-inf, 0, 0, 0).

        Returns:
            The logarithm of the quaternion
        """
        scalar_part = self.w
        vector_part = self.vector
        # |v| is computed after rescaling v by its largest component, so that it does not
        # underflow to zero when |v|² does.
        max_abs_component = jnp.max(jnp.abs(vector_part), axis=-1)
        is_real = max_abs_component == 0
        safe_max_abs_component = jnp.where(is_real, 1.0, max_abs_component)
        rescaled_norm_sq = jnp.sum((vector_part / safe_max_abs_component[..., None]) ** 2, axis=-1)

        # log(q) = log(|q|) + θ * v/|v|, with θ = atan2(|v|, s).
        # The where guards keep the gradients finite for real quaternions (|v| = 0).
        log_norm = 0.5 * jnp.log(scalar_part**2 + jnp.sum(vector_part**2, axis=-1))
        safe_vector_norm = safe_max_abs_component * jnp.sqrt(
            jnp.where(is_real, 1.0, rescaled_norm_sq)
        )
        safe_scalar_part = jnp.where(scalar_part == 0, 1.0, scalar_part)
        # θ/|v| tends to 1/s when |v| → 0 (s > 0)
        theta_over_vector_norm = jnp.where(
            is_real,
            1 / safe_scalar_part,
            jnp.arctan2(safe_vector_norm, scalar_part) / safe_vector_norm,
        )
        log_q_vector = theta_over_vector_norm[..., None] * vector_part

        # θ = π when v = 0 (s < 0), and the axis is i by convention
        is_negative_real = is_real & (scalar_part < 0)
        log_q_vector = jnp.where(
            is_negative_real[..., None],
            jnp.array([jnp.pi, 0, 0], dtype=log_q_vector.dtype),
            log_q_vector,
        )

        return self.from_scalar_vector(log_norm, log_q_vector)

    def exp(self) -> Self:
        """Compute quaternion exponential.

        For a quaternion q = s + v, the exponential is:
        exp(q) = exp(s) * (cos(|v|) + sin(|v|) * v/|v|)

        Returns:
            The exponential of the quaternion
        """
        scalar_part = self.w
        vector_part = self.vector
        vector_norm_sq = jnp.sum(vector_part**2, axis=-1)
        is_real = vector_norm_sq == 0

        # The where guard keeps the gradients finite for real quaternions (|v| = 0).
        vector_norm = jnp.where(is_real, 0.0, jnp.sqrt(jnp.where(is_real, 1.0, vector_norm_sq)))
        exp_scalar = jnp.exp(scalar_part)
        # sin(|v|)/|v| and cos(|v|) = 1 - |v|²/2 (sin(|v|/2)/(|v|/2))², written with sinc
        # so that the first and second derivatives are exact at |v| = 0
        sinc_vnorm = jnp.sinc(vector_norm / jnp.pi)
        sinc_half_vnorm = jnp.sinc(vector_norm / (2 * jnp.pi))
        cos_vnorm = 1 - 0.5 * vector_norm_sq * sinc_half_vnorm**2

        result_w = exp_scalar * cos_vnorm
        result_vector = jnp.expand_dims(exp_scalar * sinc_vnorm, -1) * vector_part

        return self.from_scalar_vector(result_w, result_vector)

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
    def ndim(self) -> int:
        """Number of dimensions of the quaternion tensor (without the quaternion dimension)."""
        return self.wxyz.ndim - 1

    @property
    def size(self) -> int:
        """Total number of quaternions."""
        return self.wxyz.size >> 2

    @property
    def dtype(self) -> jnp.dtype:
        """Data type."""
        return self.wxyz.dtype

    def reshape(self, *shape: ShapeLike) -> Self:
        """Reshape the tensor of quaternions.

        Args:
            shape: The new shape, as an int, a sequence of ints, or several ints.

        Returns:
            Quaternions with the new shape.
        """
        if len(shape) == 0:
            raise ValueError('Must specify at least one dimension')
        if len(shape) == 1:
            new_shape = _to_int_tuple(shape[0])
        else:
            new_shape = _to_int_tuple(shape)  # ty: ignore[invalid-argument-type]
        return self.from_array(self.wxyz.reshape(new_shape + (4,)))

    def flatten(self) -> Self:
        """Flatten the tensor of quaternions into one dimension."""
        return self.from_array(self.wxyz.reshape(-1, 4))

    def ravel(self) -> Self:
        """Flatten the tensor of quaternions into one dimension."""
        return self.flatten()

    def squeeze(self, axis: ShapeLike | None = None) -> Self:
        """Remove axes of length one.

        Args:
            axis: The axis or axes to remove. If None, all axes of length one are removed.

        Returns:
            Quaternions with the axes removed.
        """
        if axis is not None:
            # Negative axes are counted from the quaternion shape, not from the component axis
            axes = _to_int_tuple(axis)
            for ax in axes:
                if not -self.ndim <= ax < self.ndim:
                    raise ValueError(
                        f'axis {ax} is out of bounds for quaternions of dimension {self.ndim}'
                    )
            axis = tuple(ax % self.ndim for ax in axes)
        return self.from_array(jnp.squeeze(self.wxyz, axis=axis))

    def conjugate(self) -> Self:
        """Quaternion conjugate."""
        sign = jnp.array([1, -1, -1, -1], dtype=self.dtype)
        return self.from_array(self.wxyz * sign)

    def conj(self) -> Self:
        """Quaternion conjugate."""
        return self.conjugate()

    def block_until_ready(self) -> None:
        """Block until all pending computations are done."""
        self.wxyz.block_until_ready()

    @property
    def device(self) -> jax.Device:
        return self.wxyz.device

    def devices(self) -> set[jax.Device]:
        return self.wxyz.devices()

    def slerp(self, other: Self, t: ArrayLike) -> Self:
        """Spherical linear interpolation between two quaternions.

        Args:
            other: Target quaternion to interpolate towards
            t: Interpolation parameter in [0, 1]. t=0 returns self, t=1 returns other

        Returns:
            Interpolated quaternion
        """
        t = jnp.asarray(t)

        # Ensure both quaternions are normalized
        q1 = self.normalize()
        q2 = other.normalize()

        # Compute dot product
        dot = jnp.sum(q1.wxyz * q2.wxyz, axis=-1)

        # If dot product is negative, slerp won't take the shorter path.
        # Note that this is necessary to handle the double cover of SO(3)
        # by unit quaternions: q and -q represent the same rotation.
        q2_corrected = jnp.where(jnp.expand_dims(dot < 0, -1), -q2.wxyz, q2.wxyz)

        # θ is the angle between q1 and q2 on the unit sphere of R⁴, in [0, π/2]. Unlike
        # arccos(dot), the half-angle formula is accurate near θ = 0. The where guard keeps the
        # gradients finite when q1 = q2.
        diff_sq = jnp.sum((q1.wxyz - q2_corrected) ** 2, axis=-1)
        is_equal = diff_sq == 0
        diff_norm = jnp.where(is_equal, 0.0, jnp.sqrt(jnp.where(is_equal, 1.0, diff_sq)))
        sum_norm = jnp.linalg.norm(q1.wxyz + q2_corrected, axis=-1)
        theta = 2 * jnp.arctan2(diff_norm, sum_norm)

        # sin(tθ)/sin(θ) = t sinc(tθ)/sinc(θ), written with sinc so that the weights and their
        # derivatives are exact at θ = 0, without a linear interpolation fallback.
        # sinc(θ) >= 2/π since θ <= π/2.
        sinc_theta = jnp.sinc(theta / jnp.pi)
        weight1 = (1 - t) * jnp.sinc((1 - t) * theta / jnp.pi) / sinc_theta
        weight2 = t * jnp.sinc(t * theta / jnp.pi) / sinc_theta

        result = weight1[..., None] * q1.wxyz + weight2[..., None] * q2_corrected
        return self.from_array(result)
