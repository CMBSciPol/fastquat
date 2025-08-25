# Quaternion Class

```{eval-rst}
.. automodule:: fastquat.quaternion
   :members:
   :undoc-members:
   :show-inheritance:
```

The {class}`Quaternion` class provides a comprehensive interface for quaternion operations
optimized for JAX. All methods are compatible with JAX transformations including JIT
compilation, automatic differentiation, and vectorization.

## Constructor Methods

```{eval-rst}
.. automethod:: fastquat.Quaternion.__init__
.. automethod:: fastquat.Quaternion.from_array
.. automethod:: fastquat.Quaternion.from_scalar_vector
.. automethod:: fastquat.Quaternion.from_rotation_matrix
.. automethod:: fastquat.Quaternion.zeros
.. automethod:: fastquat.Quaternion.ones
.. automethod:: fastquat.Quaternion.full
.. automethod:: fastquat.Quaternion.random
```

## Properties

```{eval-rst}
.. autoattribute:: fastquat.Quaternion.w
.. autoattribute:: fastquat.Quaternion.x
.. autoattribute:: fastquat.Quaternion.y
.. autoattribute:: fastquat.Quaternion.z
.. autoattribute:: fastquat.Quaternion.vector
.. autoattribute:: fastquat.Quaternion.shape
.. autoattribute:: fastquat.Quaternion.dtype
```

## Core Operations

```{eval-rst}
.. automethod:: fastquat.Quaternion.norm
.. automethod:: fastquat.Quaternion.normalize
.. automethod:: fastquat.Quaternion.conjugate
.. automethod:: fastquat.Quaternion.conj
.. automethod:: fastquat.Quaternion.to_components
```

## Rotation Operations

```{eval-rst}
.. automethod:: fastquat.Quaternion.to_rotation_matrix
.. automethod:: fastquat.Quaternion.rotate_vector
```

## Interpolation

```{eval-rst}
.. automethod:: fastquat.Quaternion.slerp
```

## Advanced Operations

```{eval-rst}
.. automethod:: fastquat.Quaternion.log
.. automethod:: fastquat.Quaternion.exp
.. automethod:: fastquat.Quaternion.__pow__
```

## Array Operations

```{eval-rst}
.. automethod:: fastquat.Quaternion.reshape
.. automethod:: fastquat.Quaternion.flatten
.. automethod:: fastquat.Quaternion.ravel
.. automethod:: fastquat.Quaternion.squeeze
.. automethod:: fastquat.Quaternion.block_until_ready
```

## Device and Memory

```{eval-rst}
.. autoattribute:: fastquat.Quaternion.device
.. automethod:: fastquat.Quaternion.devices
.. autoattribute:: fastquat.Quaternion.nbytes
.. autoattribute:: fastquat.Quaternion.itemsize
.. autoattribute:: fastquat.Quaternion.size
.. autoattribute:: fastquat.Quaternion.ndim
```
