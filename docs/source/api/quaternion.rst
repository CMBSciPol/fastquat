Quaternion Class
================

.. automodule:: fastquat.quaternion
   :members:
   :undoc-members:
   :show-inheritance:

The :class:`Quaternion` class provides a comprehensive interface for quaternion operations
optimized for JAX. All methods are compatible with JAX transformations including JIT
compilation, automatic differentiation, and vectorization.

Constructor Methods
-------------------

.. automethod:: fastquat.Quaternion.__init__
.. automethod:: fastquat.Quaternion.from_array
.. automethod:: fastquat.Quaternion.from_scalar_vector
.. automethod:: fastquat.Quaternion.from_rotation_matrix
.. automethod:: fastquat.Quaternion.zeros
.. automethod:: fastquat.Quaternion.ones
.. automethod:: fastquat.Quaternion.full
.. automethod:: fastquat.Quaternion.random

Properties
----------

.. autoattribute:: fastquat.Quaternion.w
.. autoattribute:: fastquat.Quaternion.x
.. autoattribute:: fastquat.Quaternion.y
.. autoattribute:: fastquat.Quaternion.z
.. autoattribute:: fastquat.Quaternion.vector
.. autoattribute:: fastquat.Quaternion.shape
.. autoattribute:: fastquat.Quaternion.dtype

Core Operations
---------------

.. automethod:: fastquat.Quaternion.norm
.. automethod:: fastquat.Quaternion.normalize
.. automethod:: fastquat.Quaternion.inverse
.. automethod:: fastquat.Quaternion.conjugate
.. automethod:: fastquat.Quaternion.conj

Rotation Operations
-------------------

.. automethod:: fastquat.Quaternion.to_rotation_matrix
.. automethod:: fastquat.Quaternion.rotate_vector

Interpolation
-------------

.. automethod:: fastquat.Quaternion.slerp

Advanced Operations
-------------------

.. automethod:: fastquat.Quaternion.log
.. automethod:: fastquat.Quaternion.exp
.. automethod:: fastquat.Quaternion.__pow__
