# Examples

This section contains practical examples demonstrating FastQuat's capabilities.


## Quaternion Rotation and Spherical Cap Visualization

This [example](spherical_cap.ipynb) shows how to:

* Use `jax.scipy.spatial.transform.Rotation.from_euler` to create rotations
* Convert rotation matrices to quaternions using FastQuat
* Visualize quaternion rotations with spherical caps
* Demonstrate quaternion multiplication for creating compound rotations

## SLERP Visualization on the Unit Sphere

This [example](slerp_animation.ipynb) demonstrates:

* Spherical Linear Interpolation (SLERP) between quaternions
* **Animated visualization** of rotation paths on the unit sphere
* Analysis of SLERP's constant angular velocity property
* Generation of smooth rotation animations using SLERP

These examples showcase FastQuat's integration with the scientific Python ecosystem and demonstrate practical applications in 3D graphics and robotics.
