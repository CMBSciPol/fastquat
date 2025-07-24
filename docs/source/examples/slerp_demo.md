# SLERP Visualization on the Sphere

This example demonstrates Spherical Linear Interpolation (SLERP) between quaternions
and visualizes the interpolation path on the unit sphere with an animated GIF. SLERP provides smooth,
constant-angular-velocity interpolation between rotations.

For the complete interactive implementation, see the [SLERP Animation Notebook](slerp_animation.ipynb).

## Code Example

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from fastquat import Quaternion

def visualize_slerp_on_sphere():
    """Visualize SLERP interpolation between two quaternions."""

    # Define two quaternions representing different rotations
    # q1: Identity (no rotation)
    q1 = Quaternion.ones()

    # q2: 90 degree rotation around axis (1, 1, 1)
    axis = jnp.array([1.0, 1.0, 1.0])
    axis = axis / jnp.linalg.norm(axis)  # Normalize
    angle = jnp.pi / 2  # 90 degrees

    q2 = Quaternion(
        jnp.cos(angle/2),
        axis[0] * jnp.sin(angle/2),
        axis[1] * jnp.sin(angle/2),
        axis[2] * jnp.sin(angle/2)
    )

    print(f"Start quaternion q1: {q1}")
    print(f"End quaternion q2: {q2}")

    # Generate interpolation parameters
    t_values = jnp.linspace(0, 1, 20)

    # SLERP interpolation
    slerp_quaternions = q1.slerp(q2, t_values)

    # Linear interpolation for comparison
    linear_quaternions = []
    for t in t_values:
        q_linear = (1 - t) * q1 + t * q2
        q_linear_normalized = q_linear.normalize()
        linear_quaternions.append(q_linear_normalized)

    # Convert to array for easier handling
    linear_quaternions = Quaternion.from_array(
        jnp.array([q.wxyz for q in linear_quaternions])
    )

    # Test vector to rotate (unit vector along x-axis)
    test_vector = jnp.array([1.0, 0.0, 0.0])

    # Apply rotations to test vector
    slerp_rotated = slerp_quaternions.rotate_vector(test_vector)
    linear_rotated = linear_quaternions.rotate_vector(test_vector)

    # Create visualization
    fig = plt.figure(figsize=(15, 5))

    # Plot 1: Quaternion components over time
    ax1 = fig.add_subplot(131)
    ax1.plot(t_values, slerp_quaternions.w, 'r-', label='SLERP w', linewidth=2)
    ax1.plot(t_values, slerp_quaternions.x, 'g-', label='SLERP x', linewidth=2)
    ax1.plot(t_values, slerp_quaternions.y, 'b-', label='SLERP y', linewidth=2)
    ax1.plot(t_values, slerp_quaternions.z, 'm-', label='SLERP z', linewidth=2)

    ax1.plot(t_values, linear_quaternions.w, 'r--', alpha=0.7, label='Linear w')
    ax1.plot(t_values, linear_quaternions.x, 'g--', alpha=0.7, label='Linear x')
    ax1.plot(t_values, linear_quaternions.y, 'b--', alpha=0.7, label='Linear y')
    ax1.plot(t_values, linear_quaternions.z, 'm--', alpha=0.7, label='Linear z')

    ax1.set_xlabel('Interpolation Parameter t')
    ax1.set_ylabel('Quaternion Components')
    ax1.set_title('SLERP vs Linear Interpolation\nQuaternion Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rotated vectors on unit sphere
    ax2 = fig.add_subplot(132, projection='3d')

    # Draw unit sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.1, color='lightgray')

    # Plot SLERP path
    ax2.plot(slerp_rotated[:, 0], slerp_rotated[:, 1], slerp_rotated[:, 2],
             'ro-', linewidth=3, markersize=4, label='SLERP Path')

    # Plot linear interpolation path
    ax2.plot(linear_rotated[:, 0], linear_rotated[:, 1], linear_rotated[:, 2],
             'b--', alpha=0.7, linewidth=2, label='Linear Path')

    # Mark start and end points
    ax2.scatter(*slerp_rotated[0], color='green', s=100, label='Start')
    ax2.scatter(*slerp_rotated[-1], color='red', s=100, label='End')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Rotation Paths on Unit Sphere')
    ax2.legend()
    ax2.set_box_aspect([1,1,1])

    # Plot 3: Angular velocity analysis
    ax3 = fig.add_subplot(133)

    # Calculate angular distances between consecutive quaternions
    slerp_distances = []
    linear_distances = []

    for i in range(1, len(t_values)):
        # SLERP angular distance
        dot_slerp = jnp.sum(slerp_quaternions.wxyz[i-1] * slerp_quaternions.wxyz[i])
        angle_slerp = 2 * jnp.arccos(jnp.abs(jnp.clip(dot_slerp, -1, 1)))
        slerp_distances.append(angle_slerp)

        # Linear angular distance
        dot_linear = jnp.sum(linear_quaternions.wxyz[i-1] * linear_quaternions.wxyz[i])
        angle_linear = 2 * jnp.arccos(jnp.abs(jnp.clip(dot_linear, -1, 1)))
        linear_distances.append(angle_linear)

    t_diff = t_values[1:]
    ax3.plot(t_diff, slerp_distances, 'ro-', label='SLERP Angular Step', linewidth=2)
    ax3.plot(t_diff, linear_distances, 'b--', alpha=0.7, label='Linear Angular Step', linewidth=2)

    ax3.set_xlabel('Interpolation Parameter t')
    ax3.set_ylabel('Angular Distance (radians)')
    ax3.set_title('Angular Velocity Comparison\n(Constant = Better)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return slerp_quaternions, linear_quaternions, slerp_rotated, linear_rotated

def demonstrate_slerp_properties():
    """Demonstrate key properties of SLERP interpolation."""

    print("=== SLERP Demonstration ===")

    # Create quaternions
    q_start = Quaternion.ones()
    q_end = Quaternion(0.5, 0.5, 0.5, 0.5).normalize()

    print(f"Start quaternion: {q_start}")
    print(f"End quaternion: {q_end}")
    print(f"Dot product: {jnp.sum(q_start.wxyz * q_end.wxyz):.4f}")

    # Test different interpolation values
    test_t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\nSLERP Interpolation Results:")
    print("t\tQuaternion\t\t\t\tNorm")
    print("-" * 60)

    for t in test_t_values:
        q_interp = q_start.slerp(q_end, t)
        norm = q_interp.norm()
        print(f"{t:.2f}\t{q_interp}\t{norm:.6f}")

    # Test batch interpolation
    print("\nBatch SLERP Test:")
    t_batch = jnp.array([0.0, 0.3, 0.7, 1.0])
    q_batch = q_start.slerp(q_end, t_batch)
    print(f"Input t values: {t_batch}")
    print(f"Output quaternions shape: {q_batch.shape}")
    print(f"All norms ≈ 1: {jnp.allclose(q_batch.norm(), 1.0)}")

    # Demonstrate shortest path property
    print("\nShortest Path Property:")
    q_neg = Quaternion(-q_end.w, -q_end.x, -q_end.y, -q_end.z)  # Same rotation, opposite quaternion

    q_slerp_pos = q_start.slerp(q_end, 0.5)
    q_slerp_neg = q_start.slerp(q_neg, 0.5)

    print(f"SLERP to +q: {q_slerp_pos}")
    print(f"SLERP to -q: {q_slerp_neg}")
    print("(Both should represent similar intermediate rotations)")

def animation_example():
    """Show how SLERP can be used for smooth animation."""

    print("\n=== Animation with SLERP ===")

    # Create a more complex rotation sequence
    keyframe_quaternions = [
        Quaternion.ones(),  # Identity
        Quaternion(0.7071, 0.7071, 0.0, 0.0),  # 90° around X
        Quaternion(0.7071, 0.0, 0.7071, 0.0),  # 90° around Y
        Quaternion(0.7071, 0.0, 0.0, 0.7071),  # 90° around Z
        Quaternion.ones(),  # Back to identity
    ]

    # Generate smooth animation between keyframes
    frames_per_segment = 10
    animation_frames = []

    for i in range(len(keyframe_quaternions) - 1):
        q_start = keyframe_quaternions[i]
        q_end = keyframe_quaternions[i + 1]

        t_values = jnp.linspace(0, 1, frames_per_segment + 1)[:-1]  # Exclude last to avoid duplicate
        segment_frames = q_start.slerp(q_end, t_values)
        animation_frames.extend([Quaternion.from_array(q.wxyz) for q in segment_frames])

    # Add final frame
    animation_frames.append(keyframe_quaternions[-1])

    print(f"Generated {len(animation_frames)} animation frames")
    print(f"All frames normalized: {all(abs(q.norm() - 1.0) < 1e-6 for q in animation_frames)}")

    # Apply to a test object (cube vertices)
    cube_vertices = jnp.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Bottom face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Top face
    ])

    # Show first few frames
    print("\nFirst 5 animation frames:")
    for i, q in enumerate(animation_frames[:5]):
        rotated_vertices = q.rotate_vector(cube_vertices)
        center = jnp.mean(rotated_vertices, axis=0)
        print(f"Frame {i}: center at {center}")

if __name__ == "__main__":
    # Run the main visualization
    slerp_q, linear_q, slerp_rot, linear_rot = visualize_slerp_on_sphere()

    # Demonstrate properties
    demonstrate_slerp_properties()

    # Show animation example
    animation_example()
```

## Key Concepts Demonstrated

**Spherical Linear Interpolation (SLERP)**
   SLERP provides smooth interpolation between quaternions along the shortest
   arc on the 4D unit sphere. Unlike linear interpolation, SLERP maintains
   constant angular velocity.

**Shortest Path Selection**
   The implementation automatically chooses the shorter path between quaternions
   by flipping the sign of one quaternion if their dot product is negative.

**Constant Angular Velocity**
   The angular velocity plot shows that SLERP maintains nearly constant angular
   steps, while linear interpolation varies significantly.

**Normalization Preservation**
   All interpolated quaternions remain perfectly normalized, which is essential
   for representing pure rotations.

## Mathematical Background

SLERP between quaternions **q₁** and **q₂** with parameter t ∈ [0,1] is defined as:

$$\text{slerp}(q_1, q_2, t) = \frac{\sin((1-t)\theta)}{\sin\theta} q_1 + \frac{\sin(t\theta)}{\sin\theta} q_2$$

where θ is the angle between the quaternions:

$$\cos\theta = q_1 \cdot q_2 = w_1w_2 + x_1x_2 + y_1y_2 + z_1z_2$$

For quaternions that are very close (θ ≈ 0), the implementation automatically
switches to normalized linear interpolation to avoid numerical instability.

## Applications

**Computer Animation**
   SLERP is the standard method for interpolating between keyframe rotations
   in 3D animation software.

**Robotics**
   Smooth robot motion planning requires constant angular velocity rotations,
   which SLERP provides naturally.

**Camera Controls**
   Smooth camera movements in 3D applications use SLERP to interpolate
   between different viewing orientations.

**Scientific Visualization**
   When animating the rotation of molecules, astronomical objects, or other
   3D scientific data, SLERP ensures smooth, physically plausible motion.

## Expected Output

Running this example will:

1. Display three plots comparing SLERP and linear interpolation
2. Show quaternion component evolution over the interpolation parameter
3. Visualize the rotation paths on the unit sphere
4. Analyze angular velocity to demonstrate SLERP's constant-speed property
5. Print detailed information about quaternion properties and normalization
6. Demonstrate batch interpolation and animation applications

The visualization clearly shows why SLERP is preferred for rotation interpolation:
it provides smoother, more natural motion with constant angular velocity.
