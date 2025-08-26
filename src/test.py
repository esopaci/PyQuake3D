import numpy as np
import matplotlib.pyplot as plt

# Parameters
# c_h = 1.0  # Diffusion coefficient
# h = 1.0    # Domain length
# N = 100    # Number of grid points
# dy = h / N  # Spatial step
# dt = 0.00001  # Time step for stability
# phi = 0.5  # Example value
# h_0 = -10.0  # Example value
# beta = 1.0 # Example value
# c_hyd = 1.0 # Example value
# p_k = 1.0  # Initial value p(y, 0) = p_k
# r = c_h * dt / (dy ** 2)  # Stability factor (should be <= 0.5)

# # Initialize grid
# y = np.linspace(0, h, N)
# p = np.full(N, p_k)  # Initial condition p(y, 0) = p_k
# p_new = p.copy()

# # Boundary condition at y = 0
# dpdye_0 = (1 - phi) * h_0 / (4 * beta ** 2 * c_hyd ** 2)
# if beta == 0 or c_hyd == 0:
#     raise ValueError("beta or c_hyd cannot be zero")

# # Time stepping
# nt = 1000  # Number of time steps
# for n in range(nt):
#     # Boundary at y = 0 (using ghost point correction)
#     p_minus1 = p[1] - 2 * dy * dpdye_0
#     p_new[0] = p[0] + r * (p[1] - 2 * p[0] + p_minus1)
#     # Vectorized update for interior points (j = 1 to N-2)
#     p_new[1:N-1] = p[1:N-1] + r * (p[2:N] - 2 * p[1:N-1] + p[0:N-2])
#     # Boundary at y = h (Neumann: dp/dy = 0)
#     p_new[N-1] = p_new[N-2]
#     # Update
#     p[:] = p_new[:]
#     if(n%200==0):
#         plt.plot(y, p)

# # Extract p at y = 0
# p_at_y0 = p[0]

# print(f"p at y = 0 after {nt} steps: {p_at_y0}")
# # Optional: Plot

# plt.xlabel('y')
# plt.ylabel('p')
# plt.title('Pressure Distribution')
# plt.show()





# Parameters
c_h = 1.0e-6  # Diffusion coefficient
h = 3e-3    # Domain length
N = 10    # Number of grid points
dy = h / N  # Spatial step
dt = 0.01  # Time step for stability
dphi = -0.5  # Example value

# dt = 1e1  # Time step for stability
# dphi = -0.5e-10  # Example value

#h_0 = 10  # Example value
beta = 5e-11 # Example value
c_hyd = 1e-6 # Example value
p_k = 1.0e6  # Initial value p(y, 0) = p_k
r = c_h * dt / (dy ** 2)  # Stability factor (should be <= 0.5)
print('Stability factor (should be <= 0.5):',r)
# Initialize grid
y = np.linspace(0, h, N)
p = np.full(N, p_k)  # Initial condition p(y, 0) = p_k
p_new = p.copy()

# Boundary condition at y = 0
dpdye_0 = dphi * h / (2 * beta * c_hyd)
if beta == 0 or c_hyd == 0:
    raise ValueError("beta or c_hyd cannot be zero")

plt.plot(y, p)
# Time stepping
nt = 1000  # Number of time steps
for n in range(nt):
    # Boundary at y = 0 (using ghost point correction)
    p_minus1 = p[1] - 2 * dy * dpdye_0
    p_new[0] = p[0] + r * (p[1] - 2 * p[0] + p_minus1)
    # Vectorized update for interior points (j = 1 to N-2)
    p_new[1:N-1] = p[1:N-1] + r * (p[2:N] - 2 * p[1:N-1] + p[0:N-2])
    # Boundary at y = h (Neumann: dp/dy = 0)
    p_new[N-1] = p_new[N-2]
    # Update
    p[:] = p_new[:]
    print('dP:',r * (p[1] - 2 * p[0] + p_minus1)/dt)
    if(n%100==0):
        plt.plot(y, p)

# Extract p at y = 0
p_at_y0 = p[0]

print(f"p at y = 0 after {nt} steps: {p_at_y0}")
# Optional: Plot

plt.xlabel('y')
plt.ylabel('p')
#plt.yscale('log')
plt.title('Pressure Distribution')
plt.show()



c_h = 1.0e-6  # Diffusion coefficient
h = 3e-3      # Domain length
N = 10        # Number of grid points
dy = h / N    # Spatial step
dt = 0.01     # Time step (larger possible with implicit method)
dphi = -0.5   # Example value
beta = 5e-11  # Example value
c_hyd = 1e-6  # Example value
p_k = 1.0e6   # Initial value p(y, 0) = p_k
r = c_h * dt / (2 * dy ** 2)  # Stability factor for Crank-Nicolson

print('Stability factor (r):', r)

# Initialize grid
y = np.linspace(0, h, N)
p = np.full(N, p_k)  # Initial condition p(y, 0) = p_k
p_new = p.copy()

# Boundary condition at y = 0
dpdye_0 = dphi * h / (2 * beta * c_hyd)
if beta == 0 or c_hyd == 0:
    raise ValueError("beta or c_hyd cannot be zero")

# Time stepping
nt = 1000  # Number of time steps
for n in range(nt):
    # Construct tridiagonal system
    a = np.full(N, -r)  # Lower diagonal
    b = np.full(N, 1 + 2 * r)  # Main diagonal
    c = np.full(N, -r)  # Upper diagonal
    d = r * np.roll(p, 1) + (1 - 2 * r) * p + r * np.roll(p, -1)  # Right-hand side
    
    # Boundary at y = 0 (Neumann)
    a[0] = 0
    b[0] = 1 + r
    c[0] = -r
    d[0] += r * (p[1] - 2 * dy * dpdye_0)
    
    # Boundary at y = h (Neumann: dp/dy = 0)
    a[N-1] = -r
    b[N-1] = 1 + r
    c[N-1] = 0
    d[N-1] = d[N-2]  # Enforce dp/dy = 0
    
    # Solve tridiagonal system (Thomas algorithm)
    for i in range(1, N):
        w = a[i] / b[i-1]
        b[i] -= w * c[i-1]
        d[i] -= w * d[i-1]
    p_new[N-1] = d[N-1] / b[N-1]
    for i in range(N-2, -1, -1):
        p_new[i] = (d[i] - c[i] * p_new[i+1]) / b[i]
    
    # Update and monitor
    p[:] = p_new[:]
    dp_dt_approx = (p_new[1] - p[1]) / dt  # Approximate dp/dt
    print(f"t = {n * dt:.4f}, dp/dt at y=dy: {dp_dt_approx:.6e}")
    if n % 100 == 0:
        plt.plot(y, p, label=f't = {n * dt:.4f}s' if n == 0 else "")

# Extract p at y = 0
p_at_y0 = p[0]

print(f"p at y = 0 after {nt} steps: {p_at_y0}")

# Plot
plt.xlabel('y (m)')
plt.ylabel('p (Pa)')
plt.title('Pressure Distribution')
plt.legend()
plt.show()