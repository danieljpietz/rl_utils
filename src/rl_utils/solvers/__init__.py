def rk4(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)

    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
