import numpy as np
import matplotlib.pyplot as plt

g = 9.81

def functions(lamb):
    x, y = lamb, lamb**2 / 2
    dx, dy = 1, lamb
    ddx, ddy = 0, 1
    return x, y, dx, dy, ddx, ddy

def length(lamb):
    _, _, dx, dy, _, _ = functions(lamb)
    return np.sqrt(dx**2 + dy**2)

def rollercoaster_u(lamb, u):
    _, _, dx, dy, ddx, ddy = functions(lamb)
    du = (-g * dy - (dx * ddx + dy * ddy) * u**2) / (dx**2 + dy**2)
    return u, du

def rollercoaster_v(lamb, v):
    _, _, _, dy, _, _ = functions(lamb)
    s = length(lamb)
    return v / s, -g * dy / s

rollercoaster = {"u": rollercoaster_u, "v": rollercoaster_v}

def gevoels_snelheid(lamb, u):
    _, _, dx, dy, ddx, ddy = functions(lamb)
    s = length(lamb)
    return (g * dx + (ddy * dx - ddx * dy) * u**2) / s / g

def euler_methode(type_r, lamb0, roller0, stepsize, total_time):
    du = rollercoaster[type_r]
    n = int(round(total_time / stepsize))
    t = np.linspace(0, n * stepsize, n + 1)
    l = np.empty(n + 1)
    w = np.empty(n + 1)
    l[0], w[0] = lamb0, roller0
    for k in range(n):
        dlamb, droller = du(l[k], w[k])
        l[k + 1] = l[k] + stepsize * dlamb
        w[k + 1] = w[k] + stepsize * droller
    return t, l, w
    
def u_en_v(versie, lamb, roller):
    """Reken uit de oplossing zowel u = dl/dt als v = sigma*dl/dt uit."""
    s = length(lamb)
    return (roller, s * roller) if versie == "u" else (roller / s, roller)

def tangentiele_versnelling(lamb):
    _, _, dx, dy, _, _ = functions(lamb)
    return -g * dy / length(lamb)

l0, w0, T = -3.0, 0.0, 6.0
stappen = (0.01, 0.001)
h_fijn = stappen[-1]
fig, ax = plt.subplots(1, 2, figsize=(18, 4.5))
fig, axi = plt.subplots(1, 2, figsize=(18, 4.5))
for versie in ("u", "v"):
    for h in stappen:
        t, l, w = euler_methode(versie, l0, w0, h, T)
        u, v = u_en_v(versie, l, w)
        felt = gevoels_snelheid(l, u)

        if h == h_fijn and versie == "u":
            a = tangentiele_versnelling(l)
            x, y, *_ = functions(l)
            xb, yb, *_ = functions(np.linspace(l.min(), l.max(), 400))
            stride = max(1, int(round(0.2 / h)))
            ax[0].plot(xb, yb, "k-", lw=1, label="baan")
            ax[0].plot(x[::stride], y[::stride], "ro", ms=4, label="positie per 0.2s")
            ax[1].plot(t, felt, color="tab:red")
            axi[0].plot(t, a)
            axi[1].plot(t, v, color="tab:blue")

ax[0].set(xlabel="x", ylabel="y", title="Positie van het karretje op de achtbaan")
ax[0].legend()
ax[1].axhline(1, color="gray", lw=0.8, ls=":")
ax[1].set(xlabel="t (s)", ylabel="gevoelde versnelling (g)",
          title="Passagiers gevoels-snelheid")
axi[0].axhline(0, color="k", lw=0.6)
axi[0].set(xlabel="t (s)", ylabel="achtbaan's versnelling (m/s²)",
          title="Versnelling op de baan per seconde")
axi[1].axhline(0, color="k", lw=0.6)
axi[1].set(xlabel="t (s)", ylabel="snelheid v (m/s)",
          title="Snelheid langs de baan")
plt.tight_layout()
plt.show()

