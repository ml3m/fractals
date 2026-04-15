"""
Double Pendulum - Predictability Time Horizon (PTH) Parameter Space
====================================================================
Supports:
  - CPU multiprocessing (fallback)
  - ROCm/CUDA GPU acceleration via PyTorch (recommended)

Usage:
  python double_pendulum.py                         # default PTH map
  python double_pendulum.py --mode phase           # phase-color fractal
  HSA_OVERRIDE_GFX_VERSION=10.3.0 python double_pendulum.py --mode phase
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import sys

# ── Physical parameters ──────────────────────────────────────────────────────
G  = 9.82
L1 = 1.0
L2 = 1.0
M1 = 1.0
M2 = 1.0

# ── Simulation parameters (PTH mode) ──────────────────────────────────────────
T_MAX          = 60.0   # total sim time [s]
N_STEPS        = 2500    # time steps  →  dt = T_MAX / N_STEPS
EPSILON        = 1e-6   # perturbation magnitude
DIVERGE_THRESH = 0.1    # divergence threshold [rad]

# ── Simulation parameters (Phase mode) ────────────────────────────────────────
T_PHASE  = 10.0          # simulation time [s] — reference uses 10 s
DT_PHASE = 0.01          # RK4 time step [s]   — reference default

# ── Grid resolution ───────────────────────────────────────────────────────────
N_POINTS = 2000          # per axis  (2000 → 4 M sims)


def _parse_mode(argv):
    mode = "pth"
    for arg in argv[1:]:
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1].strip().lower()
        elif arg == "--phase":
            mode = "phase"
        elif arg == "--pth":
            mode = "pth"
    if mode not in {"pth", "phase"}:
        raise ValueError(f"Unsupported mode '{mode}'. Use --mode=pth or --mode=phase.")
    return mode


# ─────────────────────────────────────────────────────────────────────────────
# GPU BATCH SOLVER  (PyTorch — ROCm or CUDA)
# ─────────────────────────────────────────────────────────────────────────────

def _dp_deriv_torch(y, torch):
    a1, a2, da1, da2 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    d   = a1 - a2
    sd  = torch.sin(d);  cd = torch.cos(d)
    den1 = (M1 + M2) * L1 - M2 * L1 * cd**2
    den2 = (L2 / L1) * den1
    dda1 = (-M2*L1*da1**2*sd*cd + M2*G*torch.sin(a2)*cd
            - M2*L2*da2**2*sd - (M1+M2)*G*torch.sin(a1)) / den1
    dda2 = (M2*L2*da2**2*sd*cd
            + (M1+M2)*(G*torch.sin(a1)*cd + L1*da1**2*sd
                       - G*torch.sin(a2))) / den2
    return torch.stack([da1, da2, dda1, dda2], dim=1)


def rk4_step(y, dt, torch):
    k1 = _dp_deriv_torch(y,           torch)
    k2 = _dp_deriv_torch(y + dt/2*k1, torch)
    k3 = _dp_deriv_torch(y + dt/2*k2, torch)
    k4 = _dp_deriv_torch(y + dt   *k3, torch)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def _angular_delta_torch(x, y, torch):
    # Wrap angular difference into [-pi, pi] to avoid false jumps at boundaries.
    return torch.atan2(torch.sin(x - y), torch.cos(x - y))


def _angular_delta_np(x, y):
    return np.arctan2(np.sin(x - y), np.cos(x - y))


def _hsv_to_rgb_torch(h, s, v, torch):
    """HSV→RGB on GPU.  h, s, v are 1-D tensors in [0,1]. Returns (B,3)."""
    h6 = (h * 6.0) % 6.0
    i  = h6.long() % 6
    f  = h6 - i.float()
    p  = v * (1.0 - s)
    q  = v * (1.0 - s * f)
    t  = v * (1.0 - s * (1.0 - f))
    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p,
        torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v,
        torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t,
        torch.where(i == 3, v, torch.where(i == 4, v, q)))))
    return torch.stack([r, g, b], dim=1).clamp(0, 1)


def compute_pth_gpu(alpha_flat, beta_flat):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  PyTorch device : {device}")
    if device.type == "cuda":
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")

    B  = len(alpha_flat)
    dt = T_MAX / N_STEPS

    a0 = torch.tensor(np.radians(alpha_flat), dtype=torch.float64, device=device)
    b0 = torch.tensor(np.radians(beta_flat),  dtype=torch.float64, device=device)
    z  = torch.zeros(B, dtype=torch.float64, device=device)

    y  = torch.stack([a0,           b0, z, z], dim=1)
    yp = torch.stack([a0 + EPSILON, b0, z, z], dim=1)

    pth      = torch.full((B,), T_MAX, dtype=torch.float64, device=device)
    diverged = torch.zeros(B, dtype=torch.bool, device=device)

    LOG_EVERY = max(1, N_STEPS // 20)
    t0_wall   = time.time()

    print(f"\n  Integrating {B:,} trajectory pairs  ×  {N_STEPS} RK4 steps …\n")
    print(f"  {'Progress':<32} {'Step':>6}  {'Sim-t':>6}  "
          f"{'Diverged':>9}  {'Elapsed':>8}  {'ETA':>8}")
    print("  " + "─" * 75)

    for step in range(N_STEPS):
        t_sim = (step + 1) * dt
        y  = rk4_step(y,  dt, torch)
        yp = rk4_step(yp, dt, torch)

        d1      = _angular_delta_torch(y[:, 0], yp[:, 0], torch)
        d2      = _angular_delta_torch(y[:, 1], yp[:, 1], torch)
        dist    = torch.sqrt(d1**2 + d2**2)
        newly   = (~diverged) & (dist > DIVERGE_THRESH)
        pth[newly]      = t_sim
        diverged[newly] = True

        if (step + 1) % LOG_EVERY == 0 or step == N_STEPS - 1:
            n_div   = diverged.sum().item()
            elapsed = time.time() - t0_wall
            pct     = (step + 1) / N_STEPS
            eta     = (elapsed / pct - elapsed) if pct > 0 else 0
            bar     = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))
            print(f"  [{bar}]  {step+1:>4}/{N_STEPS}  "
                  f"t={t_sim:5.1f}s  "
                  f"{100*n_div/B:6.1f}%  "
                  f"{elapsed:7.1f}s  "
                  f"~{eta:6.1f}s")
            sys.stdout.flush()

        if diverged.all():
            print("\n  All trajectories diverged — stopping early.")
            break

    print()
    return pth.cpu().numpy()


def compute_phase_rgb_gpu(alpha_flat, beta_flat):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  PyTorch device : {device}")
    if device.type == "cuda":
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")

    B       = len(alpha_flat)
    n_steps = int(T_PHASE / DT_PHASE)
    dt      = DT_PHASE
    TWO_PI  = 2.0 * np.pi
    PI      = np.pi

    a0 = torch.tensor(np.radians(alpha_flat), dtype=torch.float32, device=device)
    b0 = torch.tensor(np.radians(beta_flat),  dtype=torch.float32, device=device)
    z  = torch.zeros(B, dtype=torch.float32, device=device)
    y  = torch.stack([a0, b0, z, z], dim=1)

    y[:, 0] = y[:, 0] % TWO_PI
    y[:, 1] = y[:, 1] % TWO_PI

    flipped = torch.zeros(B, dtype=torch.bool, device=device)

    LOG_EVERY = max(1, n_steps // 20)
    t0_wall   = time.time()

    print(f"\n  Integrating {B:,} trajectories  ×  {n_steps} RK4 steps …\n")
    print(f"  {'Progress':<32} {'Step':>6}  {'Sim-t':>6}  "
          f"{'Flipped':>9}  {'Elapsed':>8}  {'ETA':>8}")
    print("  " + "─" * 75)

    for step in range(n_steps):
        t_sim = (step + 1) * dt

        prev_a1 = y[:, 0].clone()
        prev_a2 = y[:, 1].clone()

        y = rk4_step(y, dt, torch)

        y[:, 0] = y[:, 0] % TWO_PI
        y[:, 1] = y[:, 1] % TWO_PI

        curr_a1 = y[:, 0]
        curr_a2 = y[:, 1]

        cross_a1 = (((curr_a1 < PI) & (prev_a1 > PI)) |
                    ((curr_a1 > PI) & (prev_a1 < PI))) & \
                   (curr_a1 < 5) & (prev_a1 < 5)
        cross_a2 = (((curr_a2 < PI) & (prev_a2 > PI)) |
                    ((curr_a2 > PI) & (prev_a2 < PI))) & \
                   (curr_a2 < 5) & (prev_a2 < 5)
        flipped |= cross_a1 | cross_a2

        if (step + 1) % LOG_EVERY == 0 or step == n_steps - 1:
            n_flip  = flipped.sum().item()
            elapsed = time.time() - t0_wall
            pct     = (step + 1) / n_steps
            eta     = (elapsed / pct - elapsed) if pct > 0 else 0
            bar     = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))
            print(f"  [{bar}]  {step+1:>5}/{n_steps}  "
                  f"t={t_sim:5.1f}s  "
                  f"{100*n_flip/B:6.1f}%  "
                  f"{elapsed:7.1f}s  "
                  f"~{eta:6.1f}s")
            sys.stdout.flush()

    print()

    a1 = y[:, 0]
    a2 = y[:, 1]

    x2 = torch.sin(a1) + torch.sin(a2)
    y2 = torch.cos(a1) + torch.cos(a2)

    hue = ((torch.atan2(x2, y2) + PI) / TWO_PI + 0.65) % 1.0
    sat = torch.ones_like(hue)

    pe_max  = (M1 + M2) * G * L1 * 2 + M2 * G * L2 * 2
    pe_rel  = ((M1 + M2) * G * L1 * (1 - torch.cos(a0))
               + M2 * G * L2 * (1 - torch.cos(b0)))
    e_bright = ((1 - pe_rel / pe_max).clamp(0, 1)) ** 0.35

    val = (~flipped).float() * e_bright

    rgb = _hsv_to_rgb_torch(hue, sat, val, torch)
    return rgb.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# CPU FALLBACK  (multiprocessing + scipy)
# ─────────────────────────────────────────────────────────────────────────────

def _cpu_worker(args):
    from scipy.integrate import solve_ivp
    alpha0, beta0 = args
    a0, b0 = np.radians(alpha0), np.radians(beta0)

    def ode(t, y):
        a1, a2, da1, da2 = y
        d = a1 - a2;  sd, cd = np.sin(d), np.cos(d)
        den1 = (M1+M2)*L1 - M2*L1*cd**2
        den2 = (L2/L1)*den1
        dda1 = (-M2*L1*da1**2*sd*cd + M2*G*np.sin(a2)*cd
                - M2*L2*da2**2*sd - (M1+M2)*G*np.sin(a1)) / den1
        dda2 = (M2*L2*da2**2*sd*cd
                + (M1+M2)*(G*np.sin(a1)*cd + L1*da1**2*sd
                            - G*np.sin(a2))) / den2
        return [da1, da2, dda1, dda2]

    t_eval = np.linspace(0, T_MAX, N_STEPS)
    try:
        s1 = solve_ivp(ode, (0, T_MAX), [a0,          b0, 0, 0],
                       t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
        s2 = solve_ivp(ode, (0, T_MAX), [a0+EPSILON,  b0, 0, 0],
                       t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    except Exception:
        return T_MAX
    d1   = _angular_delta_np(s1.y[0], s2.y[0])
    d2   = _angular_delta_np(s1.y[1], s2.y[1])
    dist = np.sqrt(d1**2 + d2**2)
    idx  = np.argmax(dist > DIVERGE_THRESH)
    return T_MAX if (idx == 0 and dist[0] <= DIVERGE_THRESH) else s1.t[idx]


def compute_pth_cpu(pairs):
    from multiprocessing import Pool, cpu_count
    ncpu  = cpu_count()
    total = len(pairs)
    chunk = max(1, total // (ncpu * 20))
    results = []
    done    = 0
    t0      = time.time()

    print(f"  CPU cores: {ncpu}  |  chunk size: {chunk}\n")
    print(f"  {'Progress':<32} {'Done':>8}  {'Elapsed':>8}  {'ETA':>8}")
    print("  " + "─" * 60)

    with Pool(processes=ncpu) as pool:
        for batch in [pairs[i:i+chunk] for i in range(0, total, chunk)]:
            results.extend(pool.map(_cpu_worker, batch))
            done   += len(batch)
            elapsed = time.time() - t0
            pct     = done / total
            eta     = (elapsed / pct - elapsed) if pct > 0 else 0
            bar     = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))
            print(f"  [{bar}]  {done:>6}/{total}  "
                  f"{elapsed:7.1f}s  ~{eta:6.1f}s")
            sys.stdout.flush()

    print()
    return np.array(results)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = _parse_mode(sys.argv)

    print("=" * 60)
    if mode == "pth":
        print("  Double Pendulum — PTH Parameter Space")
    else:
        print("  Double Pendulum — Phase-Color Fractal")
    print("=" * 60)
    print(f"  Grid     : {N_POINTS}×{N_POINTS} = {N_POINTS**2:,} points")
    if mode == "pth":
        print(f"  Sim time : {T_MAX} s  |  Steps: {N_STEPS}  |  dt={T_MAX/N_STEPS:.4f} s")
        print(f"  ε={EPSILON}  |  diverge threshold={DIVERGE_THRESH} rad")
    else:
        n_phase = int(T_PHASE / DT_PHASE)
        print(f"  Sim time : {T_PHASE} s  |  Steps: {n_phase}  |  dt={DT_PHASE:.4f} s")
        print("  Coloring : HSV from tip position  |  flip → black")
    print()

    # ── detect GPU backend ────────────────────────────────────────────────────
    use_gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            use_gpu = True
            print("  ✓ ROCm/CUDA detected — using GPU backend\n")
        else:
            print("  ✗ torch.cuda not available.")
            print("    If you have an RX 6700S or similar gfx1032 card, run as:")
            print("      HSA_OVERRIDE_GFX_VERSION=10.3.0 python double_pendulum_pth.py\n")
    except ImportError:
        print("  PyTorch not found → CPU multiprocessing fallback\n")

    # ── build grid ────────────────────────────────────────────────────────────
    angles     = np.linspace(-180, 180, N_POINTS)
    ag, bg     = np.meshgrid(angles, angles)
    alpha_flat = ag.ravel()
    beta_flat  = bg.ravel()

    # ── compute ───────────────────────────────────────────────────────────────
    t_start = time.time()
    if mode == "pth":
        if use_gpu:
            pth_flat = compute_pth_gpu(alpha_flat, beta_flat)
        else:
            pth_flat = compute_pth_cpu(list(zip(alpha_flat, beta_flat)))
    else:
        phase_rgb_flat = compute_phase_rgb_gpu(alpha_flat, beta_flat)

    total_s = time.time() - t_start
    print(f"  ✓ Finished in {total_s:.1f} s  ({total_s/60:.1f} min)\n")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    if mode == "pth":
        pth_map = pth_flat.reshape(N_POINTS, N_POINTS)
        colors_list = [
            (1.0, 1.0, 0.0),  # yellow  — short PTH (fast divergence)
            (1.0, 0.4, 0.0),  # orange
            (0.8, 0.0, 0.0),  # red
            (0.4, 0.0, 0.0),  # dark red
            (0.0, 0.0, 0.0),  # black   — long PTH (stable over horizon)
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "pth_cmap", colors_list, N=512
        )
        norm = mcolors.PowerNorm(gamma=0.65, vmin=0, vmax=T_MAX)
        im = ax.imshow(
            pth_map,
            extent=[-180, 180, -180, 180],
            origin='lower',
            cmap=cmap,
            norm=norm,
            interpolation='nearest',
            aspect='equal',
        )
    else:
        phase_rgb = phase_rgb_flat.reshape(N_POINTS, N_POINTS, 3)
        plt.imsave("double_pendulum_phase_raw.png",
                   phase_rgb, origin='lower')
        print(f"  Saved raw → double_pendulum_phase_raw.png")
        ax.imshow(
            phase_rgb,
            extent=[-180, 180, -180, 180],
            origin='lower',
            interpolation='bilinear',
            aspect='equal',
        )

    ax.set_xlabel(r'$\alpha_0$', fontsize=14)
    ax.set_ylabel(r'$\beta_0$',  fontsize=14)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-180, 181, 30))
    ax.tick_params(labelsize=10)

    if mode == "pth":
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Predictability Time Horizon [s]', fontsize=12)

    plt.tight_layout()
    out = "double_pendulum_pth.png" if mode == "pth" else "double_pendulum_phase.png"
    save_dpi = 200 if mode == "phase" else 150
    plt.savefig(out, dpi=save_dpi, bbox_inches='tight')
    print(f"  Saved → {out}")
    plt.show()
