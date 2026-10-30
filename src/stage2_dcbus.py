"""
Stage 2: battery + bidirectional DC-DC converter regulating the 400 V DC bus.

Averaged model of a synchronous half-bridge between the 204.8 V LFP pack and
the bus (boost when discharging, buck when charging):

    L di/dt   = v_bat - R_L*i - (1-d)*v_dc        (i > 0: battery discharging)
    C dv_dc/dt = (1-d)*i + i_pv - i_inv

Cascaded control:
  inner loop  current PI, crossover 1 kHz, with v_bat and v_dc feed-forward
              so the plant seen by the PI is 1/(sL + R_L)
  outer loop  bus-voltage PI, crossover 40 Hz, output is the current reference
              (limited to the battery's +/-1C, with anti-windup)
  optional    power feed-forward i_ff = (P_inv - P_pv)/v_bat, i.e. the outer PI
              only has to correct the error left by the power-balance estimate

The controller is digital: it samples every T_S = 50 us (20 kHz, one update per
switching period) and holds its output in between. The feed-forward uses
measured powers through a 1 ms low-pass filter with a 3 % gain error, so it is
not an ideal, perfectly known disturbance.

The PV side is the stage-1 boost (its own MPPT) and appears as a power source.
The inverter is a power sink using the loss model in inverter.py.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

from battery import Pack
from inverter import efficiency as eta_inv
import style

V_REF = 400.0
L_B, R_LB, C_DC = 1.0e-3, 0.05, 2.2e-3
I_MAX = 50.0                      # 1C for a 50 Ah pack
ETA_BOOST = 0.98
V_BAT_NOM = 204.8

WC_I = 2 * np.pi * 1000.0
KP_I = WC_I * L_B
KI_I = KP_I * WC_I / 10.0
WC_V = 2 * np.pi * 40.0
KP_V = WC_V * C_DC * V_REF / V_BAT_NOM
KI_V = KP_V * WC_V / 4.0
T_S = 50e-6                       # control sample period
TAU_FF = 1e-3                     # power-measurement filter
FF_GAIN_ERR = 0.97                # feed-forward under-estimates by 3 %


def events(t):
    """PV power (W, at the array) and AC load (W) versus time."""
    p_pv = 5000.0 if t < 0.30 else (1000.0 if t < 0.70 else 5400.0)
    p_ld = 1000.0 if t < 0.10 else (5000.0 if t < 0.50 else 500.0)
    return p_pv, p_ld


def simulate(feedforward=True, t_end=0.9, dt=5e-6, switched=False):
    """switched=True replaces the duty ratio by the real switching function of
    the half-bridge (20 kHz, centre-aligned triangle carrier, controller
    sampled at the carrier peak), to check the averaged model."""
    bat = Pack(soc0=0.6)
    n = int(t_end / dt)
    v_dc = V_REF
    p_pv0, p_ld0 = events(0.0)
    p_inv0 = p_ld0 / eta_inv(p_ld0)
    i = (p_inv0 - ETA_BOOST * p_pv0) / bat.ocv()   # start in steady state
    int_v = int_i = 0.0
    if not feedforward:
        int_v = i / KI_V
    log = {k: [] for k in ["t", "v_dc", "i_bat", "i_ref", "p_pv", "p_ld", "soc"]}
    every = max(int(round(T_S / dt)), 1)
    p_net_f = p_inv0 - ETA_BOOST * p_pv0          # filtered measurement
    one_minus_d_c, i_ref = (bat.v_terminal(i) - R_LB * i) / v_dc, i
    for k in range(n):
        t = k * dt
        p_pv, p_ld = events(t)
        p_inv = p_ld / eta_inv(p_ld)
        i_pv = ETA_BOOST * p_pv / v_dc
        i_inv = p_inv / v_dc
        v_bat = bat.v_terminal(i)
        p_net_f += (p_inv - ETA_BOOST * p_pv - p_net_f) * dt / TAU_FF

        if k % every == 0:
            # outer loop
            e_v = V_REF - v_dc
            i_ff = FF_GAIN_ERR * p_net_f / v_bat if feedforward else 0.0
            i_ref_u = KP_V * e_v + KI_V * int_v + i_ff
            i_ref = float(np.clip(i_ref_u, -I_MAX, I_MAX))
            if i_ref == i_ref_u or np.sign(e_v) != np.sign(i_ref_u):
                int_v += e_v * T_S
            # inner loop, output is the desired inductor voltage
            e_i = i_ref - i
            u = KP_I * e_i + KI_I * int_i
            one_minus_d = (v_bat - R_LB * i - u) / v_dc
            one_minus_d_c = float(np.clip(one_minus_d, 0.05, 1.0))
            if one_minus_d_c == one_minus_d:
                int_i += e_i * T_S

        if switched:
            ph = (t / T_S) % 1.0
            carrier = 2.0 * abs(ph - 0.5)              # 1 at k*T_S, 0 at mid-period
            q = 1.0 if carrier < one_minus_d_c else 0.0   # upper switch on
        else:
            q = one_minus_d_c
        di = (v_bat - R_LB * i - q * v_dc) / L_B
        dv = (q * i + i_pv - i_inv) / C_DC
        bat.step(i, dt)
        i += di * dt
        v_dc += dv * dt
        if k % (20 if not switched else 4) == 0:
            for key, val in zip(log, [t, v_dc, i, i_ref, p_pv, p_ld, bat.soc]):
                log[key].append(val)
    return {k: np.array(v) for k, v in log.items()}


def metrics(log, t_events=(0.10, 0.30, 0.50, 0.70), band=0.01):
    out = []
    t, v = log["t"], log["v_dc"]
    for j, te in enumerate(t_events):
        t_next = t_events[j + 1] if j + 1 < len(t_events) else t[-1]
        m = (t >= te) & (t < t_next)
        dev = v[m] - V_REF
        k_peak = int(np.argmax(np.abs(dev)))
        outside = np.flatnonzero(np.abs(dev) > band * V_REF)
        ts = (t[m][outside[-1]] - te) * 1000 if len(outside) else 0.0
        out.append({"t_event_s": te, "peak_dev_V": round(float(dev[k_peak]), 2),
                    "settle_1pct_ms": round(float(ts), 1)})
    return out


def run(outdir_fig="../figures", outdir_res="../results"):
    res = {"gains": {"Kp_i": round(KP_I, 3), "Ki_i": round(KI_I, 1),
                     "Kp_v": round(KP_V, 3), "Ki_v": round(KI_V, 1)}}
    logs = {}
    for ff in [False, True]:
        lg = simulate(feedforward=ff)
        logs[ff] = lg
        res["with_ff" if ff else "pi_only"] = {
            "events": metrics(lg),
            "v_min_V": round(float(lg["v_dc"].min()), 2),
            "v_max_V": round(float(lg["v_dc"].max()), 2)}
    bat = Pack()
    res["pack"] = {"e_nom_kWh": round(bat.e_nom_kwh, 2), "r0_ohm": bat.r0,
                   "r1_ohm": round(bat.r1, 4), "tau1_s": 40.0}
    res["round_trip_eff_pct"] = {f"{p} kW": round(float(100 * Pack().round_trip_efficiency(p)), 2)
                                 for p in [1.0, 2.5, 5.0]}

    fig, ax = plt.subplots(3, 1, figsize=(3.5, 4.6), sharex=True)
    lg = logs[True]
    ax[0].plot(lg["t"], lg["p_pv"] / 1000, label="PV (array)")
    ax[0].plot(lg["t"], lg["p_ld"] / 1000, label="AC load")
    ax[0].set_ylabel("Power (kW)"); ax[0].set_ylim(0, 7.5); ax[0].legend(fontsize=7, ncol=2, loc="upper center")
    for ff, lab in [(False, "PI only"), (True, "PI + power FF")]:
        ax[1].plot(logs[ff]["t"], logs[ff]["v_dc"], lw=0.8, label=lab)
    ax[1].axhspan(V_REF * 0.99, V_REF * 1.01, color="0.85", lw=0, label="$\\pm$1 % band")
    ax[1].set_ylabel("$v_{dc}$ (V)"); ax[1].legend(fontsize=7)
    ax[2].plot(lg["t"], lg["i_bat"], lw=0.8, label="$i_{bat}$")
    ax[2].plot(lg["t"], lg["i_ref"], lw=0.8, ls="--", label="$i_{ref}$")
    ax[2].set_ylabel("Battery current (A)"); ax[2].set_xlabel("Time (s)")
    ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{outdir_fig}/s2_dcbus.png", dpi=200); plt.close(fig)

    # zoom on the first load step
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for ff, lab in [(False, "PI only"), (True, "PI + power FF")]:
        m = (logs[ff]["t"] > 0.095) & (logs[ff]["t"] < 0.16)
        ax.plot((logs[ff]["t"][m] - 0.1) * 1000, logs[ff]["v_dc"][m], lw=0.9, label=lab)
    ax.axhspan(V_REF * 0.99, V_REF * 1.01, color="0.85", lw=0)
    ax.set_xlabel("Time after 1 kW to 5 kW load step (ms)"); ax.set_ylabel("$v_{dc}$ (V)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{outdir_fig}/s2_loadstep_zoom.png", dpi=200); plt.close(fig)

    # averaged vs switched model around the first load step
    sw = simulate(True, t_end=0.16, dt=2.5e-7, switched=True)
    av = logs[True]
    m_ss = (sw["t"] > 0.06) & (sw["t"] < 0.095)
    win = (sw["t"] > 0.095) & (sw["t"] < 0.15)      # stay clear of the filter edge
    per = int(round(T_S / (sw["t"][1] - sw["t"][0])))
    v_avg_sw = np.convolve(sw["v_dc"], np.ones(per) / per, mode="same")
    v_av_i = np.interp(sw["t"], av["t"], av["v_dc"])
    res["switched_check"] = {
        "v_ripple_pp_V": round(float(np.ptp(sw["v_dc"][m_ss])), 3),
        "i_ripple_pp_A": round(float(np.ptp(sw["i_bat"][m_ss])), 2),
        "max_diff_period_avg_vs_averaged_V": round(float(np.max(np.abs(v_avg_sw[win] - v_av_i[win]))), 3),
        "peak_dev_switched_V": round(float(np.min(v_avg_sw[win]) - V_REF), 2),
        "peak_dev_averaged_V": round(float(np.min(v_av_i[win]) - V_REF), 2)}
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    mm = (sw["t"] > 0.097) & (sw["t"] < 0.13)
    ax.plot((sw["t"][mm] - 0.1) * 1000, v_av_i[mm], lw=2.0, c="tab:orange", alpha=0.6, label="averaged model")
    ax.plot((sw["t"][mm] - 0.1) * 1000, sw["v_dc"][mm], lw=0.4, c="k", label="switched, 20 kHz PWM")
    ax.set_xlabel("Time after 1 kW to 5 kW load step (ms)"); ax.set_ylabel("$v_{dc}$ (V)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{outdir_fig}/s2_switched_vs_averaged.png", dpi=200); plt.close(fig)

    with open(f"{outdir_res}/stage2.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
