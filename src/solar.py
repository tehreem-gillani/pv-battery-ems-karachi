"""
Weather input and irradiance on the tilted array.

Input: NASA POWER hourly point data for Karachi (24.8607 N, 67.0011 E):
ALLSKY_SFC_SW_DWN (global horizontal, Wh/m^2 per hour), T2M (deg C),
WS2M (m/s). The files were requested with time-standard=LST. Fitting the
clear-sky shape to the data shows that POWER's LST here is UTC+4 (the nominal
zone for 67 E), not Pakistan Standard Time (UTC+5): the best match is at a
solar-time offset of +0.47 h, which equals (67.0 - 60)/15 h. So:
    solar time = HR + 0.5 + (67.0011 - 60)/15 + EoT/60   (value covers HR..HR+1)
    PKT clock hour = HR + 1
The clock hour matters for the load profile, load-shedding and peak tariff.

Steps for every hour:
 1. sun position (declination after Cooper, hour angle from solar time)
 2. clearness index kt = GHI / extraterrestrial horizontal
 3. diffuse fraction from the Erbs correlation  -> DHI, DNI
 4. plane-of-array irradiance, isotropic sky (Liu & Jordan), ground albedo 0.2
 5. incidence-angle loss on the beam part (ASHRAE model, b0 = 0.05)
"""
import numpy as np

LAT = 24.8607
G_SC = 1361.0       # solar constant, W/m^2


def load_nasa_power(path):
    rows = []
    with open(path) as f:
        in_data = False
        for line in f:
            if line.startswith("-END HEADER-"):
                in_data = True
                continue
            if not in_data or line.startswith("YEAR"):
                continue
            rows.append([float(x) for x in line.strip().split(",")])
    a = np.array(rows)
    out = {"year": a[:, 0].astype(int), "month": a[:, 1].astype(int),
           "day": a[:, 2].astype(int), "hour": a[:, 3].astype(int),
           "ghi": a[:, 4], "t_amb": a[:, 5], "wind": a[:, 6]}
    for key in ("ghi", "t_amb", "wind"):
        bad = out[key] < -900
        if bad.any():          # fill missing values from the neighbours
            idx = np.arange(len(a))
            out[key][bad] = np.interp(idx[bad], idx[~bad], out[key][~bad])
    out["ghi"] = np.maximum(out["ghi"], 0.0)
    days_in = np.cumsum([0, 31, 29 if out["year"][0] % 4 == 0 else 28, 31, 30, 31, 30,
                         31, 31, 30, 31, 30, 31])
    doy = days_in[out["month"] - 1] + out["day"]
    out["doy"] = doy
    return out


LON = 67.0011
LST_MERIDIAN = 60.0      # UTC+4, see module docstring


def equation_of_time(doy):
    b = np.radians(360.0 * (doy - 81) / 364.0)
    return 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)     # minutes


def solar_hour_of(w):
    return (w["hour"] + 0.5 + (LON - LST_MERIDIAN) / 15.0 + equation_of_time(w["doy"]) / 60.0)


def pkt_hour(w):
    return (w["hour"] + 1) % 24


def sun_geometry(doy, solar_hour, lat=LAT):
    dec = np.radians(23.45) * np.sin(np.radians(360.0 * (284 + doy) / 365.0))
    w = np.radians(15.0 * (solar_hour - 12.0))
    phi = np.radians(lat)
    cos_z = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(w)
    return dec, w, cos_z


def erbs_diffuse_fraction(kt):
    kt = np.asarray(kt, float)
    f = np.where(kt <= 0.22, 1 - 0.09 * kt,
                 np.where(kt <= 0.80,
                          0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4,
                          0.165))
    return f


def plane_of_array(w, tilt=24.0, azimuth=0.0, albedo=0.2, b0=0.05):
    """azimuth 0 = due south. Returns dict with POA components (W/m^2)."""
    doy, hr, ghi = w["doy"], w["hour"], w["ghi"]
    dec, om, cos_z = sun_geometry(doy, solar_hour_of(w))
    e0 = 1 + 0.033 * np.cos(np.radians(360.0 * doy / 365.0))
    g0h = np.maximum(G_SC * e0 * cos_z, 0.0)
    kt = np.where(g0h > 10, np.clip(ghi / np.maximum(g0h, 1e-9), 0, 1), 0.0)
    fd = erbs_diffuse_fraction(kt)
    dhi = np.where(cos_z > 0.087, fd * ghi, ghi)        # very low sun: treat all as diffuse
    bh = ghi - dhi
    dni = np.where(cos_z > 0.087, bh / np.maximum(cos_z, 0.087), 0.0)

    phi, beta, gam = np.radians(LAT), np.radians(tilt), np.radians(azimuth)
    cos_th = (np.sin(dec) * np.sin(phi) * np.cos(beta)
              - np.sin(dec) * np.cos(phi) * np.sin(beta) * np.cos(gam)
              + np.cos(dec) * np.cos(phi) * np.cos(beta) * np.cos(om)
              + np.cos(dec) * np.sin(phi) * np.sin(beta) * np.cos(gam) * np.cos(om)
              + np.cos(dec) * np.sin(beta) * np.sin(gam) * np.sin(om))
    cos_th = np.maximum(cos_th, 0.0)
    iam = np.where(cos_th > 0.05, np.clip(1 - b0 * (1 / np.maximum(cos_th, 0.05) - 1), 0, 1), 0.0)
    beam = dni * cos_th
    sky = dhi * (1 + np.cos(beta)) / 2
    ground = ghi * albedo * (1 - np.cos(beta)) / 2
    return {"poa": beam + sky + ground, "poa_eff": beam * iam + sky + ground,
            "beam": beam, "kt": kt, "cos_z": cos_z, "dni": dni, "dhi": dhi}


def cell_temperature(poa, t_amb, wind, noct=45.0, eta_over_tau_alpha=0.23):
    """Duffie & Beckman NOCT model with a wind correction (wind at 2 m)."""
    return t_amb + (poa / 800.0) * (noct - 20.0) * (9.5 / (5.7 + 3.8 * wind)) * (1 - eta_over_tau_alpha)
