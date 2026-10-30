"""
Hourly household load for a middle-income house in Karachi (assumed profile).

There is no public hourly metered dataset for Karachi homes, so the profile is
built bottom-up from appliance blocks and driven by the NASA POWER hourly air
temperature, so that cooling load follows the real weather of each year.
All times are PKT clock hours. Every number here is an assumption and is
listed below.

  base        fridge, router, standby                     0.30 kW, always
  fans        when T > 26 C                               0.20 kW (night 0.15)
  lights      18:00-24:00 0.30 kW, 05:00-07:00 0.15 kW
  morning     water pump + kettle/iron, 07:00-09:00      0.60 kW
  evening     TV, kitchen appliances, 19:00-22:00         0.40 kW
  AC #1       one 1.5-ton inverter AC: 13:00-17:00 and 22:00-07:00, when T > 28 C
              electrical power 0.55 + 0.09*(T-28) kW, max 1.5 kW
  AC #2       second room, 21:00-01:00 when T > 30 C, same law
  weekend     Saturday and Sunday 09:00-17:00 x 1.15
  noise       multiplicative, lognormal with sigma 0.10, fixed seed
"""
import datetime as dt
import numpy as np


def ac_power(t):
    return np.clip(0.55 + 0.09 * (t - 28.0), 0.0, 1.5)


def household_load(w, seed=7, ac_scale=1.0, base_scale=1.0):
    """ac_scale and base_scale are used only by the Monte Carlo study."""
    hr = (w["hour"] + 1) % 24            # PKT clock hour
    t = w["t_amb"]
    p = np.full(len(hr), 0.30 * base_scale)
    night = (hr >= 22) | (hr < 7)
    p += np.where(t > 26, np.where(night, 0.15, 0.20), 0.0)
    p += np.where((hr >= 18) & (hr <= 23), 0.30, 0.0)
    p += np.where((hr >= 5) & (hr < 7), 0.15, 0.0)
    p += np.where((hr >= 7) & (hr < 9), 0.60, 0.0)
    p += np.where((hr >= 19) & (hr < 22), 0.40, 0.0)
    ac1 = (((hr >= 13) & (hr < 17)) | night) & (t > 28)
    p += np.where(ac1, ac_scale * ac_power(t), 0.0)
    ac2 = ((hr >= 21) | (hr < 1)) & (t > 30)
    p += np.where(ac2, ac_scale * ac_power(t), 0.0)
    wd = np.array([dt.date(y, m, d).weekday() for y, m, d in zip(w["year"], w["month"], w["day"])])
    weekend = (wd >= 5) & (hr >= 9) & (hr < 17)
    p = np.where(weekend, p * 1.15, p)
    rng = np.random.default_rng(seed)
    p *= rng.lognormal(0.0, 0.10, len(p))
    return p * 1000.0                    # W
