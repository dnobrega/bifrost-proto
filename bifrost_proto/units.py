"""Bifrost normalization, numerical conversion factors, and unit labels."""

from __future__ import annotations

import math
from dataclasses import dataclass

C_LIGHT = 2.99792458e10   # cm s^-1
K_B = 1.380658e-16        # erg K^-1
M_H = 1.6726219e-24       # g
M_HE = 6.65e-24           # g
STATAMP_PER_AMP = 1.0 / 3.33564095198152e-10
STATV_PER_VOLT = 1.0 / 299.792458


@dataclass(frozen=True)
class BifrostUnits:
    """Bifrost normalization and conversion factors.

    The normalization is defined by the three fundamental Bifrost units
    ``u_l`` (length), ``u_t`` (time), and ``u_r`` (mass density).  All other
    units are derived from them using the same definitions as Bifrost.
    """

    u_l: float = 1.0e8      # cm
    u_t: float = 1.0e2      # s
    u_r: float = 1.0e-7     # g cm^-3

    @property
    def u_u(self):
        """Velocity unit [cm s^-1]."""
        return self.u_l / self.u_t

    @property
    def u_v(self):
        """Backward-compatible alias for ``u_u``."""
        return self.u_u

    @property
    def u_p(self):
        """Pressure unit [dyn cm^-2]."""
        return self.u_r * self.u_u**2

    @property
    def u_kr(self):
        """Rosseland opacity unit [cm^2 g^-1]."""
        return 1.0 / (self.u_r * self.u_l)

    @property
    def u_ee(self):
        """Specific-energy unit [erg g^-1]."""
        return self.u_u**2

    @property
    def u_e(self):
        """Energy-density unit [erg cm^-3]."""
        return self.u_r * self.u_ee

    @property
    def u_q(self):
        """Volumetric energy-rate unit [erg cm^-3 s^-1]."""
        return self.u_e / self.u_t

    @property
    def u_qor(self):
        """Specific energy-rate unit [erg g^-1 s^-1]."""
        return self.u_ee / self.u_t

    @property
    def u_flux(self):
        """Energy-flux unit [erg cm^-2 s^-1]."""
        return self.u_e * self.u_l / self.u_t

    @property
    def u_B(self):
        """Magnetic-field unit [G]."""
        return self.u_u * math.sqrt(4.0 * math.pi * self.u_r)

    @property
    def u_b(self):
        """Lower-case alias for ``u_B``."""
        return self.u_B

    @property
    def u_i(self):
        """Current-density unit [statA cm^-2]."""
        return self.u_B / self.u_l * C_LIGHT / (4.0 * math.pi)

    @property
    def u_el(self):
        """Electric-field unit [statV cm^-1]."""
        return self.u_B * self.u_l / self.u_t / C_LIGHT

    def factor(self, unit_type: str | None, system: str):
        """Multiplicative factor from stored/native units to *system*.

        ``temperature`` and ``number_density`` are stored in their physical
        table/HION units (K and cm^-3 respectively), so their native-to-CGS
        factor is unity.
        """

        system = system.lower()
        if system == "bifrost":
            return 1.0
        if system not in {"solar", "cgs", "si"}:
            raise ValueError("units must be 'bifrost', 'solar', 'cgs', or 'si'")
        if unit_type is None:
            raise ValueError("Variable unit type is not registered")

        cgs = {
            "density": self.u_r,
            "velocity": self.u_u,
            "momentum_density": self.u_r * self.u_u,
            "pressure": self.u_p,
            "energy_density": self.u_e,
            "specific_energy": self.u_ee,
            "energy_rate_density": self.u_q,
            "specific_energy_rate": self.u_qor,
            "energy_flux": self.u_flux,
            "time": self.u_t,
            "length": self.u_l,
            "magnetic_field": self.u_B,
            "current_density": self.u_i,
            "electric_field": self.u_el,
            "magnetic_gradient": self.u_B / self.u_l,
            "force_density": self.u_p / self.u_l,
            "acceleration": self.u_l / (self.u_t * self.u_t),
            "angle": 1.0,
            "inverse_time": 1.0 / self.u_t,
            "inverse_length": 1.0 / self.u_l,
            "dimensionless": 1.0,
            "temperature": 1.0,       # K
            "number_density": 1.0,    # cm^-3
            "instrument_emissivity": 1.0, # DN cm^-1 s^-1 pix^-1
            "opacity": self.u_kr,     # cm^2 g^-1
        }

        if unit_type not in cgs:
            raise ValueError(f"No unit conversion registered for unit_type={unit_type!r}")

        if system == "cgs":
            return cgs[unit_type]

        if system == "solar":
            solar = {
                "density": self.u_r,                       # g cm^-3
                "velocity": self.u_u / 1.0e5,             # km s^-1
                "momentum_density": cgs["momentum_density"],
                "pressure": self.u_p,                      # dyn cm^-2
                "energy_density": self.u_e,                # erg cm^-3
                "specific_energy": self.u_ee,              # erg g^-1
                "energy_rate_density": self.u_q,            # erg cm^-3 s^-1
                "specific_energy_rate": self.u_qor,         # erg g^-1 s^-1
                "energy_flux": self.u_flux,                   # erg cm^-2 s^-1
                "time": self.u_t,                           # s
                "length": self.length_to_Mm,                 # Mm
                "magnetic_field": self.u_B,                # G
                "current_density": self.u_i,               # statA cm^-2
                "electric_field": self.u_el,               # statV cm^-1
                "magnetic_gradient": self.u_B / self.length_to_Mm, # G Mm^-1
                "force_density": self.u_p / self.u_l,      # dyn cm^-3
                "acceleration": self.u_l / (self.u_t * self.u_t) / 1.0e5, # km s^-2
                "angle": 180.0 / math.pi,                  # deg
                "inverse_time": 1.0 / self.u_t,            # s^-1
                "inverse_length": 1.0 / self.length_to_Mm, # Mm^-1
                "dimensionless": 1.0,
                "temperature": 1.0,                        # K
                "number_density": 1.0,                     # cm^-3
                "instrument_emissivity": 1.0,              # DN cm^-1 s^-1 pix^-1
                "opacity": self.u_kr,                      # cm^2 g^-1
            }
            return solar[unit_type]

        # CGS -> SI conversion factors.
        si = {
            "density": cgs["density"] * 1.0e3,                    # kg m^-3
            "velocity": cgs["velocity"] * 1.0e-2,                # m s^-1
            "momentum_density": cgs["momentum_density"] * 10.0,   # kg m^-2 s^-1
            "pressure": cgs["pressure"] * 0.1,                    # Pa
            "energy_density": cgs["energy_density"] * 0.1,        # J m^-3
            "specific_energy": cgs["specific_energy"] * 1.0e-4,  # J kg^-1
            "energy_rate_density": cgs["energy_rate_density"] * 0.1, # W m^-3
            "specific_energy_rate": cgs["specific_energy_rate"] * 1.0e-4, # W kg^-1
            "energy_flux": cgs["energy_flux"] * 1.0e-3,             # W m^-2
            "time": cgs["time"],                                    # s
            "length": cgs["length"] * 1.0e-2,                       # m
            "magnetic_field": cgs["magnetic_field"] * 1.0e-4,    # T
            "current_density": cgs["current_density"] * 3.33564095198152e-6,
            "electric_field": cgs["electric_field"] * 29979.2458,
            "magnetic_gradient": cgs["magnetic_gradient"] * 1.0e-2, # T m^-1
            "force_density": cgs["force_density"] * 10.0,         # N m^-3
            "acceleration": cgs["acceleration"] * 1.0e-2,         # m s^-2
            "angle": 1.0,                                           # rad
            "inverse_time": cgs["inverse_time"],                  # s^-1
            "inverse_length": cgs["inverse_length"] * 1.0e2,      # m^-1
            "dimensionless": 1.0,
            "temperature": 1.0,
            "number_density": 1.0e6,                              # m^-3
            "instrument_emissivity": 1.0e2,                       # DN m^-1 s^-1 pix^-1
            "opacity": cgs["opacity"] * 0.1,                      # m^2 kg^-1
        }
        return si[unit_type]


    def label(self, unit_type: str | None, system: str, *, latex: bool = False):
        """Return the display label for *unit_type* in *system*.

        Labels are intentionally independent of the numerical normalization,
        so they are safe to use directly in plot axes and colorbars.
        """
        system = system.lower()
        if system not in {"bifrost", "solar", "cgs", "si"}:
            raise ValueError("units must be 'bifrost', 'solar', 'cgs', or 'si'")
        if unit_type is None:
            raise ValueError("Variable unit type is not registered")

        if system == "bifrost":
            return "bifrost"

        plain = {
            "solar": {
                "density": "g cm^-3",
                "velocity": "km s^-1",
                "momentum_density": "g cm^-2 s^-1",
                "pressure": "dyn cm^-2",
                "energy_density": "erg cm^-3",
                "specific_energy": "erg g^-1",
                "energy_rate_density": "erg cm^-3 s^-1",
                "specific_energy_rate": "erg g^-1 s^-1",
                "energy_flux": "erg cm^-2 s^-1",
                "time": "s",
                "length": "Mm",
                "magnetic_field": "G",
                "current_density": "statA cm^-2",
                "electric_field": "statV cm^-1",
                "magnetic_gradient": "G Mm^-1",
                "force_density": "dyn cm^-3",
                "acceleration": "km s^-2",
                "angle": "deg",
                "inverse_time": "s^-1",
                "inverse_length": "Mm^-1",
                "dimensionless": "",
                "temperature": "K",
                "number_density": "cm^-3",
                "instrument_emissivity": "DN cm^-1 s^-1 pix^-1",
                "opacity": "cm^2 g^-1",
            },
            "cgs": {
                "density": "g cm^-3",
                "velocity": "cm s^-1",
                "momentum_density": "g cm^-2 s^-1",
                "pressure": "dyn cm^-2",
                "energy_density": "erg cm^-3",
                "specific_energy": "erg g^-1",
                "energy_rate_density": "erg cm^-3 s^-1",
                "specific_energy_rate": "erg g^-1 s^-1",
                "energy_flux": "erg cm^-2 s^-1",
                "time": "s",
                "length": "cm",
                "magnetic_field": "G",
                "current_density": "statA cm^-2",
                "electric_field": "statV cm^-1",
                "magnetic_gradient": "G cm^-1",
                "force_density": "dyn cm^-3",
                "acceleration": "cm s^-2",
                "angle": "rad",
                "inverse_time": "s^-1",
                "inverse_length": "cm^-1",
                "dimensionless": "",
                "temperature": "K",
                "number_density": "cm^-3",
                "instrument_emissivity": "DN cm^-1 s^-1 pix^-1",
                "opacity": "cm^2 g^-1",
            },
            "si": {
                "density": "kg m^-3",
                "velocity": "m s^-1",
                "momentum_density": "kg m^-2 s^-1",
                "pressure": "Pa",
                "energy_density": "J m^-3",
                "specific_energy": "J kg^-1",
                "energy_rate_density": "W m^-3",
                "specific_energy_rate": "W kg^-1",
                "energy_flux": "W m^-2",
                "time": "s",
                "length": "m",
                "magnetic_field": "T",
                "current_density": "A m^-2",
                "electric_field": "V m^-1",
                "magnetic_gradient": "T m^-1",
                "force_density": "N m^-3",
                "acceleration": "m s^-2",
                "angle": "rad",
                "inverse_time": "s^-1",
                "inverse_length": "m^-1",
                "dimensionless": "",
                "temperature": "K",
                "number_density": "m^-3",
                "instrument_emissivity": "DN m^-1 s^-1 pix^-1",
                "opacity": "m^2 kg^-1",
            },
        }

        if unit_type not in plain[system]:
            raise ValueError(f"No unit label registered for unit_type={unit_type!r}")
        label = plain[system][unit_type]
        if not latex or not label:
            return label

        replacements = {
            "^-1": "$^{-1}$",
            "^-2": "$^{-2}$",
            "^-3": "$^{-3}$",
            "^2": "$^{2}$",
        }
        for old, new in replacements.items():
            label = label.replace(old, new)
        return label

    @property
    def length_to_Mm(self):
        """Length normalization expressed in megameters per Bifrost unit."""
        return self.u_l / 1.0e8
