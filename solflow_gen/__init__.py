# SolFlow_gen: generative model of solar irradiance.
#
# Samples whole-day clear-sky-index (CSI) trajectories from sun geometry, site
# coordinates and the calendar date only. No measured history and no weather
# forecast is used at generation time.
#
# GeometryFlow (models.flow) is the main model: conditional flow matching with a
# seasonal clear-sky prior and an optional sky-regime variable.
# DeepQuantile + GaussianTemporalCopula is the baseline it is compared against.

__version__ = "0.1.0"

from .config import DEFAULT_CONFIG, load_config

__all__ = ["DEFAULT_CONFIG", "load_config", "__version__"]
