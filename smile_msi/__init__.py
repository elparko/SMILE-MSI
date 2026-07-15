"""smile_msi (bare-bones): in-silico m/z -> lipid annotation for MALDI-MSI."""
__version__ = "2.0.0"            # single source of truth (pyproject reads this; see update check)
APP_NAME = "SMILE MSI"           # user-facing app name (menu bar, Dock, window titles, About)

from .lipiddb import Lipid, build_database
from .match import Annotator, Candidate
from .masses import ion_mz, ppm_error, formula_mass

__all__ = ["Lipid", "build_database", "Annotator", "Candidate",
           "ion_mz", "ppm_error", "formula_mass", "__version__", "APP_NAME"]
