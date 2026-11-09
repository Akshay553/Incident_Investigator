import pytest

from models import IncidentCatalog, SimulationSettings, ToolFamily, ToolQuery
from tools import DATA_DIR, SimulatedTools


@pytest.fixture(scope="session")
def catalog():
    return IncidentCatalog.model_validate_json(
        (DATA_DIR / "incidents.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def tools(catalog):
    return SimulatedTools(catalog, SimulationSettings(latency_ms=0))


@pytest.fixture
def query(catalog):
    return ToolQuery(
        incident_id="inc-001",
        family=ToolFamily.LOGS,
        time_range=catalog.incidents[0].context.time_range,
        services=("checkout",),
    )
