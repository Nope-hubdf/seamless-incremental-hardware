# ===============================================
## 1-st way
from unittest.mock import patch

@pytest.fixture(scope="session")
def mock_session() -> Iterator[dict[str, str]]:
  """Session-scoped fixture to mock the Flask session object."""
  with patch("app.main.session", {}) as session:
    yield session
    
# ===============================================

# ===============================================
## 2-nd way
import pytest

@pytest.fixture(scope="session")
def monkeysession() -> Iterator[pytest.MonkeyPatch]:
  with pytest.MonkeyPatch.context() as mp:
    yield mp

    
@pytest.fixture
def mock_session(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
  """Session-scoped fixture to mock the Flask session object."""
  session: dict[str, str] = {}
  monkeypatch.setattr("app.main.session", session)
  return session
# ===============================================