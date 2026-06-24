import json
from unittest.mock import patch

from agentic_reg.build import main


def test_build_no_enrich_creates_graph_file(tmp_path):
    with patch("agentic_reg.build.VectorIndex.build") as mock_vectors:
        mock_vectors.return_value.chunk_count = 12
        with patch("agentic_reg.build.PROJECT_ROOT", tmp_path):
            main(["--domain", "gdpr", "--no-enrich"])

    graph_path = tmp_path / "data" / "store" / "gdpr" / "graph.json"
    assert graph_path.exists()

    data = json.loads(graph_path.read_text(encoding="utf-8"))
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) >= 7


def test_build_rejects_unknown_domain():
    try:
        with patch("agentic_reg.build.VectorIndex.build"):
            with patch("agentic_reg.build.PROJECT_ROOT"):
                main(["--domain", "nonexistent", "--no-enrich"])
        raise AssertionError("Expected KeyError")
    except KeyError:
        pass
