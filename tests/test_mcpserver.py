"""The MCP server — its tool functions, and the tools as an assistant actually sees them.

Two layers. The tool functions are plain Python and are tested directly (they must work with
no SDK installed). The protocol layer is driven through the SDK's in-process client, which
returns the same result object a real assistant gets — so the schemas, the annotations and
the error text are asserted on the wire, not on our side of it.

Everything runs against the synthetic slide, and every result file is redirected into tmp, so
the suite never touches a real session.
"""
import asyncio
import json
import os

import numpy as np
import pytest

from smile_msi import mcpserver

mcp = pytest.importorskip("mcp", reason="the MCP server needs the optional 'mcp' extra")


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Results in tmp, app home in tmp: a test can neither read nor dirty a real session."""
    monkeypatch.setenv("SMILE_MSI_MCP_OUT", str(tmp_path / "results"))
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path / "home"))
    yield
    mcpserver.close_slide()


@pytest.fixture
def demo_slide():
    mcpserver.open_slide("demo")
    mcpserver.find_peaks(snr=3, max_peaks=12)
    return mcpserver._open["slide"]


def call(tool: str, args: dict | None = None):
    """One tool call over the real protocol, in process. Returns the CallToolResult."""
    from mcp.client import Client

    async def go():
        async with Client(mcpserver.build_server()) as client:
            return await client.call_tool(tool, args or {})

    return asyncio.run(go())


def payload(result) -> dict:
    return json.loads(result.content[0].text)


# --------------------------------------------------------------------------- #
# the tool functions
# --------------------------------------------------------------------------- #
def test_catalog_describes_every_registered_step():
    from smile_msi import registry

    catalog = mcpserver.analysis_catalog()
    assert len(catalog) == len(registry.REGISTRY)
    seg = next(c for c in catalog if c["id"] == "auto_segment")
    assert seg["needs"] == ["feature_set"]
    assert seg["params"]["n_clusters"]["default"] == 0
    assert "tic" in seg["params"]["norm"]["choices"]


def test_catalog_filters_by_category():
    stats = mcpserver.analysis_catalog("Statistics")
    assert stats and all(c["category"] == "Statistics" for c in stats)


def test_run_analysis_returns_a_preview_and_the_whole_csv(demo_slide):
    import pandas as pd

    out = mcpserver.run_analysis("roi_comparison", {"method": "mwu"})
    assert out["summary"]
    table = out["table"]
    assert table["n_rows"] == 12 and len(table["preview"]) == 12
    assert list(pd.read_csv(table["csv"]).columns) == table["columns"]


def test_a_long_table_is_previewed_not_dumped(demo_slide):
    import pandas as pd

    mcpserver.find_peaks(snr=1, max_peaks=200)
    out = mcpserver.run_analysis("roi_comparison")
    assert out["table"]["truncated"] is True
    assert len(out["table"]["preview"]) == mcpserver.PREVIEW_ROWS
    assert len(pd.read_csv(out["table"]["csv"])) == out["table"]["n_rows"]


def test_results_sort_so_the_preview_is_the_interesting_end(demo_slide):
    out = mcpserver.run_analysis("roi_comparison")
    q = [row["q_value"] for row in out["table"]["preview"] if row["q_value"] is not None]
    assert q == sorted(q)


def test_run_analysis_takes_named_regions_as_the_two_groups(demo_slide):
    out = mcpserver.run_analysis("roi_comparison", a="left", b="right")
    assert out["table"]["n_rows"] == 12


def test_non_finite_numbers_survive_json(demo_slide):
    out = mcpserver.run_analysis("roi_comparison")
    json.dumps(out)                                  # NaN would not round-trip as valid JSON


def test_ion_image_writes_a_png_and_reports_the_region_split(demo_slide):
    mz = demo_slide.api.get_features()[0]
    out = mcpserver.ion_image(mz, mask="left")
    assert os.path.getsize(out["png"]) > 0
    assert out["region"]["name"] == "left" and out["region"]["n_pixels"] > 0
    assert out["region"]["mean_inside"] is not None


def test_mean_spectrum_reports_peaks_and_a_figure(demo_slide):
    out = mcpserver.mean_spectrum(top_n=5)
    assert len(out["top_peaks"]) == 5
    assert out["top_peaks"][0]["mz"] > 0
    assert os.path.getsize(out["png"]) > 0


def test_run_script_surfaces_logs_tables_and_captures_print(demo_slide):
    out = mcpserver.run_script(
        "print('to stdout')\nlog('picked')\ntable(compare('Group A', 'Group B').head(2), 'cmp')")
    assert out["ok"] is True
    assert "picked" in out["logs"]
    assert out["stdout"] == "to stdout\n"
    assert out["tables"][0]["n_rows"] == 2


def test_run_script_reports_a_users_error_with_their_line(demo_slide):
    out = mcpserver.run_script("log('fine')\nraise ValueError('boom')")
    assert out["ok"] is False and "boom" in out["error"]


def test_annotate_identifies_the_demo_lipids(demo_slide):
    out = mcpserver.annotate(mode="negative")
    assert out["table"]["n_rows"] == 12
    assert "lipid" in out["table"]["columns"]


def test_mean_spectrum_reports_the_same_mz_as_find_peaks(demo_slide):
    picked = set(mcpserver.find_peaks(snr=3, max_peaks=12)["mz"])
    listed = [p["mz"] for p in mcpserver.mean_spectrum(top_n=12)["top_peaks"]]
    assert listed == sorted(listed)                     # m/z order, as documented
    assert set(listed) == picked                        # one peak, one m/z, in both tools


def test_annotate_measures_the_demo_fatty_acid_isotopes(demo_slide):
    import pandas as pd

    mcpserver.find_peaks(snr=3, max_peaks=40)
    out = mcpserver.annotate(mz=[281.2486, 282.2516])
    fa, m1 = pd.read_csv(out["table"]["csv"]).to_dict("records")
    assert fa["lipid"] == "FA 18:1" and fa["isotope_ok"] and fa["confidence"] == "High"
    assert 0.15 < fa["isotope_m1"] < 0.25
    assert fa["intensity"] > 0 and fa["snr"] > 0
    assert m1["isotopologue"] and "M+1 isotopologue of 281.2486" in m1["confidence_why"]


def test_a_pixel_level_test_carries_its_warning(demo_slide):
    out = mcpserver.run_analysis("multigroup_features", groups=["left", "right"])
    assert "pseudoreplication" in out["summary"]
    assert "pseudoreplication" in out["warning"]


def test_segmentation_reports_its_mask_and_background_clusters(demo_slide):
    whole = mcpserver.run_analysis("auto_segment", {"n_clusters": 2})
    seg = whole["segmentation"]
    assert seg["n_pixels_clustered"] == demo_slide.ds.n_pixels
    assert seg["background_clusters"] and "mask=" in seg["note"]
    left = mcpserver.run_analysis("auto_segment", {"n_clusters": 2}, mask="left")
    assert left["mask"] == {"name": "left", "n_pixels": 792}
    assert left["segmentation"]["n_pixels_clustered"] == 792
    assert "note" not in left["segmentation"]


def test_slide_state_explains_the_demo_has_no_session():
    with pytest.raises(ValueError, match="open_slide"):
        mcpserver.slide_state("demo")


def test_state_requires_an_open_slide():
    mcpserver.close_slide()
    with pytest.raises(RuntimeError, match="no slide open"):
        mcpserver.state()


def test_open_slide_rejects_an_unknown_reference():
    with pytest.raises(FileNotFoundError):
        mcpserver.open_slide("not-a-slide")


def test_no_tool_writes_outside_the_results_directory(demo_slide, tmp_path):
    mcpserver.run_analysis("roi_comparison")
    mcpserver.ion_image(demo_slide.api.get_features()[0])
    home = tmp_path / "home"
    written = [p for p in home.rglob("*") if p.is_file()] if home.exists() else []
    assert written == [], f"a tool wrote into the app home: {written}"


def test_send_regions_queues_staged_regions_for_the_app(demo_slide, tmp_path):
    from smile_msi import regioninbox

    sp = tmp_path / "home" / "sessions" / "slide__abc.json"
    sp.parent.mkdir(parents=True)
    sp.write_text("{}")
    demo_slide.session_path = str(sp)
    out = mcpserver.run_script("add_region('half', region('left'))")
    assert out["staged_regions"] == ["half"]
    sent = mcpserver.send_regions(groups={"half": "Group A"}, parents={"half": "left"})
    assert sent["sent"][0]["name"] == "half" and sent["sent"][0]["group"] == "Group A"
    assert sent["sent"][0]["parent"] == "left"
    assert sp.read_text() == "{}"
    regions, _ = regioninbox.take_regions(str(sp), demo_slide.ds.n_pixels)
    assert np.array_equal(regions[0]["mask"], demo_slide.api.masks["left"])
    with pytest.raises(ValueError, match="not staged"):
        mcpserver.send_regions(["half"])


def test_send_regions_needs_a_saved_session(demo_slide):
    mcpserver.run_script("add_region('half', region('left'))")
    with pytest.raises(RuntimeError, match="no saved session"):
        mcpserver.send_regions()


# --------------------------------------------------------------------------- #
# the protocol surface
# --------------------------------------------------------------------------- #
def test_every_tool_is_registered_with_a_title_and_a_description():
    from mcp.client import Client

    async def go():
        async with Client(mcpserver.build_server()) as client:
            return (await client.list_tools()).tools

    tools = asyncio.run(go())
    assert len(tools) == len(mcpserver.TOOLS)
    for tool in tools:
        assert tool.title and tool.description
        assert not tool.description.startswith(" ")          # docstring was de-indented


def test_read_only_tools_are_annotated_read_only():
    from mcp.client import Client

    async def go():
        async with Client(mcpserver.build_server()) as client:
            return {t.name: t.annotations for t in (await client.list_tools()).tools}

    ann = asyncio.run(go())
    assert ann["list_slides"].read_only_hint is True
    assert ann["run_analysis"].read_only_hint is False
    assert all(a.destructive_hint is False for a in ann.values())


def test_schemas_come_from_the_type_hints():
    from mcp.client import Client

    async def go():
        async with Client(mcpserver.build_server()) as client:
            return {t.name: t.input_schema for t in (await client.list_tools()).tools}

    schemas = asyncio.run(go())
    assert set(schemas["run_analysis"]["properties"]) == {
        "step_id", "params", "features", "a", "b", "region", "groups", "mask", "target_mz"}
    assert schemas["run_analysis"]["required"] == ["step_id"]
    assert schemas["ion_image"]["properties"]["mz"]["type"] == "number"


def test_a_tool_call_round_trips_over_the_protocol():
    assert call("open_slide", {"ref": "demo"}).is_error is False
    peaks = payload(call("find_peaks", {"snr": 3, "max_peaks": 8}))
    assert peaks["n_features"] == 8
    out = payload(call("run_analysis", {"step_id": "roi_comparison"}))
    assert out["step_id"] == "roi_comparison" and out["table"]["n_rows"] == 8


def test_a_failure_reaches_the_assistant_as_its_own_message():
    call("open_slide", {"ref": "demo"})
    result = call("run_analysis", {"step_id": "no_such_step"})
    assert result.is_error is True
    assert "no_such_step" in result.content[0].text
    assert "analysis_catalog" in result.content[0].text


def test_a_missing_feature_set_says_what_to_do_about_it():
    call("open_slide", {"ref": "demo"})
    result = call("run_analysis", {"step_id": "auto_segment"})
    assert result.is_error is True
    assert "find_peaks" in result.content[0].text
