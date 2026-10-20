"""Dashboard auto-arrange: the layout solver (pure) and Layout -> Auto-arrange in the window
(offscreen Qt; QSettings redirected to an INI file in tmp_path, never the registry)."""
import os
import time

import pytest

from waxx.util.dashboard import layout_solver as ls
from waxx.util.dashboard.layout_solver import PanelInfo, solve


def _lab_panels():
    P = PanelInfo
    return [
        P("device_control", min_size=(1100, 560), pref_size=(1600, 680), tier="primary", family="run"),
        P("camera", min_size=(640, 480), pref_size=(1000, 760), tier="primary", family="camera"),
        P("monitor", min_size=(700, 260), pref_size=(1100, 420), tier="primary", family="run"),
        P("interlock", min_size=(600, 160), pref_size=(1400, 240), tier="frequent", family="interlock"),
        P("als", min_size=(320, 360), pref_size=(380, 480), tier="frequent", family="lasers"),
        P("precilaser", min_size=(320, 360), pref_size=(380, 480), tier="frequent", family="lasers"),
        P("magnetometer", min_size=(360, 240), pref_size=(600, 360), tier="status", family="diag"),
        P("bristol", min_size=(360, 240), pref_size=(600, 360), tier="status", family="diag"),
        P("keysight", min_size=(400, 260), pref_size=(700, 380), tier="status", family="diag"),
        P("log", min_size=(420, 160), pref_size=(900, 300), tier="status", family="framework"),
        P("pdxc", min_size=(320, 240), pref_size=(420, 320), tier="rare", family="control"),
        P("relay", min_size=(300, 200), pref_size=(380, 280), tier="rare", family="control"),
    ]


def _ids(sol):
    return [pid for s in sol.slots for pid in s.members]


def _check_geometry(sol, panels):
    """Every panel placed once; on-screen slots at least their minimum and inside the region."""
    by_id = {p.id: p for p in panels}
    assert sorted(_ids(sol)) == sorted(by_id)
    for s in sol.slots:
        assert s.area in ls.AREAS
        assert s.x >= 0 and s.y >= 0
        assert s.x + s.w <= sol.width + 1 and s.y + s.h <= sol.height + 1
        need_w = max(by_id[p].min_size[0] for p in s.members)
        need_h = max(by_id[p].min_size[1] for p in s.members) + (ls.TAB_BAR_H if s.stacked else 0)
        assert s.w >= need_w - 1 and s.h >= need_h - 1, (s, need_w, need_h)
        # raised member is the most important one in its stack
        assert by_id[s.raised].weight == max(by_id[p].weight for p in s.members)


def test_large_display_shows_everything():
    panels = _lab_panels()
    sol = solve(panels, *ls.region_for_display(3840, 2160), max_evals=8000)
    assert sol.fits
    assert sol.hidden() == []
    _check_geometry(sol, panels)


def test_small_display_stacks_status_not_primary():
    panels = _lab_panels()
    sol = solve(panels, *ls.region_for_display(1920, 1080), max_evals=8000)
    assert sol.fits, sol.report()
    _check_geometry(sol, panels)
    hidden = {pid for pid, _under in sol.hidden()}
    assert hidden, "1080p cannot show all of these side by side"
    assert not hidden & {"device_control", "camera", "monitor"}
    # a frequent panel behind a tab could not simply take the place of an
    # on-screen status / rare panel (its slot is too small for it)
    by_id = {p.id: p for p in panels}
    for h in hidden:
        if by_id[h].tier != "frequent":
            continue
        for s in sol.slots:
            if by_id[s.raised].tier in ("status", "rare") and not s.stacked:
                assert not (s.w >= by_id[h].min_size[0] and s.h >= by_id[h].min_size[1]), (h, s, sol.report())


def test_primary_panels_get_spare_room_first():
    panels = _lab_panels()
    sol = solve(panels, *ls.region_for_display(2560, 1440), max_evals=8000)
    assert sol.fits
    assert sol.satisfaction["device_control"] >= max(sol.satisfaction[p] for p in ("magnetometer", "log", "pdxc"))


def test_does_not_fit_is_reported_not_hidden():
    panels = _lab_panels()
    sol = solve(panels, 900, 500, max_evals=3000)
    assert not sol.fits
    assert sol.overflow[0] > 0 or sol.overflow[1] > 0
    assert "DOES NOT FIT" in sol.summary()
    assert sorted(_ids(sol)) == sorted(p.id for p in panels)


def test_deterministic_for_a_seed():
    panels = _lab_panels()
    a = solve(panels, 2000, 1100, max_evals=4000, seed=3)
    b = solve(panels, 2000, 1100, max_evals=4000, seed=3)
    assert [(s.area, s.members) for s in a.slots] == [(s.area, s.members) for s in b.slots]


def test_allowed_areas_respected():
    panels = [PanelInfo("strip", min_size=(800, 100), tier="frequent", areas=("top",)),
              PanelInfo("a", min_size=(400, 300), tier="primary"),
              PanelInfo("b", min_size=(400, 300), tier="status", areas=("left", "right"))]
    sol = solve(panels, 1600, 900, max_evals=2000)
    area = {pid: s.area for s in sol.slots for pid in s.members}
    assert area["strip"] == "top"
    assert area["b"] in ("left", "right")


def test_placements_shape_matches_layout_tables():
    panels = _lab_panels()
    sol = solve(panels, *ls.region_for_display(1920, 1080), max_evals=4000)
    plan = sol.placements()
    assert list(plan) == _ids(sol)            # dock order, raised member first in each stack
    groups: dict[str, list[str]] = {}
    for pid, cfg in plan.items():
        assert set(cfg) == {"placement", "dock_area", "tab_group"}
        assert cfg["dock_area"] in ls.AREAS
        if cfg["placement"] == "tab":
            groups.setdefault(cfg["tab_group"], []).append(pid)
    for s in sol.slots:
        if s.stacked:
            assert groups[f"auto:{s.raised}"] == s.members


def test_family_stacks_together():
    """Two diag panels and two control panels that cannot all show: stacks keep one family each."""
    P = PanelInfo
    panels = [P("main", min_size=(900, 500), tier="primary"),
              P("d1", min_size=(400, 300), tier="status", family="diag"),
              P("d2", min_size=(400, 300), tier="status", family="diag"),
              P("c1", min_size=(400, 300), tier="rare", family="control"),
              P("c2", min_size=(400, 300), tier="rare", family="control")]
    sol = solve(panels, 1320, 820, max_evals=4000)
    fam = {p.id: p.family for p in panels}
    for s in sol.slots:
        behind = {fam[p] for p in s.members[1:]}
        assert len(behind) <= 1, sol.report()


def test_ascii_and_report_render():
    panels = _lab_panels()
    sol = solve(panels, 2400, 1200, max_evals=2000)
    pic = sol.render_ascii(100)
    assert max(len(line) for line in pic.splitlines()) <= 100
    assert "device_control" in pic
    assert "cost" in sol.report()


def test_panels_from_hints_defaults():
    ps = ls.panels_from_hints(["a", "b"], {"a": {"tier": "primary", "min": (500, 400), "pref": (600, 500)}})
    assert ps[0].tier == "primary" and ps[0].min_size == (500, 400) and ps[0].pref == (600, 500)
    assert ps[1].tier == "status" and ps[1].pref == ps[1].min_size


# --- the window ---------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ini(tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from waxx.util.dashboard import dashboard_window as dw
    path = str(tmp_path / "dash.ini")
    real = QSettings
    monkeypatch.setattr(dw, "QSettings", lambda *_a, **_k: real(path, real.Format.IniFormat))
    monkeypatch.setattr(dw.DashboardMainWindow, "_probe_data_dir", lambda self: None)
    return path


def _pump(qapp, cond=lambda: False, rounds=300):
    for _ in range(rounds):
        qapp.processEvents()
        if cond():
            return
        time.sleep(0.01)


def test_center_grows_only_with_side_docks(qapp, ini):
    """The empty central widget may take height only while a left / right dock
    shows: else a window of top / bottom docks would get an empty gap."""
    from PyQt6.QtWidgets import QLabel
    from waxx.util.dashboard import dashboard_window as dw
    from waxx.util.dashboard.panel_container import ClientPanel

    panels = [dw.PanelPlacement(ClientPanel(pid, pid, lambda pid=pid: QLabel(pid)), area=area)
              for pid, area in (("t", "top"), ("s", "left"), ("b", "bottom"))]
    win = dw.DashboardMainWindow("client", "Test dashboard", panels, host_ip="test")
    win.resize(1200, 900)
    win.show()
    _pump(qapp, lambda: win._layout_ready)
    center = win.centralWidget()
    assert center.maximumWidth() == 0
    assert center.maximumHeight() > 10000          # side dock shown
    win.panel("s").hide()
    _pump(qapp, rounds=10)
    assert center.maximumHeight() == 0             # no side dock: no gap
    used = sum(win.panel(p).geometry().height() for p in ("t", "b"))
    assert used >= win._dock_region_size()[1] - 20
    win.panel("s").show()
    _pump(qapp, rounds=10)
    assert center.maximumHeight() > 10000
    win._layout_ready = False
    win.hide()
    win.deleteLater()


def test_toolbar_dropdown_offers_auto_arrange(qapp, ini, monkeypatch):
    from PyQt6.QtWidgets import QLabel
    from waxx.util.dashboard import dashboard_window as dw
    from waxx.util.dashboard.panel_container import ClientPanel

    calls = []
    monkeypatch.setattr(dw.DashboardMainWindow, "_auto_arrange",
                        lambda self, fill_screen=False: calls.append(fill_screen))
    panels = [dw.PanelPlacement(ClientPanel("a", "A", lambda: QLabel("a")), area="right")]
    win = dw.DashboardMainWindow("client", "Test dashboard", panels, host_ip="test")
    win.show()
    _pump(qapp, lambda: win._layout_ready)
    acts = {a.text(): a for a in win._layout_picker.menu.actions() if a.text().startswith("Auto-arrange")}
    assert set(acts) == {"Auto-arrange: fit this window", "Auto-arrange: fit this screen (maximize)"}
    acts["Auto-arrange: fit this window"].trigger()
    acts["Auto-arrange: fit this screen (maximize)"].trigger()
    _pump(qapp, lambda: len(calls) == 2, rounds=50)
    assert calls == [False, True]
    # still there after the dropdown is rebuilt
    win._refresh_layout_picker()
    assert sum(a.text().startswith("Auto-arrange") for a in win._layout_picker.menu.actions()) == 2
    win._layout_ready = False
    win.hide()
    win.deleteLater()


def test_window_auto_arrange_applies_solution(qapp, ini, monkeypatch):
    from PyQt6.QtWidgets import QLabel
    from waxx.util.dashboard import dashboard_window as dw
    from waxx.util.dashboard.panel_container import ClientPanel

    ids = ("main", "cam", "d1", "d2", "c1", "closed")
    panels = [dw.PanelPlacement(ClientPanel(pid, pid.upper(), lambda pid=pid: QLabel(pid)), area="right")
              for pid in ids]
    win = dw.DashboardMainWindow("client", "Test dashboard", panels, host_ip="test")
    win.set_auto_layout_hints({
        "main": {"tier": "primary", "min": (500, 300), "pref": (700, 400)},
        "cam": {"tier": "primary", "min": (400, 300), "pref": (500, 400)},
        "d1": {"tier": "status", "family": "diag", "min": (300, 200)},
        "d2": {"tier": "status", "family": "diag", "min": (300, 200)},
        "c1": {"tier": "rare", "family": "control", "min": (300, 200)},
    })
    win.resize(1300, 800)
    win.show()
    _pump(qapp, lambda: win._layout_ready)
    win.panel("closed").hide()                      # closed by the user: must stay closed
    _pump(qapp, rounds=5)
    monkeypatch.setattr(dw.QMessageBox, "warning", lambda *a, **k: None)

    win._auto_arrange_now()
    _pump(qapp, rounds=20)

    assert win._auto_report and "cost" in win._auto_report
    assert win.panel("closed").isHidden()
    # the solver's plan is what Qt now has
    sol = win._solve_layout(*win._dock_region_size())
    area_of = {"left": dw._AREA_MAP["left"], "right": dw._AREA_MAP["right"],
               "top": dw._AREA_MAP["top"], "bottom": dw._AREA_MAP["bottom"]}
    assert sol is not None
    for pid in ("main", "cam", "d1", "d2", "c1"):
        p = win.panel(pid)
        assert not p.isHidden()
        assert win.dockWidgetArea(p) in area_of.values()
        assert pid.upper() in win._auto_report        # labels in the report
    assert "CLOSED" not in win._auto_report
    # Qt holds the solver's arrangement: same area per panel, same tab stacks
    for slot in sol.slots:
        raised = win.panel(slot.raised)
        assert win.dockWidgetArea(raised) == area_of[slot.area]
        stack = {raised.panel_id} | {d.panel_id for d in win.tabifiedDockWidgets(raised)}
        assert stack == set(slot.members)
    assert sol.elapsed_s < 2.0                         # a small problem stops early
    assert win._layouts.active == ""             # an unsaved arrangement until the user saves it
    win._layout_ready = False
    win.hide()
    win.deleteLater()
