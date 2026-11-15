import pathlib
import sys

import pytest

from helpers import make_aligned

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = pathlib.Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
import components.loaders as loaders   # noqa: E402

DATA = make_aligned()
loaders.get_aligned = lambda tickers, refresh=False: DATA      # no downloads in tests


def start_app():
    return streamlit_testing.AppTest.from_file(str(APP / "Home.py"), default_timeout=300)


def test_app_runs_with_defaults():
    app = start_app().run()
    assert not app.exception, app.exception
    assert len(app.tabs) == 6 and not app.error
    assert any("Current forecast" in m.value for m in app.markdown)


def test_custom_portfolio_is_marked_exploratory():
    app = start_app().run()
    app.sidebar.selectbox(key="preset").set_value("custom").run()
    app.sidebar.text_area(key="custom_text").set_value("SPY 60, IEF 40")
    app.sidebar.selectbox(key="horizon").set_value(10)
    app.sidebar.multiselect(key="models").set_value(["M1", "M2"])
    app.sidebar.selectbox(key="ensemble").set_value("None")
    app.sidebar.slider(key="cost").set_value(25)
    app.sidebar.button(key="run").click()
    app.run()
    assert not app.exception, app.exception
    assert any("exploratory" in w.value.lower() for w in app.warning)
    assert any(">10 days<" in m.value for m in app.markdown)       # the horizon card


def test_bad_weights_show_an_error_instead_of_crashing():
    app = start_app().run()
    app.sidebar.selectbox(key="preset").set_value("custom").run()
    app.sidebar.text_area(key="custom_text").set_value("SPY sixty")
    app.run()
    assert not app.exception and any("not usable" in e.value for e in app.error)


def test_toggles_do_not_break_the_page():
    app = start_app().run()
    for key in ("vol_rv21", "vol_vix"):
        app.checkbox(key=key).set_value(True)
    app.run()
    assert not app.exception, app.exception
    app.checkbox(key="ov_Buy & hold").set_value(False)
    app.radio(key="ov_kind").set_value("Peak-to-trough loss")
    app.toggle(key="acc_detail").set_value(True)
    app.run()
    assert not app.exception, app.exception


def test_guide_page_renders():
    guide = streamlit_testing.AppTest.from_file(str(APP / "pages" / "1_Guide.py")).run()
    assert not guide.exception and len(guide.expander) > 20


def test_rule_a_sliders_change_the_overlay_numbers():
    app = start_app().run()
    app.sidebar.multiselect(key="models").set_value(["M1"])
    app.sidebar.selectbox(key="ensemble").set_value("None")
    app.run()
    first = app.session_state["output"]["result"].overlay_stats.loc["Rule A [M1]", "avg_exposure"]
    app.sidebar.slider(key="rule_a_threshold").set_value(10)     # act much more often
    app.sidebar.slider(key="rule_a_exposure").set_value(20)
    app.run()
    assert not app.exception, app.exception
    assert app.session_state["output"]["result"].overlay_stats.loc["Rule A [M1]", "avg_exposure"] < first
