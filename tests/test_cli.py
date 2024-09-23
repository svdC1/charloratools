import charloratools as clt


def test_cli(caplog):
    caplog.set_level("INFO")
    clt.cli.run_install_script()
    found_torch = ("Found torch" in ''.join(caplog.messages))
    installed_torch = ("Completed" in ''.join(caplog.messages))
    assert found_torch or installed_torch
