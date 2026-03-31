from unittest.mock import patch


def test_cli_main_calls_uvicorn():
    with patch("uvicorn.run") as run:
        from st_service.cli import main

        main()
        run.assert_called_once()
        assert "st_service.main:app" in run.call_args[0][0]
