def main() -> None:
    import uvicorn

    from st_service.config import get_settings

    s = get_settings()
    uvicorn.run(
        "st_service.main:app",
        host=s.st_host,
        port=s.st_port,
    )


if __name__ == "__main__":
    main()
