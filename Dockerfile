FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples

RUN pip install --no-cache-dir .

# The offline demo. The build above resolves the build backend and pydantic
# from PyPI; the run makes no network call and needs no API key, because the
# embedder and the generator are injected callables and neither is wired to
# anything here. `docker run <image>` prints the table captured in the README.
CMD ["python", "-m", "examples.maintenance_kb"]
