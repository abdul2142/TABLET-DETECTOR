# Tablet Inspection Console — container image
#
# Build from this directory, with a trained model already in artifacts/:
#   python train.py
#   docker build -t tablet-console .
#   docker run --rm -p 7860:7860 tablet-console
#
# scipy / scikit-learn ship manylinux wheels, so no build toolchain is needed
# and the slim base stays small.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Bind to every interface: the app defaults to localhost, which would be
    # unreachable from outside the container.
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860 \
    # Keep Gradio's telemetry off by default.
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . ./

# Fail at build time if the model is missing, rather than on every request.
RUN python -c "\
from pathlib import Path; \
p = Path('artifacts/tablet_defect_model.joblib'); \
assert p.exists(), 'No model in artifacts/ - run python train.py before building'; \
print(f'model present: {p.stat().st_size / 1e6:.2f} MB')"

# Drop privileges. /app stays writable so the example photos can be generated
# on first start; mount it read-only and the console simply omits them.
RUN useradd --create-home --uid 1000 console && chown -R console:console /app
USER console

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/').read(1)"

CMD ["python", "app.py"]
