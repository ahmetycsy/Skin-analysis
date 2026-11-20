# Hugging Face Spaces için. Uygulama Streamlit ile, işlemcide (CPU) çalışır.
# Model dosyaları burada YOK: açılışta MODEL_REPO'daki Hugging Face model reposundan indirilir.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/home/user PATH=/home/user/.local/bin:$PATH
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt requirements-model.txt ./
RUN pip install -r requirements.txt -r requirements-model.txt

COPY --chown=user . .
USER user

EXPOSE 8501
# Spaces uygulamayı bir çerçeve (iframe) içinde gösterir; dosya yüklemenin çalışması için XSRF/CORS kapalı olmalı.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
