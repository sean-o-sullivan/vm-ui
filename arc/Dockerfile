FROM python:3.12.3


RUN useradd -m -u 1000 user


WORKDIR /app


COPY --chown=user ./requirements.txt requirements.txt


RUN pip install --no-cache-dir --upgrade -r requirements.txt


COPY --chown=user . /app


USER user


EXPOSE 7860


CMD ["python", "app.py"]
