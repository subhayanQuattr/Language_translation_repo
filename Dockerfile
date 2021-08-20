FROM python:3.6

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY ./KeywordLanguageTranslations.py .

COPY ./lid.176.ftz .

ADD ./NVSO_data ./NVSO_data

ADD ./NVSP_data ./NVSP_data

ADD ./NVS_data ./NVS_data

ADD ./ARYK_data ./ARYK_data

CMD ["python", "./KeywordLanguageTranslations.py" ]
