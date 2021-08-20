FROM python:3.6

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY ./KeywordLanguageTranslations.py .

COPY ./lid.176.ftz .

COPY ./special_keywords.json .

CMD ["python", "./KeywordLanguageTranslations.py" ]
