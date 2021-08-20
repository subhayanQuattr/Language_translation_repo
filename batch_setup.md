<h3>Build new image for language translator & Update the ecr image with latest tag</h3>

```
docker build -t langauge_translation:latest .
docker tag <container_id> <ecr-address>-language-translator-revised
docker push <ecr-address>-language-translator-revised
```

*Note: language-translator is the docker image pushed on ecr & is refered in the batch job definition*
