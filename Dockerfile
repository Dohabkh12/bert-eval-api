FROM python:3.10

WORKDIR /code

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY app.py .

# Exposer le port que Hugging Face attend
EXPOSE 7860

# Commande pour démarrer l'API Flask
CMD ["python", "app.py"]
