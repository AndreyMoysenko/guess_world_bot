FROM python:3.9-slim

WORKDIR /app
COPY requirements.app.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y curl
COPY . /app
RUN chmod +x /app/start_services.sh

# Expose port 5000 for Flask
EXPOSE 5000
CMD ["bash", "start_services.sh"]