# Use an official Python runtime as a parent image
FROM python:3.12.8

# Set the working directory in the container
WORKDIR /server

# Copy the current directory contents into the container at /app
COPY . /server

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]