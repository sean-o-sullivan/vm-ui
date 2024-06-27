# Use an official Python runtime as a parent image
FROM python:3.12.3

# Create a user to run the application
RUN useradd -m -u 1000 user

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY --chown=user ./requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code into the container
COPY --chown=user . /app

# Change to the non-root user
USER user

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "app.py"]
