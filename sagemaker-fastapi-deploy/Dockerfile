# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /opt/program

# Copy the requirements file into the container
COPY app/requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application directory (including main.py and data/movies.csv)
COPY ./app /opt/program/app

# Copy the serve script into the container and make it executable
COPY ./serve /opt/program/serve
RUN chmod +x /opt/program/serve

# Change working directory to where the app code is
WORKDIR /opt/program/app

# Expose port 8080 - SageMaker expects the container to listen on this port
EXPOSE 8080

# Define the command that SageMaker will run (via the 'serve' script)
# This CMD is mostly for local testing; SageMaker uses the ENTRYPOINT/serve script directly
CMD ["/opt/program/serve"]