# Deployment Model to Cloud Run Documentation

This documentation provides instructions for setting up a machine learning API using Flask, Google Cloud Build, and Google Cloud Run. Follow the steps below to deploy the API on the cloud.

## Step 1: Developing the Flask API

- Begin by creating the Flask API using Python programming language.
- Ensure that the machine learning model is saved in the '.h5' format and placed in the same directory as the main.py file.
- Load the machine learning model into memory and incorporate the necessary Flask code within the main.py file.

## Step 2: Setting Up Google Cloud

- Enable the Cloud Build and Cloud Run APIs.

## Step 3: Make a Dockerfile, requirements.txt, and .dockerignore

- To define the Docker image for your API, you need to create a Dockerfile. For detailed information on how to create this file, you can refer to [this guide](https://cloud.google.com/run/docs/quickstarts/build-and-deploy#containerizing) for more detailed information.
- To exclude specific files and directories when building the Docker image, you can create a .dockerignore file and specify the files and directories that should be excluded. This ensures that only relevant files are included in the final Docker image.
- Prepare a requirements.txt file that contains a list of all the necessary Python packages required for your API.

## Step 4: Cloud Build and Deployment to Cloud Run

- Build the Docker image and submit it to Google Container Registry using the following command:

```shell
gcloud builds submit --tag gcr.io/project-id/container-name
```

- Deploy the built image to Cloud Run using the following command:

```shell
gcloud run deploy --image gcr.io/project-id/container-name --platform managed
```

By executing these commands, you will initiate the Cloud Build process to build the Docker image and store it in the Google Container Registry. After that, the image will be deployed to Cloud Run, enabling you to run your API as a managed service.
