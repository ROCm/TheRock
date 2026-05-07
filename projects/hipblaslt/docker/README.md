# Build hipBLASLt docker image from Dockerfile
docker build .

# Keep all build materials inside the docker
docker build . --build-arg KEEP_BUILD_FOLDER=True

Build folder is in /root/hipBLASLt/build
