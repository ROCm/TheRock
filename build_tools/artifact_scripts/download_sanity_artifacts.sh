#!/bin/bash

aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/core-runtime_run"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/core-runtime_lib"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/sysdeps_lib"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/base_lib"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/amd-llvm_run"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/amd-llvm_lib"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
