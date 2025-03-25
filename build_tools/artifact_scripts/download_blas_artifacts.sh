#!/bin/bash

aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/core-runtime_run"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/core-runtime_lib"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/sysdeps_lib"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/base_lib"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/amd-llvm_lib"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/core-hip_lib"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/host-blas_lib"${VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/blas_lib"${TARGET_VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
aws s3 cp s3://therock-artifacts/${ARTIFACT_RUN_ID}/blas_test"${TARGET_VARIANT}".tar.xz "${BUILD_ARTIFACTS_DIR}" --no-sign-request
