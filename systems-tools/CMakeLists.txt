################################################################################
# ROCm Systems Tools
################################################################################

if(THEROCK_ENABLE_RDC)
    ##########
    # RDC (ROCm Data Center Tool)
    ##########

    therock_cmake_subproject_declare(rdc
        EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rdc"
        BACKGROUND_BUILD

        CMAKE_ARGS
            -DBUILD_PROFILER=ON
            # disable rdcd and rdci (need to discuss)
            -DBUILD_STANDALONE=OFF
            -DBUILD_RUNTIME=ON
            -DBUILD_RVS=OFF
            -DBUILD_TESTS=ON
            -DHIP_PLATFORM=amd

        BUILD_DEPS
            amd-llvm

        RUNTIME_DEPS
            ROCR-Runtime
            amdsmi
            rocprofiler-sdk
            ${THEROCK_BUNDLED_LIBCAP}
            ${THEROCK_BUNDLED_ZLIB}

        INTERFACE_LINK_DIRS
            lib
    )

    therock_cmake_subproject_glob_c_sources(rdc
        SUBDIRS .
    )

    therock_cmake_subproject_provide_package(rdc rdc lib/rdc)

    therock_cmake_subproject_activate(rdc)

    therock_test_validate_shared_lib(
        PATH rdc/dist/lib
        LIB_NAMES
            librdc_rocr.so
            librdc_rocp.so
            librdc_rvs.so
    )

    therock_provide_artifact(rdc
        TARGET_NEUTRAL
        DESCRIPTOR artifact-rdc.toml
        COMPONENTS
            dbg
            dev
            doc
            lib
            run
        SUBPROJECT_DEPS
            amdsmi
            rocprofiler-sdk
            rdc
    )

endif()
