include(ProcessorCount)

function(therock_setup_job_pools)
  set(_background_jobs "${THEROCK_BACKGROUND_BUILD_JOBS}")
  if(NOT _background_jobs OR _background_jobs LESS_EQUAL 0)
    ProcessorCount(CORE_COUNT)
    math(EXPR _background_jobs "${CORE_COUNT} / 20")
    if(_background_jobs LESS 2)
      set(_background_jobs 2)
    endif()
    message(STATUS "Configuring background job pool for ${_background_jobs} concurrent jobs")
  endif()

  set_property(GLOBAL APPEND PROPERTY JOB_POOLS therock_background=${_background_jobs})
endfunction()

therock_setup_job_pools()
