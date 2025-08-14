# Various top level targets that have dependencies added to them build system
# wide. In general, the canonical name for all such targets is prefixed with
# "therock-{name}", but also bare name targets are defined if not already.

function(therock_add_convenience_target name)
  add_custom_target("therock-${name}")
  if(NOT TARGET "${name}")
    add_custom_target("${name}" DEPENDS "therock-${name}")
  endif()
endfunction()

function(therock_add_convenience_target_all name)
  add_custom_target("therock-${name}" ALL)
  if(NOT TARGET "${name}")
    add_custom_target("${name}" DEPENDS "therock-${name}")
  endif()
endfunction()

# Populates all artifacts and distributions (i.e. all artifact-foo targets).
therock_add_convenience_target(artifacts)
# Builds archives for all artifacts (i.e. all archive-foo targets).
therock_add_convenience_target(archives)
# Populates all distribution directories (i.e. all dist-foo targets).
therock_add_convenience_target(dist)
# Expunges configure/build byproducts and artifacts for all projects.
therock_add_convenience_target(expunge)
