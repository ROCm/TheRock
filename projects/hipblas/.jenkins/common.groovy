// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean sameOrg=false)
{
    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, null, sameOrg)
        }
    }
    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        if (pullRequest.labels.contains("noSolver"))
        {
            project.paths.build_command = project.paths.build_command.replaceAll(' -c', ' -cn')
        }

        if (pullRequest.labels.contains("debug"))
        {
            project.paths.build_command = project.paths.build_command.replaceAll(' -c', ' -cg')
        }
    }

    String centos = platform.jenkinsLabel.contains('centos7') ? 'source scl_source enable devtoolset-7' : ':'

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${getDependenciesCommand}
                ${centos}
                ${project.paths.build_command}
                """
    platform.runCommand(this, command)
}

def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    String stagingDir = "${project.paths.project_build_prefix}/build/release/clients/staging"

    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        if (pullRequest.labels.contains("debug"))
        {
            stagingDir = "${project.paths.project_build_prefix}/build/debug/clients/staging"
        }
    }

    String gtestCommonEnv = "HIPBLAS_CLIENT_RAM_GB_LIMIT=95 GTEST_LISTENER=NO_PASS_LINE_IN_LOG"

    def command = """#!/usr/bin/env bash
                    set -x
                    pushd ${stagingDir}
                    ${gtestCommonEnv} ./hipblas-test --gtest_output=xml --gtest_color=yes
                    popd
                """

    platform.runCommand(this, command)

    def yamlTestCommand = """#!/usr/bin/env bash
                    set -x
                    pushd ${stagingDir}
                    ${gtestCommonEnv} ./hipblas-test --gtest_output=xml --gtest_color=yes --yaml hipblas_smoke.yaml
                    popd
                """
    platform.runCommand(this, yamlTestCommand)
}

def runCoverageCommand (platform, project, String cmdDir = "release-debug")
{
    //Temporary workaround due to bug in container
    String centos7Workaround = platform.jenkinsLabel.contains('centos7') ? 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib64/' : ''

    String gtestCommonEnv = "HIPBLAS_CLIENT_RAM_GB_LIMIT=95 GTEST_LISTENER=NO_PASS_LINE_IN_LOG"

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/${cmdDir}
                export LD_LIBRARY_PATH=/opt/rocm/lib/
                ${centos7Workaround}
                ${gtestCommonEnv} make coverage_cleanup coverage GTEST_FILTER=-*known_bug*
            """

    platform.runCommand(this, command)

    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/build/${cmdDir}/coverage-report",
                reportFiles: "index.html",
                reportName: "Code coverage report",
                reportTitles: "Code coverage report"])
}

def runPackageCommand(platform, project, jobName, label='', buildDir='')
{
    def command

    label = label != '' ? '-' + label.toLowerCase() : ''
    String ext = platform.jenkinsLabel.contains('ubuntu') ? "deb" : "rpm"
    String dir = jobName.contains('Debug') ? "debug" : "release"

    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        if (pullRequest.labels.contains("debug"))
        {
            dir = "debug"
        }
    }
    if (buildDir != '')
    {
        dir = buildDir
    }

    command = """
            set -x
            cd ${project.paths.project_build_prefix}/build/${dir}
            make package
            mkdir -p package
            if [ ! -z "$label" ]
            then
                for f in hipblas*.$ext
                do
                    mv "\$f" "hipblas${label}-\${f#*-}"
                done
            fi
            mv *.${ext} package/
        """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/${dir}/package/*.${ext}""")
}

return this
