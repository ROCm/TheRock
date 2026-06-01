#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName, buildCommand, label->

    def prj  = new rocProject('hipBLAS', 'PreCheckin-CUDA')

    //customize for project
    prj.paths.build_command = buildCommand
    prj.libraryDependencies = ['hipBLAS-common']

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project, jobName)
    }

    def packageCommand =
    {
        platform, project->

        commonGroovy.runPackageCommand(platform, project, jobName, label)
    }
    def testCommand =
    {
        platform, project->

        commonGroovy.runTestCommand(platform, project)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

def setupCI(urlJobName, jobNameList, buildCommand, runCI, label)
{
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            runCI(nodeDetails, jobName, buildCommand, label)
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        runCI(['ubuntu20-cuda11':['anycuda']], urlJobName, buildCommand, label)
    }

}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = []
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = [:]

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    String hostBuildCommand = './install.sh -c --compiler=g++ --cuda'
    setupCI(urlJobName, jobNameList, hostBuildCommand, runCI, 'g++')
}
