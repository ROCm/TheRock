def rocmnode(name) {
    return '(rocmtest || miopen) && (' + name + ')'
}

def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME"
        lsb_release -sd
        uname -r
        cat /sys/module/amdgpu/version
        ls /opt/ -la
    """
}

// Given a pattern, check if the log contains the pattern and return the context.
def checkForPattern(pattern, log) {
    def lines = log.split('\n')
    for (int i = 0; i < lines.size(); i++) {
        if (lines[i] =~ pattern) {
            echo "Found pattern match in log for ${pattern}"
            
            // Get the two lines before and after failure.
            def contextStart = Math.max(0, i - 2)
            def contextEnd = Math.min(lines.size() - 1, i + 2)
            def contextLines = []
            for (int j = contextStart; j <= contextEnd; j++) {
                contextLines.add(lines[j])
            }
            
            return [found: true, matchedLine: lines[i], context: contextLines.join('\n')]
        }
    }
    echo "No pattern match found in log for ${pattern}"
    return [found: false, matchedLine: "", context: ""]
}

// Scan the build logs for failures and send notifications.
def sendFailureNotifications() {
    // Error patterns to scan build logs for specific failure types and send detailed notifications.
    def failurePatterns = [
        [pattern: /login attempt to .* failed with status: 401 Unauthorized/, description: "Docker registry authentication failed"],
        [pattern: /docker login failed/, description: "Docker login failed"],
        [pattern: /HTTP request sent .* 404 Not Found/, description: "HTTP request failed with 404"],
        [pattern: /cat: .* No such file or directory/, description: "GPU not found"],
        [pattern: /GPU not found/, description: "GPU not found"],
        [pattern: /Could not connect to Redis at .* Connection timed out/, description: "Redis connection timed out"]
    ]
    
    // Get the build log.
    def buildLog = sh(script: 'wget -q --no-check-certificate -O - ' + BUILD_URL + 'consoleText', returnStdout: true)
    // Check for patterns in the log.
    def foundPatterns = []
    for (patternMap in failurePatterns) {
        def result = checkForPattern(patternMap.pattern, buildLog)
        if (result.found) {
            foundPatterns.add([
                description: patternMap.description,
                matchedLine: result.matchedLine,
                context: result.context
            ])
        }
    }
    // Send a notification for each matched failure pattern.
    for (patternMap in foundPatterns) {
        withCredentials([string(credentialsId: 'ck_ci_errors_webhook_url', variable: 'WEBHOOK_URL')]) {
        sh '''
            curl -X POST "${WEBHOOK_URL}" \
            -H 'Content-Type: application/json' \
            -d '{"text": "\\n\\n**Build Failed**\\n\\n**Issues detected:** ''' + patternMap.description + '''\\n\\n**Log context:**\\n```\\n''' + patternMap.context.replace("'", "\\'") + '''\\n```\\n\\n**Job:** ''' + env.JOB_NAME + '''\\n\\n**Build:** #''' + env.BUILD_NUMBER + '''\\n\\n**URL:** ''' + env.RUN_DISPLAY_URL + '''"}'
        '''
        }
    }
}

def generateAndArchiveBuildTraceVisualization(String buildTraceFileName) {
    try {
        checkout scm

        // Retrieve the build trace artifact
        def traceFileExists = false
        try {
            copyArtifacts(
                projectName: env.JOB_NAME,
                selector: specific(env.BUILD_NUMBER),
                filter: buildTraceFileName
            )
            traceFileExists = fileExists(buildTraceFileName)
        } catch (Exception e) {
            echo "Could not copy build trace artifact: ${e.getMessage()}"
            traceFileExists = false
            return
        }
        
        sh """
            echo "post artifact download:"
            ls -la
        """
        
        // Pull image
        def image = "ghcr.io/puppeteer/puppeteer:24.30.0"
        echo "Pulling image: ${image}"
        def retimage = docker.image("${image}")
        retimage.pull()

        // Create a temporary workspace
        sh """#!/bin/bash
            ls -la
            mkdir -p workspace
            cp ./script/infra_helper/capture_build_trace.js ./workspace
            cp ${buildTraceFileName} ./workspace/${buildTraceFileName}
            chmod 777 ./workspace
            ls -la ./workspace
        """

        // Run container to get snapshot
        def dockerOpts = "--cap-add=SYS_ADMIN -v \"\$(pwd)/workspace:/workspace\" -e NODE_PATH=/home/pptruser/node_modules -e BUILD_TRACE_FILE=${buildTraceFileName}"
        // Create unique image name by sanitizing job name
        def sanitizedJobName = env.JOB_NAME.replaceAll(/[\/\\:*?"<>| ]/, '_')
        def architectureName = (buildTraceFileName =~ /(gfx[0-9a-zA-Z]+)/)[0][1]
        def imageName = "perfetto_snapshot_${sanitizedJobName}_build_${env.BUILD_NUMBER}_${architectureName}.png"
        sh """
            docker run --rm ${dockerOpts} ${image} node /workspace/capture_build_trace.js
            mv ./workspace/perfetto_snapshot_build.png ./workspace/${imageName}
        """
        
        // Archive the snapshot
        sh """
            mv ./workspace/${imageName} ${imageName}
        """
        archiveArtifacts "${imageName}"

        // Notify the channel
        withCredentials([string(credentialsId: 'ck_ci_build_perf_webhook_url', variable: 'WEBHOOK_URL')]) {
        sh '''
            # Create build trace filename with build number based on the original filename
            BUILD_TRACE_WITH_NUMBER=$(echo "''' + buildTraceFileName + '''" | sed 's/.json/_''' + sanitizedJobName + '''_''' + env.BUILD_NUMBER + '''_''' + architectureName + '''.json/')
            
            # Convert image to base64
            echo "Converting image to base64..."
            IMAGE_BASE64=$(base64 -w 0 ''' + imageName + ''')
            echo "Image base64 length: ${#IMAGE_BASE64}"
            
            # Convert build trace to base64
            echo "Converting build trace to base64..."
            BUILD_TRACE_BASE64=$(base64 -w 0 ''' + buildTraceFileName + ''')
            echo "Build trace base64 length: ${#BUILD_TRACE_BASE64}"
            
            # Create JSON payload with base64 data
            echo "Creating JSON payload..."
            {
                printf '{\n'
                printf '    "jobName": "%s",\n' "''' + env.JOB_NAME + '''"
                printf '    "buildNumber": "%s",\n' "''' + env.BUILD_NUMBER + '''"
                printf '    "jobUrl": "%s",\n' "''' + env.RUN_DISPLAY_URL + '''"
                printf '    "imageName": "%s",\n' "''' + imageName + '''"
                printf '    "architecture": "%s",\n' "''' + architectureName + '''"
                printf '    "imageData": "%s",\n' "$IMAGE_BASE64"
                printf '    "buildTraceName": "%s",\n' "$BUILD_TRACE_WITH_NUMBER"
                printf '    "buildTraceData": "%s"\n' "$BUILD_TRACE_BASE64"
                printf '}\n'
            } > webhook_payload.json
            
            echo "JSON payload created, size: $(wc -c < webhook_payload.json) bytes"
            
            curl -X POST "${WEBHOOK_URL}" \
            -H "Content-Type: application/json" \
            -d @webhook_payload.json
            
            # Clean up temporary file
            rm -f webhook_payload.json
        '''
        }
    } catch (Exception e) {
        echo "Throwing error exception while generating build trace visualization"
        echo 'Exception occurred: ' + e.toString()
    }
}

class Version {
    int major, minor, patch
    @Override
    String toString() {
        return [major, minor, patch].findAll().join('.')
    }
}
def parseVersion(String versionString) {
    if (!versionString) return null
    int[] tokens = versionString.split(/\./).collect { it as int } // Splits the string by '.' and converts each part to an integer.
    return new Version(
        major: tokens[0],
        minor: tokens.length > 1 ? tokens[1] : null,
        patch: tokens.length > 2 ? tokens[2] : null,
    )
}

def nthreads() {
    def nproc = sh(returnStdout: true, script: 'nproc')
    echo "Number of cores: ${nproc}"
    def n = nproc.toInteger()
    if (n > 64){
        n = 64
    }
    echo "Number of threads used for building: ${n}"
    return n
}

def runShell(String command){
    def responseCode = sh returnStatus: true, script: "${command} > tmp.txt"
    def output = readFile(file: "tmp.txt")
    return (output != "")
}

def shouldRunCICheck() {
    // Define patterns for files that should not trigger CI
    def skipFilePatterns = [
        /^\.github\/.*/, // GitHub workflow files
        /^docs\/.*/, // Documentation files
        /^LICENSE$/, // License file
        /^.*\.gitignore$/, // Git ignore files
        /.*\.md$/ // Markdown files
    ]
    
    try {
        // Get the list of changed files
        def changedFiles = sh(
            returnStdout: true,
            script: '''
                if [ "$CHANGE_ID" != "" ]; then
                    # For PR builds, compare against target branch
                    git diff --name-only origin/$CHANGE_TARGET...HEAD
                else
                    # For regular builds, compare against previous commit
                    git diff --name-only HEAD~1..HEAD
                fi
            '''
        ).trim().split('\n')
        
        if (changedFiles.size() == 1 && changedFiles[0] == '') {
            echo "No changed files detected - this might be a manual trigger or merge commit, running CI for safety"
            return true
        }
        
        echo "Changed files: ${changedFiles.join(', ')}"
        
        // Check if any changed files are not in the skip patterns
        def hasFilesRequiringCI = changedFiles.any { file ->
            !skipFilePatterns.any { pattern ->
                file ==~ pattern
            }
        }
        
        if (hasFilesRequiringCI) {
            echo "Found files that require CI"
            return true
        } else {
            echo "Only non-relevant files changed, skipping CI"
            return false
        } 
    } catch (Exception e) {
        echo "Error checking changed files: ${e.getMessage()}, running CI by default"
        return true
    }
}

def getBaseDockerImageName(){
    def img
    if (params.USE_CUSTOM_DOCKER != ""){
        img = "${params.USE_CUSTOM_DOCKER}"
    }
    else{
        def ROCM_numeric = parseVersion("${params.ROCMVERSION}")
        if ( ROCM_numeric.major <= 7 && ROCM_numeric.minor < 2 ){
            img = "${env.CK_DOCKERHUB}:ck_ub24.04_rocm${params.ROCMVERSION}"
            }
        else{
            img = "${env.CK_DOCKERHUB_PRIVATE}:ck_ub24.04_rocm${params.ROCMVERSION}"
            }
        }
    return img
}

def getDockerImageName(){
    def img
    def base_name = getBaseDockerImageName()
    if (params.USE_CUSTOM_DOCKER != ""){
        img = "${params.USE_CUSTOM_DOCKER}"
    }
    else{
       if (params.COMPILER_VERSION == "") {
           img = "${base_name}"
       }
       else{
          if (params.COMPILER_COMMIT == ""){
             img = "${base_name}_${params.COMPILER_VERSION}"
          }
          else{
             def commit = "${params.COMPILER_COMMIT}"[0..6]
             img = "${base_name}_${params.COMPILER_VERSION}_${commit}"
          }
       }
    }
    return img
}

def check_host() {
    if ("${env.CK_SCCACHE}" != "null"){
        def SCCACHE_SERVER="${env.CK_SCCACHE.split(':')[0]}"
        echo "sccache server: ${SCCACHE_SERVER}"
        sh "chmod +w -R ${env.WORKSPACE}"
        sh '''ping -c 1 -p 6379 "${SCCACHE_SERVER}" | echo $? > tmp.txt'''
        def output = readFile(file: "tmp.txt")
        echo "tmp.txt contents: \$output"
        return (output != "0")
    }
    else{
        return 1
    }
}

def check_arch_name(){
    def arch_name = ""
    sh 'rocminfo | tee rocminfo.log'
    if ( runShell('grep -n "gfx90a" rocminfo.log') ){
        arch_name = "gfx90a"
    }
    else if ( runShell('grep -n "gfx942" rocminfo.log') ) {
        arch_name = "gfx942"
    }
    else if ( runShell('grep -n "gfx101" rocminfo.log') ) {
        arch_name = "gfx101"
    }
    else if ( runShell('grep -n "gfx103" rocminfo.log') ) {
        arch_name = "gfx103"
    }
    else if ( runShell('grep -n "gfx11" rocminfo.log') ) {
        arch_name = "gfx11"
    }
    else if ( runShell('grep -n "gfx120" rocminfo.log') ) {
        arch_name = "gfx12"
    }
    else if ( runShell('grep -n "gfx908" rocminfo.log') ) {
        arch_name = "gfx908"
    }
    else if ( runShell('grep -n "gfx950" rocminfo.log') ) {
        arch_name = "gfx950"
    }
    return arch_name
}

def getDockerImage(Map conf=[:]){
    def image
    if ( conf.get("docker_name", "") != "" ){
        image = conf.get("docker_name", "")
        echo "Using special docker: ${image}"
    }
    else{
        image = getDockerImageName()
        echo "Using default docker: ${image}"
    }
    //Check if image exists
    def retimage
    try
    {
        echo "Pulling image: ${image}"
        retimage = docker.image("${image}")
        withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
            retimage.pull()
        }
    }
    catch(Exception ex)
    {
        error "Unable to locate image: ${image}"
    }
    return [retimage, image]
}

def buildDocker(install_prefix){
    show_node_info()
    env.DOCKER_BUILDKIT=1
    checkout scm
    def image_name = getDockerImageName()
    def base_image_name = getBaseDockerImageName()
    echo "Building Docker for ${image_name}"
    def dockerArgs = "--build-arg PREFIX=${install_prefix} --build-arg CK_SCCACHE='${env.CK_SCCACHE}' --build-arg compiler_version='${params.COMPILER_VERSION}' --build-arg compiler_commit='${params.COMPILER_COMMIT}' --build-arg ROCMVERSION='${params.ROCMVERSION}' "
    if(params.COMPILER_VERSION == "amd-staging" || params.COMPILER_VERSION == "amd-mainline" || params.COMPILER_COMMIT != ""){
        dockerArgs = dockerArgs + " --no-cache --build-arg BASE_DOCKER='${base_image_name}' -f Dockerfile.compiler . "
    }
    else if(params.RUN_AITER_TESTS){
        image_name = "${env.CK_DOCKERHUB_PRIVATE}:ck_aiter"
        dockerArgs = dockerArgs + " --no-cache -f Dockerfile.aiter --build-arg AITER_BRANCH='${params.aiter_branch}' --build-arg CK_AITER_BRANCH='${params.ck_aiter_branch}' . "
    }
     else if(params.RUN_PYTORCH_TESTS){
        image_name = "${env.CK_DOCKERHUB}:ck_pytorch"
        dockerArgs = dockerArgs + " --no-cache -f Dockerfile.pytorch --build-arg CK_PYTORCH_BRANCH='${params.ck_pytorch_branch}' . "
    }
   else{
        dockerArgs = dockerArgs + " -f Dockerfile . "
    }
    echo "Build Args: ${dockerArgs}"
    try{
        if(params.BUILD_DOCKER || params.RUN_AITER_TESTS || params.RUN_PYTORCH_TESTS){
            //force building the new docker if that parameter is true
            echo "Building image: ${image_name}"
            retimage = docker.build("${image_name}", dockerArgs)
            withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
                retimage.push()
            }
            sh 'docker images -q -f dangling=true | xargs --no-run-if-empty docker rmi'
        }
        else{
            echo "Checking for image: ${image_name}"
            sh "docker manifest inspect --insecure ${image_name}"
            echo "Image: ${image_name} found! Skipping building image"
        }
    }
    catch(Exception ex){
        echo "Unable to locate image: ${image_name}. Building image now"
        retimage = docker.build("${image_name}", dockerArgs)
        withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
            retimage.push()
        }
    }
}

def get_docker_options(){
    def dockerOpts
    if ( params.BUILD_INSTANCES_ONLY ){
        dockerOpts = "--network=host --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    }
    else{ //only add kfd and dri paths if you actually going to run somthing on GPUs
        dockerOpts = "--network=host --device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    }
    if (params.COMPILER_VERSION == "amd-staging" || params.COMPILER_VERSION == "amd-mainline" || params.COMPILER_COMMIT != ""){
    // the  --env COMPRESSED_BUNDLE_FORMAT_VERSION=2 env variable is required when building code with offload-compress flag with
    // newer clang22 compilers and running with older hip runtima libraries
        dockerOpts = dockerOpts + " --env HIP_CLANG_PATH='/llvm-project/build/bin' --env COMPRESSED_BUNDLE_FORMAT_VERSION=2 "
    }
    // on some machines the group ids for video and render groups may not be the same as in the docker image!
    def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3')
    def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3')
    dockerOpts = dockerOpts + " --group-add=${video_id} --group-add=${render_id} "
    echo "Docker flags: ${dockerOpts}"
    return dockerOpts
}

def build_client_examples(String arch){
    def cmd = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                -DGPU_TARGETS="${arch}" \
                -DCMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" \
                -DCMAKE_HIP_COMPILER="${params.BUILD_COMPILER}" \
                -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
    return cmd
}

def build_and_run_fmha(String arch){
    def cmd = """ cmake -G Ninja -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                -DGPU_TARGETS="${arch}" \
                -DCMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" \
                -DCMAKE_HIP_COMPILER="${params.BUILD_COMPILER}" .. && \
                ninja -j128 tile_example_fmha_fwd tile_example_fmha_bwd && \
                cd ../ &&
                example/ck_tile/01_fmha/script/run_full_test.sh "CI_${params.COMPILER_VERSION}" "${env.BRANCH_NAME}" "${NODE_NAME}" "${arch}" """
    return cmd
}

def cmake_build(Map conf=[:]){

    def config_targets = conf.get("config_targets","check")
    def build_envs = "CTEST_PARALLEL_LEVEL=4 " + conf.get("build_env","")
    def prefixpath = conf.get("prefixpath","/opt/rocm")
    def setup_args = conf.get("setup_args","")
    // make sure all unit tests always run on develop branch
    def runAllUnitTests = (env.BRANCH_NAME == "develop") ? true : params.RUN_ALL_UNIT_TESTS

    if (prefixpath != "/usr/local"){
        setup_args = setup_args + " -DCMAKE_PREFIX_PATH=${prefixpath} "
    }

    //cmake_env can overwrite default CXX variables.
    def cmake_envs = "CXX=${params.BUILD_COMPILER} CXXFLAGS='-Werror' " + conf.get("cmake_ex_env","")

    if(conf.get("build_install","") == "true")
    {
        config_targets = 'install ' + config_targets
        setup_args = ' -DBUILD_DEV=On -DCMAKE_INSTALL_PREFIX=../install' + setup_args
    } else{
        setup_args = ' -DBUILD_DEV=On' + setup_args
    }
    if (params.DISABLE_DL_KERNELS){
        setup_args = setup_args + " -DDISABLE_DL_KERNELS=ON "
    }

    setup_args = " -DCMAKE_BUILD_TYPE=release " + setup_args

    def pre_setup_cmd = """
            #!/bin/bash
            ulimit -c unlimited
            rm -rf build
            mkdir build
            rm -rf install
            mkdir install
            cd build
        """
    def invocation_tag=""
    if (setup_args.contains("gfx12")){
        invocation_tag="gfx12"
    }
    if (setup_args.contains("gfx11")){
        invocation_tag="gfx11"
    }
    if (setup_args.contains("gfx101")){
        invocation_tag="gfx101"
    }
    if (setup_args.contains("gfx103")){
        invocation_tag="gfx103"
    }
    if (setup_args.contains("gfx908")){
        invocation_tag="gfx908"
    }
    if (setup_args.contains("gfx90a")){
        invocation_tag="gfx90a"
    }
    if (setup_args.contains("gfx94")){
        invocation_tag="gfx94"
    }
    if (setup_args.contains("gfx95")){
        invocation_tag="gfx95"
    }
    echo "invocation tag: ${invocation_tag}"
    def redis_pre_setup_cmd = pre_setup_cmd
    if(check_host() && params.USE_SCCACHE && "${env.CK_SCCACHE}" != "null" && "${invocation_tag}" != "") {
        redis_pre_setup_cmd = pre_setup_cmd + """
            #!/bin/bash
            export ROCM_PATH=/opt/rocm
            export SCCACHE_ENABLED=true
            export SCCACHE_LOG_LEVEL=debug
            export SCCACHE_IDLE_TIMEOUT=14400
            export COMPILERS_HASH_DIR=/tmp/.sccache
            export SCCACHE_BIN=/usr/local/.cargo/bin/sccache
            export SCCACHE_EXTRAFILES=/tmp/.sccache/rocm_compilers_hash_file
            export SCCACHE_REDIS="redis://${env.CK_SCCACHE}"
            echo "connect = ${env.CK_SCCACHE}" >> ../script/redis-cli.conf
            export SCCACHE_C_CUSTOM_CACHE_BUSTER="${invocation_tag}"
            echo \$SCCACHE_C_CUSTOM_CACHE_BUSTER
            stunnel ../script/redis-cli.conf
            ../script/sccache_wrapper.sh --enforce_redis
        """
        try {
            def cmd1 = conf.get("cmd1", """
                    ${redis_pre_setup_cmd}
                """)
            sh cmd1
            setup_args = " -DCMAKE_HIP_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_C_COMPILER_LAUNCHER=sccache " + setup_args
        }
        catch(Exception err){
            echo "could not connect to redis server: ${err.getMessage()}. will not use sccache."
            def cmd2 = conf.get("cmd2", """
                    ${pre_setup_cmd}
                """)
            sh cmd2
        }
    }
    else{
        def cmd3 = conf.get("cmd3",  """
                ${pre_setup_cmd}
            """)
        sh cmd3
    }

    // reduce parallelism when compiling, clang uses too much memory
    def nt = nthreads()
    def cmd
    def setup_cmd
    def build_cmd
    def execute_cmd = conf.get("execute_cmd", "")
    if(!setup_args.contains("NO_CK_BUILD")){
        if (params.NINJA_BUILD_TRACE) {
            echo "running ninja build trace"
        }
        if (params.RUN_BUILDER_TESTS && !setup_args.contains("-DCK_CXX_STANDARD=") && !setup_args.contains("gfx10") && !setup_args.contains("gfx11")) {
            setup_args = " -D CK_EXPERIMENTAL_BUILDER=ON "  + setup_args
        }
        setup_cmd = conf.get(
            "setup_cmd",
            """${cmake_envs} cmake -G Ninja ${setup_args} -DCMAKE_CXX_FLAGS=" -O3 " .. """
        )
        build_cmd = conf.get(
            "build_cmd",
            "${build_envs} ninja -j${nt} ${config_targets}"
        )

        cmd = conf.get("cmd", """
            ${setup_cmd}
            ${build_cmd}
            ${execute_cmd}
        """)
    }
    else{
        cmd = conf.get("cmd", """
            ${execute_cmd}
        """)
    }

    echo cmd

    dir("build"){
        // Start sccache monitoring
        if(check_host() && params.USE_SCCACHE && "${env.CK_SCCACHE}" != "null" && "${invocation_tag}" != "") {
            sh """
                chmod +x ../script/monitor_sccache_during_build.sh
                mkdir -p logs
                export SCCACHE_C_CUSTOM_CACHE_BUSTER="${invocation_tag}"
                ../script/monitor_sccache_during_build.sh build_monitor &
                MONITOR_PID=\$!
                echo "Monitor PID: \$MONITOR_PID"
                echo \$MONITOR_PID > monitor.pid
            """
        }
        try {
            //build CK
            sh cmd
        } catch (Exception buildError) {
            echo "Build failed: ${buildError.getMessage()}"
            throw buildError
        } finally {
            // Stop sccache monitoring
            if(check_host() && params.USE_SCCACHE && "${env.CK_SCCACHE}" != "null" && "${invocation_tag}" != "") {
                sh """
                    # Stop monitoring
                    if [ -f monitor.pid ]; then
                        MONITOR_PID=\$(cat monitor.pid)
                        kill \$MONITOR_PID 2>/dev/null || echo "Monitor already stopped"
                        rm -f monitor.pid
                    fi
                """
                
                // Archive the monitoring logs
                try {
                    archiveArtifacts artifacts: "logs/*monitor*.log", allowEmptyArchive: true
                } catch (Exception e) {
                    echo "Could not archive sccache monitoring logs: ${e.getMessage()}"
                }
            }
        }

        //run tests except when NO_CK_BUILD or BUILD_LEGACY_OS are set
        if(!setup_args.contains("NO_CK_BUILD") && !params.BUILD_LEGACY_OS){
            sh "python3 ../script/ninja_json_converter.py .ninja_log --legacy-format --output ck_build_trace_${check_arch_name()}.json"
            archiveArtifacts "ck_build_trace_${check_arch_name()}.json"
            sh "python3 ../script/parse_ninja_trace.py ck_build_trace_${check_arch_name()}.json"
            if (params.NINJA_BUILD_TRACE || params.BUILD_INSTANCES_ONLY){
                if (params.NINJA_FTIME_TRACE) {
                    echo "running ClangBuildAnalyzer"
                    sh "/ClangBuildAnalyzer/build/ClangBuildAnalyzer  --all . clang_build.log"
                    sh "/ClangBuildAnalyzer/build/ClangBuildAnalyzer  --analyze clang_build.log > clang_build_analysis_${check_arch_name()}.log"
                    archiveArtifacts "clang_build_analysis_${check_arch_name()}.log"
                }


                // do not run unit tests when building instances only
                if(!params.BUILD_INSTANCES_ONLY){
                    if (!runAllUnitTests){
                        sh "../script/launch_tests.sh"
                    }
                    else{
                        sh "ninja check"
                    }
                    if (params.RUN_BUILDER_TESTS && !setup_args.contains("-DCK_CXX_STANDARD=") && !setup_args.contains("gfx10") && !setup_args.contains("gfx11")) {
                        sh 'ninja check-builder'
                    }
                    if(params.BUILD_PACKAGES){
                        echo "Build ckProfiler packages"
                        sh 'ninja -j64 package'
                        sh "mv composablekernel-ckprofiler_*.deb composablekernel-ckprofiler_1.2.0_amd64_${check_arch_name()}.deb"
                        stash includes: "composablekernel-ckprofiler**.deb", name: "profiler_package_${check_arch_name()}"
                    }
                }
                if(params.BUILD_INSTANCES_ONLY){
                    // build deb packages
                    echo "Build library package"
                    sh 'ninja -j64 package'
                    sh 'mv composablekernel-dev_*.deb composablekernel-dev_all_targets_1.2.0_amd64.deb'
                    stash includes: "composablekernel-dev**.deb", name: "lib_package"
                }
            }
            else{
                // run unit tests unless building library for all targets
                if (!params.BUILD_INSTANCES_ONLY){
                    if (!runAllUnitTests){
                        sh "../script/launch_tests.sh"
                    }
                    else{
                        sh "ninja check"
                    }
                    if (params.RUN_BUILDER_TESTS && !setup_args.contains("-DCK_CXX_STANDARD=") && !setup_args.contains("gfx10") && !setup_args.contains("gfx11")) {
                        sh 'ninja check-builder'
                    }
                    if(params.BUILD_PACKAGES){
                        echo "Build ckProfiler packages"
                        sh 'ninja -j64 package'
                        sh "mv composablekernel-ckprofiler_*.deb composablekernel-ckprofiler_1.2.0_amd64_${check_arch_name()}.deb"
                        stash includes: "composablekernel-ckprofiler**.deb", name: "profiler_package_${check_arch_name()}"
                    }
                }
            }
        }
    }

    //check the node gpu architecture
    def arch_name = check_arch_name()
    if (params.RUN_CK_TILE_FMHA_TESTS){
        try{
            archiveArtifacts "perf_fmha_*.log"
            stash includes: "perf_fmha_**.log", name: "perf_fmha_log_${arch_name}"
        }
        catch(Exception err){
            echo "could not locate the requested artifacts: ${err.getMessage()}. will skip the stashing."
        }
    }
}

def buildHipClangJob(Map conf=[:]){
        show_node_info()
        checkout scm
        def prefixpath = conf.get("prefixpath", "/opt/rocm")
        def dockerOpts = get_docker_options()
        def image
        def retimage
        (retimage, image) = getDockerImage(conf)

        gitStatusWrapper(credentialsId: "${env.ck_git_creds}", gitHubContext: "${env.STAGE_NAME}", account: 'ROCm', repo: 'composable_kernel') {
            withDockerContainer(image: image, args: dockerOpts) {
                timeout(time: 20, unit: 'HOURS')
                {
                    cmake_build(conf)
                }
            }
        }
        return retimage
}

def buildHipClangJobAndReboot(Map conf=[:]){
    try{
        buildHipClangJob(conf)
    }
    catch(e){
        echo "throwing error exception for the stage"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
}

def Build_CK(Map conf=[:]){
        show_node_info()
        checkout scm
        def prefixpath = conf.get("prefixpath", "/opt/rocm")
        def dockerOpts=get_docker_options()
        def image
        def retimage

        gitStatusWrapper(credentialsId: "${env.ck_git_creds}", gitHubContext: "${env.STAGE_NAME}", account: 'ROCm', repo: 'composable_kernel') {
            try {
                (retimage, image) = getDockerImage(conf)
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 2, unit: 'MINUTES'){
                        sh 'rocminfo | tee rocminfo.log'
                        if ( !runShell('grep -n "gfx" rocminfo.log') ){
                            throw new Exception ("GPU not found")
                        }
                        else{
                            echo "GPU is OK"
                        }
                    }
                }
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                throw e
            }
            withDockerContainer(image: image, args: dockerOpts) {
                timeout(time: 20, unit: 'HOURS')
                {
                    //check whether to run performance tests on this node
                    def arch = check_arch_name()
                    cmake_build(conf)
                    if ( params.RUN_INDUCTOR_TESTS && !params.BUILD_LEGACY_OS && arch == 1 ){
                            echo "Run inductor codegen tests"
                            sh """
                                  python3 -m venv ${env.WORKSPACE}
                                  . ${env.WORKSPACE}/bin/activate
                                  python3 -m pip install pytest build setuptools setuptools_scm
                                  python3 -m pip install .
                                  python3 -m pytest python/test/test_gen_instances.py
                            """
                    }
                    // run performance tests, stash the logs, results will be processed on the master node
					dir("script"){
                        if (params.RUN_PERFORMANCE_TESTS){
                        if (params.RUN_FULL_QA && (arch == "gfx90a" || arch == "gfx942")){
                            // run full tests on gfx90a or gfx942
                            echo "Run full performance tests"
                            sh "./run_full_performance_tests.sh 0 QA_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} ${arch}"
                            archiveArtifacts "perf_*.log"
                            stash includes: "perf_**.log", name: "perf_log_${arch}"
                        }
                        else if (!params.RUN_FULL_QA && (arch == "gfx90a" || arch == "gfx942")){
                            // run standard tests on gfx90a or gfx942
                            echo "Run performance tests"
                            sh "./run_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} ${arch}"
                            archiveArtifacts "perf_*.log"
                            stash includes: "perf_**.log", name: "perf_log_${arch}"
                        }
                        // disable performance tests on gfx1030 for now.
                        //else if ( arch == "gfx10"){
                            // run basic tests on gfx1030
                        //    echo "Run gemm performance tests"
                        //    sh "./run_gemm_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} gfx10"
                        //    archiveArtifacts "perf_onnx_gemm_gfx10.log"
                        //    stash includes: "perf_onnx_gemm_gfx10.log", name: "perf_log_gfx10"
                        //}
                        else if ( arch == "gfx11"){
                            // run basic tests on gfx11
                            echo "Run gemm performance tests"
                            sh "./run_gemm_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} gfx11"
                            archiveArtifacts "perf_onnx_gemm_gfx11.log"
                            stash includes: "perf_onnx_gemm_gfx11.log", name: "perf_log_gfx11"
                        }
                        else if ( arch == "gfx120" ){
                            // run basic tests on gfx12
                            echo "Run gemm performance tests"
                            sh "./run_gemm_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} gfx12"
                            archiveArtifacts "perf_onnx_gemm_gfx12.log"
                            stash includes: "perf_onnx_gemm_gfx12.log", name: "perf_log_gfx12"
                        }
                        else if ( arch == "gfx908" ){
                            // run basic tests on gfx908
                            echo "Run performance tests"
                            sh "./run_gemm_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} gfx908"
                            archiveArtifacts "perf_onnx_gemm_gfx908.log"
                            stash includes: "perf_onnx_gemm_gfx908.log", name: "perf_log_gfx908"
                        }
                        else if ( arch == "gfx950" ){
                            // run basic tests on gfx950
                            echo "Run performance tests"
                            sh "./run_gemm_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} gfx950"
                            archiveArtifacts "perf_onnx_gemm_gfx950.log"
                            stash includes: "perf_onnx_gemm_gfx950.log", name: "perf_log_gfx950"
                        }
                        }
                    }
                    if (params.hipTensor_test && arch == "gfx90a" ){
                        // build and test hipTensor on gfx90a node
                        sh """#!/bin/bash
                            rm -rf rocm-libraries
                            git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
                            cd rocm-libraries
                            git sparse-checkout init --cone
                            git sparse-checkout set projects/hiptensor
                            git checkout "${params.hipTensor_branch}"
                        """
                        dir("rocm-libraries/projects/hiptensor"){
                            sh """#!/bin/bash
                                mkdir -p build
                                ls -ltr
                                CC=hipcc CXX=hipcc cmake -Bbuild . -D CMAKE_PREFIX_PATH="${env.WORKSPACE}/install"
                                cmake --build build -- -j
                                ctest --test-dir build
                            """
                        }
                    }
                }
            }
        }
        return retimage
}

def Build_CK_and_Reboot(Map conf=[:]){
    try{
        Build_CK(conf)
    }
    catch(e){
        echo "throwing error exception while building CK"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
}

def process_results(Map conf=[:]){
    checkout scm
    //use older image that has user jenkins
    def image = "${env.CK_DOCKERHUB}:ck_ub22.04_rocm6.3"

    gitStatusWrapper(credentialsId: "${env.ck_git_creds}", gitHubContext: "${env.STAGE_NAME}", account: 'ROCm', repo: 'composable_kernel') {
        try
        {
            echo "Pulling image: ${image}"
            def retimage = docker.image("${image}")
            withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
                retimage.pull()
            }
        }
        catch(Exception ex)
        {
            error "Unable to locate image: ${image}"
        }
    }

    withDockerContainer(image: image, args: '--cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v=/var/jenkins/:/var/jenkins') {
        timeout(time: 15, unit: 'MINUTES'){
            try{
                dir("script"){
                    if (params.RUN_CK_TILE_FMHA_TESTS){
                        try{
                            unstash "perf_fmha_log_gfx942"
                        }
                        catch(Exception err){
                            echo "could not locate the FMHA performance logs for gfx942: ${err.getMessage()}."
                        }
                        try{
                            unstash "perf_fmha_log_gfx90a"
                        }
                        catch(Exception err){
                            echo "could not locate the FMHA performance logs for gfx90a: ${err.getMessage()}."
                        }
                    }
                    if (params.BUILD_INSTANCES_ONLY){
                        // unstash deb packages
                        try{
                            unstash "lib_package"
                        }
                        catch(Exception err){
                            echo "could not locate lib_package."
                        }
                        sh "sshpass -p ${env.ck_deb_pw} scp -o StrictHostKeyChecking=no composablekernel-*.deb ${env.ck_deb_user}@${env.ck_deb_ip}:/var/www/html/composable_kernel/"
                    }
                    if (params.BUILD_PACKAGES){
                        // unstash deb packages
                        try{
                            unstash "profiler_package_gfx90a"
                        }
                        catch(Exception err){
                            echo "could not locate profiler_package_gfx90a."
                        }
                        try{
                            unstash "profiler_package_gfx942"
                        }
                        catch(Exception err){
                            echo "could not locate profiler_package_gfx942."
                        }
                        try{
                            unstash "profiler_package_gfx950"
                        }
                        catch(Exception err){
                            echo "could not locate profiler_package_gfx950."
                        }
                        try{
                            unstash "profiler_package_gfx12"
                        }
                        catch(Exception err){
                            echo "could not locate profiler_package_gfx12."
                        }
                        sh "sshpass -p ${env.ck_deb_pw} scp -o StrictHostKeyChecking=no composablekernel-ckprofiler*.deb ${env.ck_deb_user}@${env.ck_deb_ip}:/var/www/html/composable_kernel/"
                    }
                    else{
                        // unstash perf files to master
                        try{
                            unstash "perf_log_gfx90a"
                        }
                        catch(Exception err){
                            echo "could not locate the gfx90a performance logs: ${err.getMessage()}."
                        }
                        try{
                            unstash "perf_log_gfx942"
                        }
                        catch(Exception err){
                            echo "could not locate the gfx942 performance logs: ${err.getMessage()}."
                        }
                        try{
                            unstash "perf_log_gfx950"
                        }
                        catch(Exception err){
                            echo "could not locate the gfx950 performance logs: ${err.getMessage()}."
                        }
                        try{
                            unstash "perf_log_gfx908"
                        }
                        catch(Exception err){
                            echo "could not locate the gfx908 performance logs: ${err.getMessage()}."
                        }
                        try{
                            unstash "perf_log_gfx11"
                        }
                        catch(Exception err){
                            echo "could not locate the gfx11 performance logs: ${err.getMessage()}."
                        }
                        try{

                            unstash "perf_log_gfx12"
                        }
                        catch(Exception err){
                            echo "could not locate the gfx12 performance logs: ${err.getMessage()}."
                        }
                    }
                    // process the logs
                    sh "./process_perf_data.sh"
                }
            }
            catch(e){
                echo "Throwing error exception while processing performance test results"
                echo 'Exception occurred: ' + e.toString()
                throw e
            }
            finally{
                echo "Finished processing performance test results"
            }
        }
    }
}

def run_aiter_tests(Map conf=[:]){
    show_node_info()
    checkout scm
    //use the latest pytorch image
    def image = "${env.CK_DOCKERHUB_PRIVATE}:ck_aiter"
    def dockerOpts=get_docker_options() + ' --group-add irc '

    gitStatusWrapper(credentialsId: "${env.ck_git_creds}", gitHubContext: "${env.STAGE_NAME}", account: 'ROCm', repo: 'composable_kernel') {
        try
        {
            echo "Pulling image: ${image}"
            retimage = docker.image("${image}")
            withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
                retimage.pull()
            }
        }
        catch(Exception ex)
        {
            error "Unable to locate image: ${image}"
        }
    }

    withDockerContainer(image: image, args: dockerOpts) {
        timeout(time: 5, unit: 'HOURS'){
            try{
                sh "rocminfo"
                sh "python3 --version"
                sh "python3 /home/jenkins/workspace/aiter/op_tests/test_gemm_a8w8.py"
                sh "python3 /home/jenkins/workspace/aiter/op_tests/test_gemm_a8w8_blockscale.py"
                sh "python3 /home/jenkins/workspace/aiter/op_tests/test_mha.py"
                sh "python3 /home/jenkins/workspace/aiter/op_tests/test_mha_varlen.py"
                sh "python3 /home/jenkins/workspace/aiter/op_tests/test_moe.py"
                sh "python3 /home/jenkins/workspace/aiter/op_tests/test_moe_2stage.py"
                sh "python3 /home/jenkins/workspace/aiter/op_tests/test_moe_blockscale.py"
                sh "python3 /home/jenkins/workspace/aiter/op_tests/test_moe_ep.py"
                sh "python3 /home/jenkins/workspace/aiter/op_tests/test_moe_sorting.py"
                sh "python3 /home/jenkins/workspace/aiter/op_tests/test_moe_sorting_mxfp4.py"
                sh "python3 /home/jenkins/workspace/aiter/op_tests/test_moe_tkw1.py"
            }
            catch(e){
                echo "Throwing error exception while running AITER tests"
                echo 'Exception occurred: ' + e.toString()
                throw e
            }
            finally{
                echo "Finished running AITER tests"
            }
        }
    }
}


def run_pytorch_tests(Map conf=[:]){
    show_node_info()
    checkout scm
    //use the latest pytorch-nightly image
    def image = "${env.CK_DOCKERHUB}:ck_pytorch"
    def dockerOpts=get_docker_options() + ' --group-add irc '

    gitStatusWrapper(credentialsId: "${env.ck_git_creds}", gitHubContext: "${env.STAGE_NAME}", account: 'ROCm', repo: 'composable_kernel') {
        try
        {
            echo "Pulling image: ${image}"
            retimage = docker.image("${image}")
            withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
                retimage.pull()
            }
        }
        catch(Exception ex)
        {
            error "Unable to locate image: ${image}"
        }
    }

    withDockerContainer(image: image, args: dockerOpts) {
        timeout(time: 2, unit: 'HOURS'){
            try{
                sh "rocminfo"
                sh "python3 --version"
                sh "python3 /tmp/pytorch/tools/amd_build/build_amd.py"
                sh "USE_ROCM_CK_SDPA=1 PYTORCH_ROCM_ARCH=gfx942 python /tmp/pytorch/setup.py develop"
            }
            catch(e){
                echo "Throwing error exception while building Pytorch"
                echo 'Exception occurred: ' + e.toString()
                throw e
            }
            finally{
                echo "Finished building Pytorch"
            }
        }
    }
}

//launch develop branch daily jobs
CRON_SETTINGS = BRANCH_NAME == "develop" ? '''0 23 * * * % RUN_FULL_QA=true;RUN_CK_TILE_FMHA_TESTS=true;RUN_PERFORMANCE_TESTS=true;FORCE_CI=true
                                              0 22 * * * % RUN_FULL_QA=true;DISABLE_DL_KERNELS=true;RUN_TILE_ENGINE_BASIC_TESTS=true;RUN_TILE_ENGINE_GEMM_TESTS=true;RUN_PERFORMANCE_TESTS=true;RUN_ALL_UNIT_TESTS=true;FORCE_CI=true
                                              0 21 * * * % RUN_GROUPED_CONV_LARGE_CASES_TESTS=true;hipTensor_test=true;BUILD_GFX101=false;BUILD_GFX908=false;BUILD_GFX942=true;BUILD_GFX950=true;RUN_PERFORMANCE_TESTS=true;RUN_ALL_UNIT_TESTS=true;FORCE_CI=true;BUILD_PACKAGES=true
                                              0 19 * * * % BUILD_DOCKER=true;COMPILER_VERSION=amd-staging;BUILD_COMPILER=/llvm-project/build/bin/clang++;USE_SCCACHE=false;NINJA_BUILD_TRACE=true;RUN_ALL_UNIT_TESTS=true;FORCE_CI=true
                                              0 17 * * * % BUILD_DOCKER=true;COMPILER_VERSION=amd-mainline;BUILD_COMPILER=/llvm-project/build/bin/clang++;USE_SCCACHE=false;NINJA_BUILD_TRACE=true;RUN_ALL_UNIT_TESTS=true;FORCE_CI=true
                                              0 15 * * * % BUILD_INSTANCES_ONLY=true;USE_SCCACHE=false;NINJA_BUILD_TRACE=true;FORCE_CI=true
                                              0 13 * * * % RUN_AITER_TESTS=true;BUILD_LEGACY_OS=true;USE_SCCACHE=false;RUN_PERFORMANCE_TESTS=false;FORCE_CI=true
                                              0 11 * * * % RUN_PYTORCH_TESTS=true;RUN_CODEGEN_TESTS=false;USE_SCCACHE=false;RUN_PERFORMANCE_TESTS=false;BUILD_GFX101=false;BUILD_GFX103=false;BUILD_GFX11=false;BUILD_GFX12=false;BUILD_GFX90A=false;FORCE_CI=true''' : ""

pipeline {
    agent none
    triggers {
        parameterizedCron(CRON_SETTINGS)
    }
    options {
        parallelsAlwaysFailFast()
    }
    parameters {
        booleanParam(
            name: "BUILD_DOCKER",
            defaultValue: false,
            description: "Force building docker image (default: false), set to true if docker image needs to be updated.")
        string(
            name: 'USE_CUSTOM_DOCKER',
            defaultValue: '',
            description: 'If you want to use a custom docker image, please specify it here (default: leave blank).')
        string(
            name: 'ROCMVERSION',
            defaultValue: '7.1.1',
            description: 'Specify which ROCM version to use: 7.1.1 (default).')
        string(
            name: 'COMPILER_VERSION',
            defaultValue: '',
            description: 'Specify which version of compiler to use: release, amd-staging, amd-mainline, or leave blank (default).')
        string(
            name: 'COMPILER_COMMIT',
            defaultValue: '',
            description: 'Specify which commit of compiler branch to use: leave blank to use the latest commit (default), or use some specific commit of llvm-project branch.')
        string(
            name: 'BUILD_COMPILER',
            defaultValue: '/opt/rocm/llvm/bin/clang++',
            description: 'Build CK with /opt/rocm/bin/hipcc, /llvm-project/build/bin/clang++, or with /opt/rocm/llvm/bin/clang++ (default).')
        booleanParam(
            name: "RUN_FULL_QA",
            defaultValue: false,
            description: "Select whether to run small set of performance tests (default) or full QA")
        booleanParam(
            name: "DISABLE_DL_KERNELS",
            defaultValue: false,
            description: "Select whether to build DL kernels (default: OFF)")
        booleanParam(
            name: "hipTensor_test",
            defaultValue: false,
            description: "Use the CK build to verify hipTensor build and tests (default: OFF)")
        string(
            name: 'hipTensor_branch',
            defaultValue: 'develop',
            description: 'Specify which branch of hipTensor to use (default: develop)')
        booleanParam(
            name: "USE_SCCACHE",
            defaultValue: true,
            description: "Use the sccache for building CK (default: ON)")
        booleanParam(
            name: "RUN_CPPCHECK",
            defaultValue: false,
            description: "Run the cppcheck static analysis (default: OFF)")
        booleanParam(
            name: "RUN_PERFORMANCE_TESTS",
            defaultValue: false,
            description: "Run the performance tests (default: OFF)")
        booleanParam(
            name: "RUN_GROUPED_CONV_LARGE_CASES_TESTS",
            defaultValue: false,
            description: "Run the grouped conv large cases tests (default: OFF)")
        booleanParam(
            name: "RUN_CONV_COMPREHENSIVE_DATASET",
            defaultValue: false,
            description: "Run comprehensive convolution dataset tests before important changes (default: OFF)")
        booleanParam(
            name: "RUN_CODEGEN_TESTS",
            defaultValue: true,
            description: "Run codegen tests (default: ON)")
        booleanParam(
            name: "RUN_CK_TILE_FMHA_TESTS",
            defaultValue: false,
            description: "Run the ck_tile FMHA tests (default: OFF)")
        booleanParam(
            name: "RUN_TILE_ENGINE_BASIC_TESTS",
            defaultValue: false,
            description: "Run the tile_engine_basic tests (default: OFF)")
        booleanParam(
            name: "RUN_TILE_ENGINE_GEMM_TESTS",
            defaultValue: false,
            description: "Run the tile_engine_gemm tests (default: OFF)")
        booleanParam(
            name: "BUILD_INSTANCES_ONLY",
            defaultValue: false,
            description: "Test building instances for various architectures simultaneously (default: OFF)")
        booleanParam(
            name: "BUILD_PACKAGES",
            defaultValue: false,
            description: "Build packages for the libraries and/or ckProfiler (default: OFF)")
        booleanParam(
            name: "BUILD_GFX908",
            defaultValue: false,
            description: "Build CK and run tests on gfx908 (default: OFF)")
        booleanParam(
            name: "BUILD_GFX90A",
            defaultValue: true,
            description: "Build CK and run tests on gfx90a (default: ON)")
        booleanParam(
            name: "BUILD_GFX942",
            defaultValue: true,
            description: "Build CK and run tests on gfx942 (default: ON)")
        booleanParam(
            name: "BUILD_GFX950",
            defaultValue: true,
            description: "Build CK and run tests on gfx950 (default: ON)")
        booleanParam(
            name: "BUILD_GFX101",
            defaultValue: false,
            description: "Build CK and run tests on gfx101 (default: OFF)")
        booleanParam(
            name: "BUILD_GFX103",
            defaultValue: true,
            description: "Build CK and run tests on gfx103 (default: ON)")
        booleanParam(
            name: "BUILD_GFX11",
            defaultValue: true,
            description: "Build CK and run tests on gfx11 (default: ON)")
        booleanParam(
            name: "BUILD_GFX12",
            defaultValue: true,
            description: "Build CK and run tests on gfx12 (default: ON)")
        booleanParam(
            name: "NINJA_BUILD_TRACE",
            defaultValue: false,
            description: "Generate a ninja build trace (default: OFF)")
        booleanParam(
            name: "NINJA_FTIME_TRACE",
            defaultValue: false,
            description: "Generate a detailed time trace (default: OFF)")
        booleanParam(
            name: "BUILD_LEGACY_OS",
            defaultValue: false,
            description: "Try building CK with legacy OS dockers: RHEL8 and SLES15 (default: OFF)")
        booleanParam(
            name: "RUN_INDUCTOR_TESTS",
            defaultValue: true,
            description: "Run inductor codegen tests (default: ON)")
        booleanParam(
            name: "RUN_BUILDER_TESTS",
            defaultValue: true,
            description: "Run CK_BUILDER tests (default: ON)")
        booleanParam(
            name: "RUN_ALL_UNIT_TESTS",
            defaultValue: false,
            description: "Run all unit tests (default: OFF)")
        booleanParam(
            name: "RUN_PYTORCH_TESTS",
            defaultValue: false,
            description: "Try building PYTORCH with latest CK develop branch (default: OFF)")
        string(
            name: 'ck_pytorch_branch',
            defaultValue: 'develop',
            description: 'Specify which branch of CK to test with Pytorch (default: develop)')
        booleanParam(
            name: "RUN_AITER_TESTS",
            defaultValue: false,
            description: "Run AITER tests with latest CK develop branch (default: OFF)")
        string(
            name: 'aiter_branch',
            defaultValue: 'main',
            description: 'Specify which branch of AITER to use (default: main)')
        string(
            name: 'ck_aiter_branch',
            defaultValue: 'develop',
            description: 'Specify which branch of CK to test with AITER (default: develop)')
        booleanParam(
            name: "FORCE_CI",
            defaultValue: false,
            description: "Force CI to run even when only non-relevant files are changed (default: OFF)")
    }
    environment{
        dbuser = "${dbuser}"
        dbpassword = "${dbpassword}"
        dbsship = "${dbsship}"
        dbsshport = "${dbsshport}"
        dbsshuser = "${dbsshuser}"
        dbsshpassword = "${dbsshpassword}"
        ck_git_creds = "${ck_git_creds}"
        gerrit_cred="${gerrit_cred}"
        DOCKER_BUILDKIT = "1"
    }
    stages{
        stage("Determine CI Execution") {
            agent{ label rocmnode("nogpu") }
            steps {
                script {
                    env.SHOULD_RUN_CI = String.valueOf(params.FORCE_CI.toBoolean() || shouldRunCICheck())
                    echo "SHOULD_RUN_CI: ${env.SHOULD_RUN_CI}"
                }
            }
        }
        stage("Build Docker"){
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel{
                stage('Docker /opt/rocm'){
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildDocker('/opt/rocm')
                        cleanWs()
                    }
                }
            }
        }
        stage("Static checks") {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel{
                stage('Clang Format and Cppcheck') {
                    when {
                        beforeAgent true
                        expression { params.RUN_CPPCHECK.toBoolean() }
                    }
                    agent{ label rocmnode("nogpu") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_cmd = "(cd .. && git ls-files \'*.h\' \
                                \'*.hpp\' \
                                \'*.cpp\' \
                                \'*.h.in\' \
                                \'*.hpp.in\' \
                                \'*.cpp.in\' \
                                \'*.cl\' \
                                | grep -v 'build/' \
                                | grep -v 'include/rapidjson' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-18 -style=file {} | diff - {}\') && \
                                /cppcheck/build/bin/cppcheck ../* -v -j \$(nproc) -I ../include -I ../profiler/include -I ../library/include \
                                -D CK_ENABLE_FP64 -D CK_ENABLE_FP32 -D CK_ENABLE_FP16 -D CK_ENABLE_FP8 -D CK_ENABLE_BF16 -D CK_ENABLE_BF8 -D CK_ENABLE_INT8 \
                                -D __gfx908__ -D __gfx90a__ -D __gfx942__ -D __gfx1030__ -D __gfx1100__ -D __gfx1101__ -D __gfx1102__ \
                                -U __gfx803__ -U __gfx900__ -U __gfx906__ -U CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4 \
                                --file-filter=*.cpp --force --enable=all --output-file=ck_cppcheck.log"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd)
                        archiveArtifacts "build/ck_cppcheck.log"
                        cleanWs()
                    }
                }
                stage('Clang Format') {
                    when {
                        beforeAgent true
                        expression { !params.RUN_CPPCHECK.toBoolean() }
                    }
                    agent{ label rocmnode("nogpu") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_cmd = "(cd .. && git ls-files \
                                \'*.h\' \
                                \'*.hpp\' \
                                \'*.cpp\' \
                                \'*.h.in\' \
                                \'*.hpp.in\' \
                                \'*.cpp.in\' \
                                \'*.cl\' \
                                | grep -v 'build/' \
                                | grep -v 'include/rapidjson' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-18 -style=file {} | diff - {}\')"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd)
                        cleanWs()
                    }
                }
            }
        }
         stage("Run Pytorch Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run Pytorch Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_PYTORCH_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942")}
                    steps{
                        run_pytorch_tests()
                        cleanWs()
                    }
                }
            }
        }
        stage("Run AITER Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run AITER Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_AITER_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942")}
                    steps{
                        run_aiter_tests()
                        cleanWs()
                    }
                }
                stage("Run AITER Tests on gfx950")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_AITER_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx950")}
                    steps{
                        run_aiter_tests()
                        cleanWs()
                    }
                }
            }
        }
        stage("Run Grouped Conv Large Case Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run Grouped Conv Large Case Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_GROUPED_CONV_LARGE_CASES_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a")}
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ ../script/cmake-ck-dev.sh  ../ gfx90a && \
                                           make -j64 test_grouped_convnd_fwd_large_cases test_grouped_convnd_bwd_data_large_cases test_grouped_convnd_fwd_bias_clamp_large_cases && \
                                           ./bin/test_grouped_convnd_fwd_large_cases && ./bin/test_grouped_convnd_bwd_data_large_cases && ./bin/test_grouped_convnd_fwd_bias_clamp_large_cases"""
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }
        stage("Run Comprehensive Convolution Dataset Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run Comprehensive Dataset Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CONV_COMPREHENSIVE_DATASET.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a")}
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ cd ../build && \
                                           ../script/cmake-ck-dev.sh  ../ gfx90a && \
                                           make -j64 test_grouped_convnd_fwd_dataset_xdl \
                                           test_grouped_convnd_bwd_data_dataset_xdl \
                                           test_grouped_convnd_bwd_weight_dataset_xdl && \
                                           cd ../test_data && \
                                           # Dataset generation modes:
                                           # - small: ~60 test cases (minimal, quick testing - 3 models, 2 batch sizes, 2 image sizes)
                                           # - half: ~300 test cases (moderate coverage - 16 models, 3 batch sizes, 5 image sizes), ~ 17 hours testing time
                                           # - full: ~600 test cases (comprehensive - 16 models, 5 batch sizes, 9 image sizes), ~ 40 hours testing time
                                           ./generate_test_dataset.sh small && \
                                           cd ../build && \
                                           ./bin/test_grouped_convnd_fwd_dataset_xdl && \
                                           ./bin/test_grouped_convnd_bwd_data_dataset_xdl && \
                                           ./bin/test_grouped_convnd_bwd_weight_dataset_xdl"""
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }
        stage("Run Codegen Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run Codegen Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CODEGEN_TESTS.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a")}
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" ../codegen && \
                                           make -j64 check"""
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }
        stage("Run CK_TILE_FMHA Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run CK_TILE_FMHA Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CK_TILE_FMHA_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = build_and_run_fmha("gfx90a")
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
                stage("Run CK_TILE_FMHA Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CK_TILE_FMHA_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = build_and_run_fmha("gfx942")
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
                stage("Run CK_TILE_FMHA Tests on gfx950")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CK_TILE_FMHA_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx950") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = build_and_run_fmha("gfx950")
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
                stage("Run CK_TILE_FMHA Tests on gfx1201")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CK_TILE_FMHA_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1201") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = build_and_run_fmha("gfx1201")
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }
        stage("Run TILE_ENGINE_BASIC Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run TILE_ENGINE_BASIC Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_TILE_ENGINE_BASIC_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                                            -D CMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" \
                                            -D CMAKE_BUILD_TYPE=Release \
                                            -D GPU_TARGETS="gfx942" \
                                            -D GEMM_UNIVERSAL_DATATYPE="fp8;fp16" \
                                            -D GEMM_UNIVERSAL_LAYOUT="rcr;rrr;crr;ccr" \
                                            -D GEMM_UNIVERSAL_CONFIG_FILE="default_ci_config.json" \
                                            -D GEMM_MULTI_D_DATATYPE="fp16" \
                                            -D GEMM_MULTI_D_LAYOUT="rcrr;rrrr;crrr;ccrr" \
                                            -D GEMM_MULTI_D_CONFIG_FILE="default_ci_config.json" \
                                            -D GEMM_PRESHUFFLE_DATATYPE="fp16;fp8;bf16;bf8" \
                                            -D GEMM_PRESHUFFLE_LAYOUT="rcr" \
                                            -D GEMM_PRESHUFFLE_CONFIG_FILE="default_ci_config.json" .. && \
                                           ninja -j${nthreads()} benchmark_gemm_universal_all benchmark_gemm_preshuffle_all benchmark_gemm_multi_d_all """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }
        stage("Run TILE_ENGINE_GEMM Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run TILE_ENGINE_GEMM Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_TILE_ENGINE_GEMM_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                                            -D CMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" \
                                            -D CMAKE_BUILD_TYPE=Release \
                                            -D GPU_TARGETS="gfx90a" \
                                            -D GEMM_UNIVERSAL_DATATYPE="fp8;fp16" \
                                            -D GEMM_UNIVERSAL_LAYOUT="rcr;rrr;crr;ccr" \
                                            -D GEMM_STREAMK_DATATYPE="fp8;fp16" \
                                            -D GEMM_STREAMK_LAYOUT="rcr" \
                                            -D GEMM_MULTI_D_DATATYPE="fp16" \
                                            -D GEMM_MULTI_D_LAYOUT="rcrr;rrrr;crrr;ccrr" \
                                            -D GEMM_PRESHUFFLE_DATATYPE="fp16;fp8;bf16;bf8" \
                                            -D GEMM_PRESHUFFLE_LAYOUT="rcr" .. && \
                                           ninja -j${nthreads()} benchmark_gemm_universal_all benchmark_gemm_preshuffle_all benchmark_gemm_multi_d_all benchmark_gemm_streamk_all && \
                                           python3 ../tile_engine/ops/gemm/gemm_universal/gemm_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json && \
                                           python3 ../tile_engine/ops/gemm/gemm_preshuffle/gemm_preshuffle_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json && \
                                           python3 ../tile_engine/ops/gemm/gemm_multi_d/gemm_multi_d_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
                stage("Run TILE_ENGINE_GEMM Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_TILE_ENGINE_GEMM_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                                            -D CMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" \
                                            -D CMAKE_BUILD_TYPE=Release \
                                            -D GPU_TARGETS="gfx942" \
                                            -D GEMM_UNIVERSAL_DATATYPE="fp8;fp16" \
                                            -D GEMM_UNIVERSAL_LAYOUT="rcr;rrr;crr;ccr" \
                                            -D GEMM_STREAMK_DATATYPE="fp8;fp16" \
                                            -D GEMM_STREAMK_LAYOUT="rcr" \
                                            -D GEMM_MULTI_D_DATATYPE="fp16" \
                                            -D GEMM_MULTI_D_LAYOUT="rcrr;rrrr;crrr;ccrr" \
                                            -D GEMM_PRESHUFFLE_DATATYPE="fp16;fp8;bf16;bf8" \
                                            -D GEMM_PRESHUFFLE_LAYOUT="rcr" .. && \
                                           ninja -j${nthreads()} benchmark_gemm_universal_all benchmark_gemm_preshuffle_all benchmark_gemm_multi_d_all benchmark_gemm_streamk_all && \
                                           python3 ../tile_engine/ops/gemm/gemm_universal/gemm_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json && \
                                           python3 ../tile_engine/ops/gemm/gemm_preshuffle/gemm_preshuffle_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json && \
                                           python3 ../tile_engine/ops/gemm/gemm_multi_d/gemm_multi_d_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
                stage("Run TILE_ENGINE_GEMM Tests on gfx1201")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_TILE_ENGINE_GEMM_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1201") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                                            -D CMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" \
                                            -D CMAKE_BUILD_TYPE=Release \
                                            -D GPU_TARGETS="gfx1201" \
                                            -D GEMM_UNIVERSAL_DATATYPE="fp16" \
                                            -D GEMM_UNIVERSAL_LAYOUT="rcr;rrr;crr;ccr" .. && \
                                           ninja -j${nthreads()} benchmark_gemm_universal_all && \
                                           python3 ../tile_engine/ops/gemm/gemm_universal/gemm_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }

		stage("Build CK and run Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Build CK with RHEL8")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        setup_args = """ -DGPU_TARGETS="gfx942" -DCK_CXX_STANDARD="17" -DCK_USE_ALTERNATIVE_PYTHON=/opt/Python-3.8.13/bin/python3.8 """
                        execute_args = " "
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: " ", build_type: 'Release', docker_name: "${env.CK_DOCKERHUB_PRIVATE}:ck_rhel8_rocm6.3")
                        cleanWs()
                    }
                }
                stage("Build CK with SLES15")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        setup_args = """ -DGPU_TARGETS="gfx942" -DCK_USE_ALTERNATIVE_PYTHON=/opt/Python-3.8.13/bin/python3.8 """
                        execute_args = " "
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: " ", build_type: 'Release', docker_name: "${env.CK_DOCKERHUB_PRIVATE}:ck_sles15_rocm6.3")
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { (params.BUILD_GFX942.toBoolean() || params.RUN_FULL_QA.toBoolean()) && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx942" """
                        execute_args = build_client_examples("gfx942")
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx950")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX950.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx950") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx950" """
                        execute_args = build_client_examples("gfx950")
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx908")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX908.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx908") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx908" """
                        execute_args = build_client_examples("gfx908")
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX90A.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx90a" -DCK_CXX_STANDARD="17" """
                        execute_args = build_client_examples("gfx90a")
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK instances for all supported targets")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_INSTANCES_ONLY.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    steps{
                        script {
                            def execute_args = """ cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                                                -DCMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" \
                                                -DCMAKE_HIP_COMPILER="${params.BUILD_COMPILER}" \
                                                -D CMAKE_BUILD_TYPE=Release .. && ninja -j64 """

                            buildHipClangJobAndReboot(setup_cmd: "",  build_cmd: "", build_type: 'Release', execute_cmd: execute_args)
                        }
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx1010")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX101.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1010") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx10-1-generic" """
                        execute_args = build_client_examples("gfx10-1-generic")
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx1030")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX103.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1030") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx10-3-generic" """
                        execute_args = build_client_examples("gfx10-3-generic")
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx11")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX11.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label 'miopen && (gfx1101 || gfx1100)' }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx11-generic" """
                        execute_args = build_client_examples("gfx11-generic")
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx1201")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX12.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1201") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx12-generic" """
                        execute_args = build_client_examples("gfx12-generic")
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
            }
            post {
                always {
                    node(rocmnode("nogpu")) {
                        script {
                            // Simulate capture
                            generateAndArchiveBuildTraceVisualization("ck_build_trace_gfx11.json")
                            generateAndArchiveBuildTraceVisualization("ck_build_trace_gfx12.json")
                            generateAndArchiveBuildTraceVisualization("ck_build_trace_gfx90a.json")
                            generateAndArchiveBuildTraceVisualization("ck_build_trace_gfx942.json")
                            generateAndArchiveBuildTraceVisualization("ck_build_trace_gfx950.json")
                        }
                        cleanWs()
                    }
                }
                success {
                    script {
                        // Report the parent stage build ck and run tests status
                        gitStatusWrapper(credentialsId: "${env.ck_git_creds}", gitHubContext: "${env.STAGE_NAME}", account: 'ROCm', repo: 'composable_kernel') {
                            echo "Reporting success status for build ck and run tests"
                        }
                    }
                }
            }
        }
        stage("Process Performance Test Results")
        {
            parallel
            {
                stage("Process results"){
                    when {
                        beforeAgent true
                        expression { (params.RUN_PERFORMANCE_TESTS.toBoolean() || params.BUILD_INSTANCES_ONLY.toBoolean() || params.RUN_CK_TILE_FMHA_TESTS.toBoolean()|| params.BUILD_PACKAGES.toBoolean()) && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent { label 'mici' }
                    steps{
                        process_results()
                        cleanWs()
                    }
                }
            }
            post {
                success {
                    script {
                        // Report the skipped parent's stage status
                        gitStatusWrapper(credentialsId: "${env.ck_git_creds}", gitHubContext: "Process Performance Test Results", account: 'ROCm', repo: 'composable_kernel') {
                            echo "Process Performance Test Results stage skipped."
                        }
                        // Report the skipped stage's status
                        gitStatusWrapper(credentialsId: "${env.ck_git_creds}", gitHubContext: "Process results", account: 'ROCm', repo: 'composable_kernel') {
                            echo "Process Performance Test Results stage skipped."
                        }
                    }
                }
            }
        }
    }
    post {
        failure {
            node(rocmnode("nogpu")) {
                script {
                    sendFailureNotifications()
                }
            }
        }
    }
}
