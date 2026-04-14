/*
 * ROCmGPU Host App — Installs and manages the ROCmGPU DriverKit extension.
 *
 * Usage:
 *   ROCmGPUApp install    — Request DEXT activation (prompts user approval)
 *   ROCmGPUApp uninstall  — Deactivate the DEXT
 *   ROCmGPUApp status     — Check if DEXT is loaded
 *
 * The DEXT is embedded at:
 *   ROCmGPUApp.app/Contents/Library/SystemExtensions/
 *       com.rocm.gpu.driver.dext
 *
 * First install requires user approval in:
 *   System Settings > General > Login Items & Extensions > Driver Extensions
 */

import Foundation
import SystemExtensions

let dextBundleID = "com.rocm.gpu.driver"

class ExtensionDelegate: NSObject, OSSystemExtensionRequestDelegate {
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<Void, Error> = .success(())

    func request(
        _ request: OSSystemExtensionRequest,
        didFinishWithResult result: OSSystemExtensionRequest.Result
    ) {
        switch result {
        case .completed:
            print("DEXT activation completed successfully.")
            self.result = .success(())
        case .willCompleteAfterReboot:
            print("DEXT will activate after reboot.")
            self.result = .success(())
        @unknown default:
            print("Unknown result: \(result)")
            self.result = .success(())
        }
        semaphore.signal()
    }

    func request(
        _ request: OSSystemExtensionRequest,
        didFailWithError error: Error
    ) {
        print("DEXT activation failed: \(error.localizedDescription)")
        self.result = .failure(error)
        semaphore.signal()
    }

    func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
        print("""
            User approval required!
            Go to: System Settings > General > Login Items & Extensions
                   > Driver Extensions
            Enable "ROCmGPU" and try again.
            """)
    }

    func request(
        _ request: OSSystemExtensionRequest,
        actionForReplacingExtension existing: OSSystemExtensionProperties,
        withExtension ext: OSSystemExtensionProperties
    ) -> OSSystemExtensionRequest.ReplacementAction {
        print("Replacing existing DEXT (v\(existing.bundleShortVersion) -> v\(ext.bundleShortVersion))")
        return .replace
    }
}

func install() {
    print("Requesting DEXT activation for: \(dextBundleID)")
    let delegate = ExtensionDelegate()
    let request = OSSystemExtensionRequest.activationRequest(
        forExtensionWithIdentifier: dextBundleID,
        queue: .main
    )
    request.delegate = delegate
    OSSystemExtensionManager.shared.submitRequest(request)

    // Wait for completion (with timeout)
    let timeout = delegate.semaphore.wait(timeout: .now() + 60)
    if timeout == .timedOut {
        print("Timed out waiting for DEXT activation.")
        print("Check System Settings for pending approval.")
        exit(1)
    }

    switch delegate.result {
    case .success:
        exit(0)
    case .failure:
        exit(1)
    }
}

func uninstall() {
    print("Requesting DEXT deactivation for: \(dextBundleID)")
    let delegate = ExtensionDelegate()
    let request = OSSystemExtensionRequest.deactivationRequest(
        forExtensionWithIdentifier: dextBundleID,
        queue: .main
    )
    request.delegate = delegate
    OSSystemExtensionManager.shared.submitRequest(request)

    let timeout = delegate.semaphore.wait(timeout: .now() + 30)
    if timeout == .timedOut {
        print("Timed out waiting for DEXT deactivation.")
        exit(1)
    }

    switch delegate.result {
    case .success:
        exit(0)
    case .failure:
        exit(1)
    }
}

func status() {
    // Check if our DEXT is loaded by looking for it in IOKit registry
    print("Checking DEXT status...")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/sbin/systemextensionsctl")
    process.arguments = ["list"]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe

    do {
        try process.run()
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""

        if output.contains(dextBundleID) {
            print("ROCmGPU DEXT is registered.")
            // Check for [activated enabled] status
            for line in output.components(separatedBy: "\n") {
                if line.contains(dextBundleID) {
                    print("  \(line.trimmingCharacters(in: .whitespaces))")
                }
            }
        } else {
            print("ROCmGPU DEXT is NOT registered.")
            print("Run 'ROCmGPUApp install' to activate.")
        }
    } catch {
        print("Failed to check status: \(error)")
    }
}

// --- Main ---

let args = CommandLine.arguments
let command = args.count > 1 ? args[1] : "install"

switch command {
case "install":
    install()
case "uninstall":
    uninstall()
case "status":
    status()
default:
    print("Usage: ROCmGPUApp [install|uninstall|status]")
    print("  install   — Activate the ROCmGPU DriverKit extension")
    print("  uninstall — Deactivate the DEXT")
    print("  status    — Check if DEXT is loaded")
    exit(1)
}
