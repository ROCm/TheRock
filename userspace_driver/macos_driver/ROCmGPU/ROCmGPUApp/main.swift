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
 *       ai.rocm.gpu.driver.dext
 *
 * First install requires user approval in:
 *   System Settings > General > Login Items & Extensions > Driver Extensions
 */

import Foundation
import SystemExtensions

let dextBundleID = "ai.rocm.gpu.driver"

class ExtensionDelegate: NSObject, OSSystemExtensionRequestDelegate {
    func request(
        _ request: OSSystemExtensionRequest,
        didFinishWithResult result: OSSystemExtensionRequest.Result
    ) {
        switch result {
        case .completed:
            print("DEXT activation completed successfully.")
            exit(0)
        case .willCompleteAfterReboot:
            print("DEXT will activate after reboot.")
            exit(0)
        @unknown default:
            print("Unknown result: \(result)")
            exit(0)
        }
    }

    func request(
        _ request: OSSystemExtensionRequest,
        didFailWithError error: Error
    ) {
        print("DEXT activation failed: \(error.localizedDescription)")
        let ns = error as NSError
        print("  domain: \(ns.domain)")
        print("  code:   \(ns.code)")
        print("  userInfo: \(ns.userInfo)")
        exit(1)
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
    // Diagnostics: confirm we're finding the app bundle + DEXT
    print("Bundle.main.bundleURL: \(Bundle.main.bundleURL.path)")
    let sysExtDir = Bundle.main.bundleURL
        .appendingPathComponent("Contents/Library/SystemExtensions")
    print("Looking for DEXTs in: \(sysExtDir.path)")
    if let entries = try? FileManager.default.contentsOfDirectory(atPath: sysExtDir.path) {
        print("  entries: \(entries)")
        for entry in entries where entry.hasSuffix(".dext") {
            let plist = sysExtDir.appendingPathComponent(entry)
                .appendingPathComponent("Info.plist")
            if let data = try? Data(contentsOf: plist),
               let dict = try? PropertyListSerialization.propertyList(from: data, options: [], format: nil) as? [String: Any] {
                print("  \(entry) CFBundleIdentifier: \(dict["CFBundleIdentifier"] ?? "???")")
            }
        }
    } else {
        print("  (directory unreadable)")
    }
    print("Requesting DEXT activation for: \(dextBundleID)")
    let delegate = ExtensionDelegate()
    // Use a background queue for callbacks so the main RunLoop can pump
    // messages from OSSystemExtensionManager. Delegate exits the process
    // directly on completion/failure.
    let request = OSSystemExtensionRequest.activationRequest(
        forExtensionWithIdentifier: dextBundleID,
        queue: DispatchQueue.global(qos: .userInitiated)
    )
    request.delegate = delegate
    OSSystemExtensionManager.shared.submitRequest(request)

    // Safety timeout: exit if nothing happens in 120s
    DispatchQueue.global().asyncAfter(deadline: .now() + 120) {
        print("Timed out waiting for DEXT activation.")
        print("Check System Settings > General > Login Items & Extensions > Driver Extensions")
        exit(1)
    }

    // Pump the main RunLoop so activation can progress. Delegate calls
    // exit() directly when done (success or failure).
    RunLoop.main.run()
}

func uninstall() {
    print("Requesting DEXT deactivation for: \(dextBundleID)")
    let delegate = ExtensionDelegate()
    let request = OSSystemExtensionRequest.deactivationRequest(
        forExtensionWithIdentifier: dextBundleID,
        queue: DispatchQueue.global(qos: .userInitiated)
    )
    request.delegate = delegate
    OSSystemExtensionManager.shared.submitRequest(request)

    DispatchQueue.global().asyncAfter(deadline: .now() + 60) {
        print("Timed out waiting for DEXT deactivation.")
        exit(1)
    }

    RunLoop.main.run()
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

// Xcode's Run action auto-injects flag args like -NSDocumentRevisionsDebugMode.
// Skip those and take the first non-flag positional arg as the command.
// Default to "install" when none is present (typical dev inner loop).
let positionalArgs = CommandLine.arguments.dropFirst().filter { !$0.hasPrefix("-") }
let command = positionalArgs.first ?? "install"

switch command {
case "install":
    install()
case "uninstall":
    uninstall()
case "status":
    status()
default:
    print("Usage: ROCmGPUApp [install|uninstall|status]")
    print("  install   — Activate the ROCmGPU DriverKit extension (default)")
    print("  uninstall — Deactivate the DEXT")
    print("  status    — Check if DEXT is loaded")
    exit(1)
}
