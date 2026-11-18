// Opens R3Con again and again with the pointer already resting on its window's top row, for
// .github/workflows/mac-app.yml: swift tests/app/launch_pointer_check.swift APP LOG_DIR
// As after Sparkle's Install and Relaunch, the pointer is where the window opens, so AppKit hit-tests
// the band over the top row as soon as it joins the window, before the page has said where its
// controls are: that crashed 0.6.36 at launch. It fails if the app dies before it is asked to quit, or
// if its log shows a Python method raising into AppKit, for which ctypes hands AppKit garbage. The
// first launch only finds where the window opens. Where the runner drops synthetic events, it warns
// that the check is untested. Each launch's log is LOG_DIR/launch-N.log.
import AppKit
import CoreGraphics
import Foundation

let app = URL(fileURLWithPath: CommandLine.arguments[1])
let logs = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
let executable = app.appendingPathComponent("Contents/MacOS/R3Con")
// After the one that finds where the window opens.
let launches = 5
// In the top row, past the traffic lights and the brand, on the status: as in window_drag_check.swift.
let restAt = CGPoint(x: 160, y: 14)
let pointerTolerance: CGFloat = 1
let pollSeconds = 0.25
// Up to this long for the window and its page, which then asks the server for its keys...
let openSeconds = 90.0
let pageLoaded = "\"GET /api/keys HTTP/1.1\" 200"
// ...then this long more for the window's first look at the page's controls, and at least this long
// after the launch, the pointer still on the top row.
let settleSeconds = 5.0
let restSeconds = 10.0
let quitSeconds = 20.0
// The start of what Python logs for an exception raised in a method that AppKit called.
let raisedIntoAppKit = "ctypes callback function"
let logTailLines = 60

// Line by line, so that the job's log shows each launch as it goes.
setvbuf(stdout, nil, _IOLBF, 0)

func finish(_ level: String, _ message: String, failing: Bool = false) -> Never {
    print("::\(level)::\(message)")
    exit(failing ? 1 : 0)
}

func windowFrame(of pid: pid_t) -> CGRect? {
    let windows = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] ?? []
    let window = windows.first {
        ($0[kCGWindowOwnerPID as String] as? pid_t) == pid && ($0[kCGWindowLayer as String] as? Int) == 0
    }
    guard let bounds = window?[kCGWindowBounds as String] as? NSDictionary else { return nil }
    return CGRect(dictionaryRepresentation: bounds as CFDictionary)
}

func rest(at point: CGPoint) {
    let move = CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: .left)
    move?.post(tap: .cghidEventTap)
}

func pointer() -> CGPoint {
    CGEvent(source: nil)?.location ?? .zero
}

func text(of url: URL) -> String {
    (try? Data(contentsOf: url)).map { String(decoding: $0, as: UTF8.self) } ?? ""
}

// Sleeps in the run loop, where Process and NSRunningApplication learn of changes.
func sleepInRunLoop(_ seconds: TimeInterval) {
    RunLoop.current.run(until: Date().addingTimeInterval(seconds))
}

func ending(of process: Process) -> String {
    let how = process.terminationReason == .uncaughtSignal ? "signal" : "status"
    return "\(how) \(process.terminationStatus)"
}

func fail(_ message: String, _ process: Process, _ log: URL) -> Never {
    if process.isRunning {
        kill(process.processIdentifier, SIGKILL)
    }
    print("The last \(logTailLines) lines of \(log.path):")
    for line in text(of: log).split(separator: "\n", omittingEmptySubsequences: false).suffix(logTailLines) {
        print(line)
    }
    finish("error", message, failing: true)
}

func launch(_ number: Int) -> (Process, URL) {
    let log = logs.appendingPathComponent("launch-\(number).log")
    let process = Process()
    process.executableURL = executable
    // As mac-app.yml's first launch: INFO, and each line the app logs also on its stderr.
    var environment = ProcessInfo.processInfo.environment
    environment["R3CON_LOG_LEVEL"] = "INFO"
    environment["OS_ACTIVITY_DT_MODE"] = "YES"
    process.environment = environment
    do {
        try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let output = try FileHandle(forWritingTo: log)
        process.standardOutput = output
        process.standardError = output
        try process.run()
    } catch {
        finish("error", "R3Con didn't start from \(executable.path): \(error)", failing: true)
    }
    return (process, log)
}

var target: CGPoint?
var rested = 0
for number in 0...launches {
    if let target {
        rest(at: target)
        let at = pointer()
        if abs(at.x - target.x) > pointerTolerance || abs(at.y - target.y) > pointerTolerance {
            finish("warning", "This runner drops synthetic mouse events (the pointer stayed at \(at)): the launch is untested.")
        }
    }
    let (process, log) = launch(number)
    let started = Date()
    var frame: CGRect?
    var loaded: Date?
    while true {
        sleepInRunLoop(pollSeconds)
        let elapsed = Date().timeIntervalSince(started)
        if !process.isRunning {
            let pointed = target.map { " with the pointer at \($0)" } ?? ""
            fail("R3Con died \(Int(elapsed)) s into launch \(number)\(pointed), by \(ending(of: process)).", process, log)
        }
        if let target {
            rest(at: target)
        }
        if frame == nil, let opened = windowFrame(of: process.processIdentifier) {
            frame = opened
            let active = NSRunningApplication(processIdentifier: process.processIdentifier)?.isActive ?? false
            print("Launch \(number): the window opened at \(opened) after \(Int(elapsed)) s, the pointer at \(pointer()); the app is active: \(active).")
        }
        if loaded == nil, text(of: log).contains(pageLoaded) {
            loaded = Date()
        }
        if frame != nil, let loaded, Date().timeIntervalSince(loaded) >= settleSeconds, elapsed >= restSeconds {
            break
        }
        if elapsed >= openSeconds {
            fail("Launch \(number) showed no window, or no page that asked for its keys, within \(Int(openSeconds)) s.", process, log)
        }
    }
    if let target, let frame {
        if frame.contains(target) {
            rested += 1
        } else {
            print("::warning::Launch \(number)'s window opened at \(frame), away from the pointer at \(target).")
        }
    }
    if let frame {
        target = CGPoint(x: frame.minX + restAt.x, y: frame.minY + restAt.y)
    }

    _ = NSRunningApplication(processIdentifier: process.processIdentifier)?.terminate()
    let asked = Date()
    while process.isRunning && Date().timeIntervalSince(asked) < quitSeconds {
        sleepInRunLoop(pollSeconds)
    }
    if process.isRunning {
        fail("Launch \(number) didn't end within \(Int(quitSeconds)) s of a quit.", process, log)
    }
    if process.terminationReason == .uncaughtSignal {
        fail("R3Con died on a quit, by \(ending(of: process)).", process, log)
    }
    let raised = text(of: log).split(separator: "\n").filter { $0.contains(raisedIntoAppKit) }
    if !raised.isEmpty {
        fail("Launch \(number) raised into AppKit: \(raised.joined(separator: " | "))", process, log)
    }
    print("Launch \(number) lived until its quit, then ended by \(ending(of: process)).")
}

if rested < launches {
    finish("warning", "Only \(rested) of \(launches) launches opened under the pointer: the rest are untested.")
}
print("R3Con opened \(launches) times with the pointer resting on its top row, and lived until each quit.")
