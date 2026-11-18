// Opens R3Con again and again with the pointer already resting on its window's top row, for
// .github/workflows/mac-app.yml: swift tests/app/launch_pointer_check.swift APP LOG_DIR
// As after Sparkle's Install and Relaunch, the pointer is where the window opens, so AppKit hit-tests
// the band over the top row as soon as it joins the window, before the page has said where its
// controls are: that crashed 0.6.36 at launch. AppKit does that only for the app in front, so each
// launch is asked to come forward as soon as its window shows, and if it hasn't a moment later the
// check clicks where the pointer rests, which macOS always lets activate it. It fails if the app isn't
// active when its page loads, as that launch would prove nothing; if it dies before it is asked to
// quit; or if its log shows a Python method raising into AppKit, for which ctypes hands AppKit garbage.
// The first launch also finds where the window opens, for the pointer to wait there before the rest.
// Where the runner drops synthetic events, it warns that the check is untested. Each launch's log is
// LOG_DIR/launch-N.log.
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
// How long after its window shows a launch may take to come forward before the check clicks on it.
let activateSeconds = 0.25
let clickPauseSeconds = 0.05
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

// Sleeps in the run loop, where Process and NSRunningApplication learn of changes.
func sleepInRunLoop(_ seconds: TimeInterval) {
    RunLoop.current.run(until: Date().addingTimeInterval(seconds))
}

func windowFrame(of pid: pid_t) -> CGRect? {
    let windows = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] ?? []
    let window = windows.first {
        ($0[kCGWindowOwnerPID as String] as? pid_t) == pid && ($0[kCGWindowLayer as String] as? Int) == 0
    }
    guard let bounds = window?[kCGWindowBounds as String] as? NSDictionary else { return nil }
    return CGRect(dictionaryRepresentation: bounds as CFDictionary)
}

func topRow(of window: CGRect) -> CGPoint {
    CGPoint(x: window.minX + restAt.x, y: window.minY + restAt.y)
}

func rest(at point: CGPoint) {
    let move = CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: .left)
    move?.post(tap: .cghidEventTap)
}

func click(at point: CGPoint) {
    for phase in [CGEventType.leftMouseDown, CGEventType.leftMouseUp] {
        let event = CGEvent(mouseEventSource: nil, mouseType: phase, mouseCursorPosition: point, mouseButton: .left)
        event?.setIntegerValueField(.mouseEventClickState, value: 1)
        event?.post(tap: .cghidEventTap)
        sleepInRunLoop(clickPauseSeconds)
    }
}

func pointer() -> CGPoint {
    CGEvent(source: nil)?.location ?? .zero
}

func isActive(_ pid: pid_t) -> Bool {
    NSRunningApplication(processIdentifier: pid)?.isActive ?? false
}

func text(of url: URL) -> String {
    (try? Data(contentsOf: url)).map { String(decoding: $0, as: UTF8.self) } ?? ""
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

// Where the pointer waits before each launch: the top row of the window the launch before opened.
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
    let pid = process.processIdentifier
    let started = Date()
    var frame: CGRect?
    // Where the pointer rests once the window shows, and when it showed.
    var spot: CGPoint?
    var shown: Date?
    var clicked = false
    var loaded: Date?
    while true {
        sleepInRunLoop(pollSeconds)
        let elapsed = Date().timeIntervalSince(started)
        if !process.isRunning {
            let pointed = (spot ?? target).map { " with the pointer at \($0)" } ?? ""
            fail("R3Con died \(Int(elapsed)) s into launch \(number)\(pointed), by \(ending(of: process)).", process, log)
        }
        if frame == nil, let opened = windowFrame(of: pid) {
            frame = opened
            shown = Date()
            spot = target ?? topRow(of: opened)
            // Without ignoringOtherApps, which has done nothing since macOS 14.
            _ = NSRunningApplication(processIdentifier: pid)?.activate(options: [])
            print("Launch \(number): the window opened at \(opened) after \(Int(elapsed)) s, the pointer at \(pointer()).")
        }
        if let resting = spot ?? target {
            rest(at: resting)
        }
        let active = isActive(pid)
        if !active, !clicked, let shown, let spot, Date().timeIntervalSince(shown) >= activateSeconds {
            click(at: spot)
            clicked = true
            print("Launch \(number) didn't come forward when asked: the check clicked its top row at \(spot).")
        }
        if loaded == nil, text(of: log).contains(pageLoaded) {
            loaded = Date()
            if !active {
                fail("Launch \(number) wasn't the active app when its page loaded, so AppKit may not have hit-tested its window: it proves nothing.", process, log)
            }
            print("Launch \(number): its page loaded after \(Int(elapsed)) s, the app active, the pointer at \(pointer()).")
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
        target = topRow(of: frame)
    }

    _ = NSRunningApplication(processIdentifier: pid)?.terminate()
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
print("R3Con opened \(launches) times with the pointer resting on its top row, active as its page loaded, and lived until each quit.")
