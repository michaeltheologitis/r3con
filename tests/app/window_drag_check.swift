// Drags R3Con's window by its top row, then double-clicks there, with synthetic mouse events, for
// .github/workflows/mac-app.yml: swift tests/app/window_drag_check.swift PID
// It fails if the window doesn't follow the drag, or doesn't zoom where a title bar double-click zooms;
// where the runner drops synthetic events, it warns that the drag is untested. The screen may stop the
// window short of the whole drag, so it need only follow it, rightwards, by a clear share of it.
import CoreGraphics
import Foundation

let pid = pid_t(CommandLine.arguments[1])!
// In the top row, past the traffic lights and the brand, on the status: where a press drags.
let pressAt = CGPoint(x: 160, y: 14)
let dragBy = CGPoint(x: 120, y: 80)
// About a real mouse's: a move every 16 ms.
let dragSteps = 20
let dragStepSeconds = 0.016
let followedShare: CGFloat = 0.25
let pointerTolerance: CGFloat = 1
let eventPauseSeconds = 0.05
// For the page's first answer of where its controls are, which the drag needs.
let controlsWaitSeconds = 2.0
// As a hand holds a press before it moves.
let pressHoldSeconds = 0.3
let dragSettleSeconds = 0.5
let zoomSettleSeconds = 1.0
// The user's choice of what double-clicking a title bar does, its default, and two that don't zoom.
let doubleClickAction = "AppleActionOnDoubleClick"
let maximizeAction = "Maximize"
let minimizeAction = "Minimize"
let noAction = "None"

func windowFrame() -> CGRect? {
    let windows = CGWindowListCopyWindowInfo(.optionOnScreenOnly, kCGNullWindowID) as? [[String: Any]] ?? []
    let window = windows.first {
        ($0[kCGWindowOwnerPID as String] as? pid_t) == pid && ($0[kCGWindowLayer as String] as? Int) == 0
    }
    guard let bounds = window?[kCGWindowBounds as String] as? NSDictionary else { return nil }
    return CGRect(dictionaryRepresentation: bounds as CFDictionary)
}

func post(_ type: CGEventType, at point: CGPoint, clicks: Int64 = 1, pause: TimeInterval = eventPauseSeconds) {
    let event = CGEvent(mouseEventSource: nil, mouseType: type, mouseCursorPosition: point, mouseButton: .left)!
    event.setIntegerValueField(.mouseEventClickState, value: clicks)
    event.post(tap: .cghidEventTap)
    Thread.sleep(forTimeInterval: pause)
}

func finish(_ level: String, _ message: String, failing: Bool = false) -> Never {
    print("::\(level)::\(message)")
    exit(failing ? 1 : 0)
}

Thread.sleep(forTimeInterval: controlsWaitSeconds)
guard let before = windowFrame() else { finish("error", "R3Con has no window on screen.", failing: true) }
let start = CGPoint(x: before.minX + pressAt.x, y: before.minY + pressAt.y)
post(.mouseMoved, at: start)
let pointer = CGEvent(source: nil)?.location ?? .zero
if abs(pointer.x - start.x) > pointerTolerance || abs(pointer.y - start.y) > pointerTolerance {
    finish("warning", "This runner drops synthetic mouse events (the pointer stayed at \(pointer)): the drag is untested.")
}

post(.leftMouseDown, at: start)
Thread.sleep(forTimeInterval: pressHoldSeconds)
for step in 1...dragSteps {
    let share = CGFloat(step) / CGFloat(dragSteps)
    post(.leftMouseDragged, at: CGPoint(x: start.x + dragBy.x * share, y: start.y + dragBy.y * share), pause: dragStepSeconds)
}
post(.leftMouseUp, at: CGPoint(x: start.x + dragBy.x, y: start.y + dragBy.y))
Thread.sleep(forTimeInterval: dragSettleSeconds)
guard let after = windowFrame() else { finish("error", "R3Con's window is gone after a drag.", failing: true) }
let moved = CGPoint(x: after.minX - before.minX, y: after.minY - before.minY)
print("A drag of \(dragBy) on the window's top row moved it by \(moved), from \(before) to \(after).")
if moved.x < dragBy.x * followedShare {
    finish("error", "The window didn't follow a drag of \(dragBy) on its top row: it moved by \(moved).", failing: true)
}

let action = UserDefaults.standard.string(forKey: doubleClickAction) ?? maximizeAction
if action == minimizeAction || action == noAction {
    finish("notice", "A title bar double-click here is set to \(action): the zoom is untested.")
}
let again = CGPoint(x: after.minX + pressAt.x, y: after.minY + pressAt.y)
for clicks: Int64 in [1, 2] {
    post(.leftMouseDown, at: again, clicks: clicks)
    post(.leftMouseUp, at: again, clicks: clicks)
}
Thread.sleep(forTimeInterval: zoomSettleSeconds)
guard let zoomed = windowFrame(), zoomed != after else {
    finish("error", "A double-click on the window's top row left it at \(after).", failing: true)
}
print("A double-click on the window's top row zoomed it from \(after) to \(zoomed).")
