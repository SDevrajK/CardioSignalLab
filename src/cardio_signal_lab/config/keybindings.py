"""Keyboard shortcut definitions for CardioSignalLab.

Defines default keybindings for all application actions using Qt Key constants.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence

# Define keybindings using Qt key constants for cross-platform compatibility
KEYBINDINGS = {
    # File operations
    "file_open": QKeySequence.StandardKey.Open,  # Ctrl+O
    "file_save": QKeySequence.StandardKey.Save,  # Ctrl+S
    "file_export": QKeySequence(Qt.Key.Key_Control | Qt.Key.Key_E),  # Ctrl+E
    "file_quit": QKeySequence.StandardKey.Quit,  # Ctrl+Q

    # Edit operations (peak correction)
    "edit_undo": QKeySequence.StandardKey.Undo,  # Ctrl+Z
    "edit_redo": QKeySequence.StandardKey.Redo,  # Ctrl+Y
    "edit_add_peak": QKeySequence(Qt.Key.Key_Insert),  # Insert
    "edit_delete_peak": QKeySequence(Qt.Key.Key_Delete),  # Delete

    # View operations
    "view_zoom_in": QKeySequence("Ctrl+="),  # Ctrl+= (Ctrl++ requires Shift on most keyboards)
    "view_zoom_out": QKeySequence.StandardKey.ZoomOut,  # Ctrl+-
    "view_reset": QKeySequence(Qt.Key.Key_R),  # R
    "view_fit": QKeySequence(Qt.Key.Key_F),  # F
    "view_multi_signal": QKeySequence(Qt.Key.Key_Escape),  # ESC
    "view_jump_to_time": QKeySequence(Qt.Key.Key_Control | Qt.Key.Key_T),  # Ctrl+T
    "view_pan_mode": QKeySequence(Qt.Key.Key_P),  # P
    "view_zoom_mode": QKeySequence(Qt.Key.Key_Z),  # Z

    # Peak navigation
    "peak_next": QKeySequence(Qt.Key.Key_Right),  # Right arrow
    "peak_previous": QKeySequence(Qt.Key.Key_Left),  # Left arrow
    "peak_first": QKeySequence(Qt.Key.Key_Home),  # Home
    "peak_last": QKeySequence(Qt.Key.Key_End),  # End

    # Signal navigation
    "signal_pan_left": QKeySequence(Qt.Key.Key_Shift | Qt.Key.Key_Left),  # Shift+Left
    "signal_pan_right": QKeySequence(Qt.Key.Key_Shift | Qt.Key.Key_Right),  # Shift+Right
    "signal_page_forward": QKeySequence(Qt.Key.Key_PageDown),  # Page Down
    "signal_page_backward": QKeySequence(Qt.Key.Key_PageUp),  # Page Up

    # Help
    "help_show": QKeySequence.StandardKey.HelpContents,  # F1
}

# Human-readable descriptions for help display
KEYBINDING_DESCRIPTIONS = {
    # File operations
    "file_open": "Open XDF or CSV file",
    "file_save": "Save current session",
    "file_export": "Export processed signals",
    "file_quit": "Exit application",

    # Edit operations
    "edit_undo": "Undo last peak correction",
    "edit_redo": "Redo peak correction",
    "edit_add_peak": "Add peak at cursor position",
    "edit_delete_peak": "Delete selected peak",

    # View operations
    "view_zoom_in": "Zoom in on signal",
    "view_zoom_out": "Zoom out on signal",
    "view_reset": "Reset view to default",
    "view_fit": "Fit signal to window",
    "view_multi_signal": "Return to multi-signal view",
    "view_jump_to_time": "Jump to specific time",
    "view_pan_mode": "Pan mode (drag to pan)",
    "view_zoom_mode": "Zoom mode (drag to zoom)",

    # Peak navigation
    "peak_next": "Navigate to next peak",
    "peak_previous": "Navigate to previous peak",
    "peak_first": "Jump to first peak",
    "peak_last": "Jump to last peak",

    # Signal navigation
    "signal_pan_left": "Pan signal left",
    "signal_pan_right": "Pan signal right",
    "signal_page_forward": "Page forward in time",
    "signal_page_backward": "Page backward in time",

    # Help
    "help_show": "Show keyboard shortcuts help",
}

# Action groups for organizing help display
ACTION_GROUPS = {
    "File": [
        "file_open",
        "file_save",
        "file_export",
        "file_quit",
    ],
    "Edit": [
        "edit_undo",
        "edit_redo",
        "edit_add_peak",
        "edit_delete_peak",
    ],
    "View": [
        "view_zoom_in",
        "view_zoom_out",
        "view_reset",
        "view_fit",
        "view_multi_signal",
        "view_jump_to_time",
        "view_pan_mode",
        "view_zoom_mode",
    ],
    "Peak Navigation": [
        "peak_next",
        "peak_previous",
        "peak_first",
        "peak_last",
    ],
    "Signal Navigation": [
        "signal_pan_left",
        "signal_pan_right",
        "signal_page_forward",
        "signal_page_backward",
    ],
    "Help": [
        "help_show",
    ],
}


def get_keysequence(action: str) -> QKeySequence:
    """Get QKeySequence for an action.

    Args:
        action: Action name (e.g., "file_open")

    Returns:
        QKeySequence for the action, or empty sequence if not found

    Example:
        >>> seq = get_keysequence("file_open")
        >>> print(seq.toString())
        Ctrl+O
    """
    keybinding = KEYBINDINGS.get(action)
    if keybinding is None:
        return QKeySequence()

    # Convert StandardKey to QKeySequence if needed
    if isinstance(keybinding, QKeySequence.StandardKey):
        return QKeySequence(keybinding)

    return keybinding


def get_description(action: str) -> str:
    """Get human-readable description for an action.

    Args:
        action: Action name (e.g., "file_open")

    Returns:
        Description string, or empty string if not found
    """
    return KEYBINDING_DESCRIPTIONS.get(action, "")


def get_shortcut_text(action: str) -> str:
    """Get formatted shortcut text for display.

    Args:
        action: Action name (e.g., "file_open")

    Returns:
        Formatted shortcut string (e.g., "Ctrl+O")

    Example:
        >>> text = get_shortcut_text("file_save")
        >>> print(text)
        Ctrl+S
    """
    seq = get_keysequence(action)
    return seq.toString(QKeySequence.SequenceFormat.NativeText) if seq else ""


def get_help_text() -> str:
    """Generate formatted help text with all keyboard shortcuts.

    Returns:
        Multi-line string with all shortcuts organized by group

    Example:
        >>> help_text = get_help_text()
        >>> print(help_text)
        File:
          Ctrl+O - Open XDF or CSV file
          Ctrl+S - Save current session
          ...
    """
    lines = []
    for group_name, actions in ACTION_GROUPS.items():
        lines.append(f"\n{group_name}:")
        for action in actions:
            shortcut = get_shortcut_text(action)
            description = get_description(action)
            if shortcut:
                lines.append(f"  {shortcut:15} - {description}")
    return "\n".join(lines)
