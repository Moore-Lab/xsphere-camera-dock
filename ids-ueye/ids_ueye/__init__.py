"""Driver/API + test GUI for IDS uEye cameras (e.g. DCC1545M-GL)."""

from .camera import IDSUEye, list_devices

__all__ = ["IDSUEye", "list_devices"]
