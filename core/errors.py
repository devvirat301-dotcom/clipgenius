"""Custom exceptions so the UI can show friendly messages."""


class ClipGeniusError(Exception):
    """Base class. The message is always safe to show to the user."""


class InvalidURLError(ClipGeniusError):
    pass


class VideoUnavailableError(ClipGeniusError):
    pass


class DownloadError(ClipGeniusError):
    pass


class TranscriptionError(ClipGeniusError):
    pass


class FFmpegError(ClipGeniusError):
    pass
