"""The package root's public API - the surface user code actually imports.

[10] Every other test in this suite reaches into submodules
(`from livekit.plugins.voxist.exceptions import TranscriptLostError`), which
is NOT what user code is documented to do. That left the documented import
path - `from livekit.plugins.voxist import TranscriptLostError` - completely
unasserted: a refactor that dropped or typoed an `__init__.py` entry would
keep the whole suite green while production code failed at import.

So this file asserts the ROOT, generically: every name in `__all__` must be
importable from the package and be the very same object the submodule defines.
That covers TranscriptLostError and every export added after it.
"""

import importlib

import pytest

import livekit.plugins.voxist as voxist


def test_all_is_declared_and_unique():
    assert voxist.__all__, "the package must declare its public API"
    assert len(set(voxist.__all__)) == len(voxist.__all__), (
        f"duplicate names in __all__: {voxist.__all__}"
    )


@pytest.mark.parametrize("name", voxist.__all__)
def test_every_exported_name_is_reachable_from_the_package_root(name):
    """`from livekit.plugins.voxist import <name>` must work."""
    assert hasattr(voxist, name), (
        f"{name!r} is advertised in __all__ but missing from the package root"
    )


@pytest.mark.parametrize(
    "name",
    [n for n in voxist.__all__ if not n.startswith("__")],
)
def test_root_export_is_the_same_object_as_the_submodule_definition(name):
    """
    A root export must not be a stale copy or a same-named lookalike.

    Identity is the assertion that matters: `except voxist.ConnectionError`
    in user code has to catch exactly what connection.py raises, which is the
    class defined in the submodule.
    """
    exported = getattr(voxist, name)
    origin = getattr(exported, "__module__", None)
    assert origin is not None and origin.startswith("livekit.plugins.voxist"), (
        f"{name!r} resolves to {origin!r}, outside this package"
    )

    submodule = importlib.import_module(origin)
    assert hasattr(submodule, name), (
        f"{name!r} claims to come from {origin}, which does not define it"
    )
    assert getattr(submodule, name) is exported, (
        f"{name!r} at the package root is a DIFFERENT object from "
        f"{origin}.{name}"
    )


def test_transcript_lost_error_is_importable_from_the_root():
    """The specific export [10] was about, spelled the way users spell it."""
    from livekit.plugins.voxist import TranscriptLostError
    from livekit.plugins.voxist.exceptions import (
        TranscriptLostError as SubmoduleTranscriptLostError,
    )
    from livekit.plugins.voxist.exceptions import VoxistError

    assert TranscriptLostError is SubmoduleTranscriptLostError
    assert issubclass(TranscriptLostError, VoxistError)


def test_deprecated_pool_era_exceptions_stay_importable():
    """
    [F] These are never raised any more, but they are PUBLIC names: deleting
    them would break `except ConnectionPoolExhaustedError` in existing user
    code at import time. They must keep importing, keep their inheritance, and
    say in their docstrings that they are deprecated and what replaced them.
    """
    from livekit.plugins.voxist import ConnectionPoolExhaustedError
    from livekit.plugins.voxist.exceptions import (
        BackpressureError,
        OwnershipViolationError,
        VoxistError,
    )
    from livekit.plugins.voxist.exceptions import (
        ConnectionError as VoxistConnectionError,
    )

    assert issubclass(ConnectionPoolExhaustedError, VoxistConnectionError)
    assert issubclass(BackpressureError, VoxistError)
    assert issubclass(OwnershipViolationError, VoxistError)

    for exc in (
        ConnectionPoolExhaustedError,
        BackpressureError,
        OwnershipViolationError,
    ):
        doc = exc.__doc__ or ""
        assert "DEPRECATED" in doc, f"{exc.__name__} does not say it is deprecated"
        assert "never raised" in doc, f"{exc.__name__} does not say it is dead"


def test_no_pool_fiction_in_the_package_docstring():
    """
    [14] The package docstring must not advertise the deleted architecture.

    The pool is gone; a feature list promising pooling (or automatic
    reconnection the plugin does not own) sends users looking for machinery
    that does not exist.
    """
    doc = voxist.__doc__ or ""
    features = doc.split("Example:")[0]
    assert "pooling" not in features.lower()
    assert "connection_pool_size" not in doc, (
        "the documented example must not pass a deprecated, ignored parameter"
    )


def test_no_pool_fiction_in_the_stt_class_docstring():
    """[14] Same for VoxistSTT, whose usage example users copy verbatim."""
    doc = voxist.VoxistSTT.__doc__ or ""
    features = doc.split("Task Lifecycle")[0]
    assert "pooling" not in features.lower()
    assert "connection_pool_size" not in doc, (
        "the documented example must not pass a deprecated, ignored parameter"
    )
