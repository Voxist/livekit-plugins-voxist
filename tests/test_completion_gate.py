"""
Completion-gate scope: WHOSE delivery, WHOSE audio, and WHOSE ending.

Three defects, all of them in how the gate scoped or interpreted its facts,
and all of them producing the same shape of harm - a session that behaved
correctly killed by a terminal error, or a session that lost everything
reported as clean:

1. the session-scoped branch of _outcome read the SESSION flag for finals but
   the PER-ATTEMPT flag for interims, and _run clears that flag at the top of
   every attempt - so "only interims were delivered -> complete" was
   unreachable on any verdict a retry rendered;
2. the input-exhaustion probe collapsed "I could not look" into "there is no
   audio left", which let a channel still holding the whole session's audio
   be declared empty and completed as a clean SUCCESS with zero events;
3. "the engine produced no non-empty transcript" was read as "the transcript
   was lost", so a participant who never said anything recognizable - the
   default case for an open microphone - killed the stream with a
   non-retryable TranscriptLostError.

Plus the end of a turn itself: the plugin waited for a socket close that the
Kroko engine never sends, so every Kroko turn paid the full drain timeout and
was then classified as an ending imposed on us.

Helpers come from test_stream so both files drive the stream the same way.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from livekit.agents import APIConnectionError
from livekit.agents.stt import SpeechEventType

from livekit.plugins.voxist.exceptions import TranscriptLostError
from livekit.plugins.voxist.stream import VoxistSTTStream, _SessionOutcome

from .test_stream import (
    DEFAULT_CONFIG,
    FakeWS,
    frame,
    make_stream,
    mock_event_ch,
    speech_frame,
)


def feed_ws_error(ws):
    """
    A transport error mid-stream, exactly as aiohttp reports one: the receive
    iterator yields a message of type ERROR rather than raising.
    """
    ws.incoming.put_nowait(
        SimpleNamespace(type=aiohttp.WSMsgType.ERROR, data=None)
    )


class RenamedQueueChan:
    """
    livekit renamed Chan._queue: the channel still WORKS - it iterates, closes
    and reports qsize() as always - but the deque the plugin's probe reaches
    for is no longer called _queue. Deleting the real attribute instead breaks
    livekit's own iteration, which would test a channel that cannot exist.
    """

    def __init__(self, chan):
        self._chan = chan

    def __getattr__(self, name):
        if name == "_queue":
            raise AttributeError(name)
        return getattr(self._chan, name)

    def __aiter__(self):
        return self._chan.__aiter__()


def emitted_types(events):
    return [c.args[0].type for c in events.send_nowait.call_args_list]


def emitted_texts(events):
    return [
        c.args[0].alternatives[0].text
        for c in events.send_nowait.call_args_list
        if c.args[0].alternatives
    ]


class TestInterimDeliveryIsSessionScoped:
    """
    Interims that already reached the caller are a FACT about the session, not
    about the attempt that happened to ship them. The gate had no
    session-scoped interim flag at all.
    """

    @pytest.mark.asyncio
    async def test_retry_still_knows_interims_reached_the_caller(self):
        """
        The fold that gives the session an interim memory: a retry clears the
        per-attempt flag, so without it the session forgets that its text was
        already delivered.
        """
        stream = await make_stream()
        mock_event_ch(stream)

        # As attempt 1 left it: an INTERIM_TRANSCRIPT was emitted.
        stream._interim_received_this_attempt = True
        stream.end_input()  # nothing consumed: attempt 2 completes empty

        await asyncio.wait_for(stream._run(), timeout=5.0)

        assert stream._interim_received_this_attempt is False, (
            "precondition: the per-attempt flag is cleared by the retry"
        )
        assert stream._interim_delivered_in_session is True, (
            "the session must remember that text reached the caller as "
            "interims; the taxonomy's interim case is unreachable otherwise"
        )

    @pytest.mark.asyncio
    async def test_interim_only_session_survives_a_reset_during_the_drain(
        self, caplog
    ):
        """
        THE failure scenario, end to end and with interim_results at its
        default.

        Attempt 1 delivers START_OF_SPEECH + INTERIM_TRANSCRIPT("bonjour"),
        writes Done, and then the TCP connection resets during the post-Done
        drain - aiohttp yields WSMsgType.ERROR, so the attempt raises before
        it ever reaches the gate. livekit retries. Attempt 2 finds the input
        consumed and the audio unreplayable, and used to build an outcome with
        BOTH delivery flags False, because the session branch read the
        per-attempt interim flag that _run had just cleared: terminal
        TranscriptLostError, one recoverable=False event, stream dead - on a
        session whose text was already on the caller's screen.
        """
        ws1, ws2 = FakeWS(), FakeWS()
        dial = AsyncMock(side_effect=[ws1, ws2])
        stream = await make_stream(dial=dial)
        events = mock_event_ch(stream)

        async def attempt1():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws1.feed_json({"type": "partial", "text": "bonjour"})
            await asyncio.sleep(0.05)
            stream.end_input()
            await asyncio.sleep(0.05)
            feed_ws_error(ws1)  # connection reset during the drain

        task = asyncio.create_task(attempt1())
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        assert SpeechEventType.INTERIM_TRANSCRIPT in emitted_types(events), (
            "precondition: the text really did reach the caller"
        )
        assert ws1.sent_text == ["Done"], "precondition: attempt 1 wrote Done"

        # Attempt 2, the framework's retry: nothing left to send, audio gone.
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._run(), timeout=5.0)

        assert stream._session_complete, (
            "the session's text was delivered as interims; a retry must not "
            "turn it into a terminal transcript loss"
        )
        assert dial.await_count == 1, "there was nothing left to dial for"
        assert any("interim" in r.message for r in caplog.records), (
            "completing on interims must stay loud: finals-only consumers "
            "still see this session's tail as lost"
        )


class TestUnverifiedProbeRunsTheSession:
    """
    The input-exhaustion probe reads livekit's private Chan._queue. When it
    cannot look, the plugin must not decide the session's fate by inference:
    both shortcuts skip the exchange entirely.
    """

    @pytest.mark.asyncio
    async def test_queued_audio_is_transcribed_not_declared_empty(self):
        """
        THE failure scenario: on a livekit-agents release that renamed
        Chan._queue, a caller doing push_frame() x N + end_input() before the
        attempt reached its guard got the "ended with no audio to transcribe"
        shortcut - closed channel, probe answering True by fallback, nothing
        consumed yet. No dial, no send, no error, ZERO SpeechEvents, and every
        audio frame still sitting in the channel: total transcript loss
        reported as a clean, complete session, delivered by the fail-safe
        itself.
        """
        ws = FakeWS()
        dial = AsyncMock(return_value=ws)
        stream = await make_stream(dial=dial)
        events = mock_event_ch(stream)

        stream._input_ch.send_nowait(speech_frame())
        stream._input_ch.send_nowait(speech_frame())
        stream.end_input()
        stream._input_ch = RenamedQueueChan(stream._input_ch)
        assert stream._audio_consumed is False, (
            "precondition: nothing has been consumed yet"
        )

        async def scenario():
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "bonjour", "confidence": 0.9})
            await asyncio.sleep(0.05)
            ws.end()

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        assert dial.await_count == 1, (
            "the session's audio was still in the channel: it must be sent, "
            "not declared absent"
        )
        assert ws.sent_bytes, "the queued audio really was delivered"
        assert ws.sent_text == ["Done"]
        assert emitted_texts(events) == ["bonjour"], (
            "the transcript must reach the caller instead of vanishing into "
            "an 'empty session'"
        )
        assert stream._session_complete

    @pytest.mark.asyncio
    async def test_send_loop_still_treats_an_unverified_probe_as_end_of_session(
        self,
    ):
        """
        The other call site keeps the opposite fallback, and that asymmetry is
        the point of the tri-state. For end-of-SESSION detection a wrong True
        merges two segments; a wrong False injects 400ms of endpointing
        silence before a Done that forces the flush anyway - dead air at the
        end of every turn.
        """
        stream = await make_stream()
        ws = FakeWS()
        stream._ws = ws
        mock_event_ch(stream)

        stream._input_ch.send_nowait(speech_frame())
        stream.end_input()
        stream._input_ch = RenamedQueueChan(stream._input_ch)

        assert stream._probe_pending_input_only_sentinels() is None
        assert stream._pending_input_only_sentinels() is True

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

        # 400ms of injected silence at the 16kHz wire rate would be four
        # 100ms chunks of zeros appended after the caller's audio.
        assert not any(not any(chunk) for chunk in ws.sent_bytes), (
            "the trailing sentinel is end of SESSION, not a segment boundary: "
            "no endpointing silence may be shipped before Done"
        )
        assert ws.sent_text == ["Done"]


class TestUnintelligibleAudioIsNotLostAudio:
    """
    A real microphone never produces pure zeros, so _carries_signal counts
    room noise, coughing and background music as real audio. Equating "no
    non-empty transcript" with "the transcript was lost" therefore made a
    fatal, non-retryable error the DEFAULT outcome for a participant who
    simply never said anything recognizable.
    """

    @pytest.mark.asyncio
    async def test_empty_final_on_a_concluded_exchange_completes(self, caplog):
        """
        The engine answers {"type": "final", "text": ""} - it processed the
        audio and found nothing - and the exchange ends normally. That is an
        empty session, not a lost one.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        events = mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            stream.end_input()
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": ""})
            await asyncio.sleep(0.05)
            ws.end()

        task = asyncio.create_task(scenario())
        with caplog.at_level(logging.INFO, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        assert stream._real_audio_this_attempt, (
            "precondition: an open mic delivers real audio, so the loss "
            "verdict was reachable"
        )
        assert stream._session_complete, (
            "the engine said it had nothing to transcribe; that is an answer, "
            "not a lost transcript"
        )
        assert emitted_types(events) == [], "there was nothing to deliver"
        assert any("empty result" in r.message for r in caplog.records), (
            "an empty session must be traceable to the engine's own verdict"
        )

    @pytest.mark.asyncio
    async def test_engine_that_never_answers_still_raises(self, monkeypatch):
        """
        The protection this must not regress: real audio consumed, the engine
        returns NOTHING at all - crashed, or a heartbeat death, which look
        identical from here - and the exchange never concludes. Still a
        terminal loss.
        """
        monkeypatch.setattr(VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.3)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        stream._input_ch.send_nowait(speech_frame())
        stream.end_input()

        with pytest.raises(TranscriptLostError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        assert not stream._session_complete

    @pytest.mark.asyncio
    async def test_engine_text_that_was_never_delivered_is_still_a_loss(self):
        """
        The narrowness of the empty-report fact, pinned: the engine produced
        TEXT ("bonj"), interim_results is disabled so nothing reached the
        caller, and no final followed. Crediting "the engine answered" without
        asking WHAT it answered would turn that loss into a clean, empty
        session.
        """
        config = dict(DEFAULT_CONFIG, interim_results=False)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws), config=config)
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "partial", "text": "bonj"})
            await asyncio.sleep(0.05)
            stream.end_input()
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": ""})
            await asyncio.sleep(0.05)
            ws.end()

        task = asyncio.create_task(scenario())
        with pytest.raises(TranscriptLostError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task
        assert not stream._session_complete

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "concluded,engine_reported_empty,expected",
        [
            # The engine answered "nothing" and the exchange ended properly:
            # an empty session.
            (True, True, None),
            # It never answered: a crashed engine or a heartbeat death.
            (True, False, TranscriptLostError),
            # It answered "nothing", but the exchange was INTERRUPTED - the
            # empty answer covers only what the engine had seen so far, and
            # the audio in flight past it is still unaccounted for.
            (False, True, TranscriptLostError),
            (False, False, TranscriptLostError),
        ],
    )
    async def test_taxonomy_over_the_engine_answer_axis(
        self, concluded, engine_reported_empty, expected
    ):
        """The new axis of the taxonomy, driven directly at the gate."""
        stream = await make_stream()
        mock_event_ch(stream)

        outcome = _SessionOutcome(
            delivered_final=False,
            delivered_interim=False,
            unrecoverable_audio=True,
            concluded=concluded,
            detail=(
                f"concluded={concluded} "
                f"engine_reported_empty={engine_reported_empty}"
            ),
            engine_reported_empty=engine_reported_empty,
        )

        if expected is None:
            stream._finish_session(outcome)
            assert stream._session_complete
        else:
            with pytest.raises(expected):
                stream._finish_session(outcome)
            assert not stream._session_complete


class TestEngineThatNeverClosesTheSocket:
    """
    The gateway only forwards "Done" and waits for the ENGINE to close
    (simple-websocket-proxy.gateway.ts:1360, closing the client from its
    upstream-close handler at 948-973). Kroko never closes - it sends its
    final and holds the socket open (asr-all/kroko/bench/asr_bench.py:79) -
    so waiting for a close cost SESSION_DRAIN_TIMEOUT_SECONDS of dead air at
    the end of EVERY turn on the engine staging already routes the primary
    languages to, and then classified the healthy session as an ending
    imposed on us.
    """

    @pytest.mark.asyncio
    async def test_post_done_final_ends_the_turn_without_a_close(self, caplog):
        """
        The default constants, not test-shrunk ones: a turn must end shortly
        after the engine's post-Done final, nowhere near the 5s backstop, and
        without warning about a trailing transcript that was never missing.
        """
        assert VoxistSTTStream.SESSION_DRAIN_TIMEOUT_SECONDS == 5.0, (
            "precondition: the backstop is the shipped one"
        )
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        events = mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            stream.end_input()
            await asyncio.sleep(0.05)
            # The engine answers and then holds the socket open forever.
            ws.feed_json({"type": "final", "text": "bonjour", "confidence": 0.9})

        loop = asyncio.get_running_loop()
        start = loop.time()
        task = asyncio.create_task(scenario())
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task
        elapsed = loop.time() - start

        assert stream._session_complete
        assert emitted_texts(events) == ["bonjour"]
        assert elapsed < 2.0, (
            f"the turn took {elapsed:.1f}s: an engine that never closes must "
            "not cost the whole drain backstop in end-of-turn dead air"
        )
        assert not [
            r for r in caplog.records if "trailing transcript" in r.message
        ], (
            "the engine delivered its result: this is a normal ending, not "
            "one imposed on us"
        )

    @pytest.mark.asyncio
    async def test_second_final_of_one_flush_is_not_truncated(self, monkeypatch):
        """
        Why an IDLE period rather than "the first post-Done frame ends it":
        the engine emits one final per silence-delimited segment, so a tail
        holding two segments produces two finals, and returning on the first
        would drop the second silently.
        """
        monkeypatch.setattr(VoxistSTTStream, "POST_FINAL_IDLE_SECONDS", 0.3)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        events = mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            stream.end_input()
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "premier segment"})
            await asyncio.sleep(0.1)  # decode time, well inside the idle bound
            ws.feed_json({"type": "final", "text": "second segment"})

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        assert emitted_texts(events) == ["premier segment", "second segment"]
        assert stream._session_complete

    @pytest.mark.asyncio
    async def test_slow_first_final_is_governed_by_the_backstop(
        self, monkeypatch
    ):
        """
        The idle timer starts only once a first post-Done transcript has
        arrived, so a loaded engine's slow final is bounded by
        SESSION_DRAIN_TIMEOUT_SECONDS - not cut off after
        POST_FINAL_IDLE_SECONDS.
        """
        monkeypatch.setattr(VoxistSTTStream, "POST_FINAL_IDLE_SECONDS", 0.2)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        events = mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            stream.end_input()
            # Five times the idle bound: a slow decode, not a wedge.
            await asyncio.sleep(1.0)
            ws.feed_json({"type": "final", "text": "bonjour"})

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        assert emitted_texts(events) == ["bonjour"], (
            "a slow first final must not be abandoned at the idle bound"
        )
        assert stream._session_complete

    @pytest.mark.asyncio
    async def test_transcript_before_done_does_not_end_the_drain(
        self, monkeypatch, caplog
    ):
        """
        A transcript that arrived BEFORE "Done" says the engine was working
        earlier, not that it has finished flushing. Ending the drain on it -
        the tempting simplification, since it makes the idle rule uniform -
        would let a server that wedges the moment it is asked to finalize pass
        as a clean ending, and would report a session whose last segment never
        came as an unqualified success.
        """
        monkeypatch.setattr(VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 1.0)
        monkeypatch.setattr(VoxistSTTStream, "POST_FINAL_IDLE_SECONDS", 0.2)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "bonjour"})
            await asyncio.sleep(0.4)  # twice the idle bound, still pre-Done
            stream.end_input()
            # ...and the engine says nothing more, ever.

        task = asyncio.create_task(scenario())
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        assert stream._session_complete, "the delivered final is not lost"
        assert any(
            "trailing transcript" in r.message for r in caplog.records
        ), (
            "a server that never answered Done ended the exchange on its own "
            "terms, not the protocol's: the caller must be told a trailing "
            "transcript may be missing"
        )


class TestDrainStillReportsFactsNotVerdicts:
    """
    The drain hands the gate facts; every verdict stays in one place. These
    pin the two structural invariants around the new drain path.
    """

    @pytest.mark.asyncio
    async def test_drain_does_not_decide_success(self):
        """
        _await_engine_finalization returns an outcome or None - it never sets
        _session_complete and never raises the terminal error itself.
        """
        import ast
        import inspect
        import textwrap

        src = inspect.getsource(VoxistSTTStream._await_engine_finalization)
        tree = ast.parse(textwrap.dedent(src))

        # Parsed, not grepped - but broadly. The substring version failed the
        # moment a COMMENT here mentioned TranscriptLostError to explain which
        # defect an ordering fix prevented, and the first AST rewrite
        # overcorrected into a check that missed setattr, AugAssign, tuple
        # targets and `raise <name>` - reading as protection while forbidding
        # almost nothing.
        FORBIDDEN_NAMES = {"_session_complete", "_finish_session"}

        touched: set[str] = set()
        for node in ast.walk(tree):
            # Attribute writes in every assignment form
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            elif isinstance(node, ast.NamedExpr):
                targets = [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Attribute):
                        touched.add(sub.attr)
                    elif isinstance(sub, ast.Name):
                        touched.add(sub.id)
            # Any call, however routed, plus setattr's string argument
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute):
                    touched.add(fn.attr)
                elif isinstance(fn, ast.Name):
                    touched.add(fn.id)
                    if fn.id in ("setattr", "getattr"):
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and isinstance(
                                arg.value, str
                            ):
                                touched.add(arg.value)
            # Every raise, including `raise exc` and a bare re-raise
            if isinstance(node, ast.Raise) and node.exc is not None:
                for sub in ast.walk(node.exc):
                    if isinstance(sub, ast.Name):
                        touched.add(sub.id)
                    elif isinstance(sub, ast.Attribute):
                        touched.add(sub.attr)

        assert not (FORBIDDEN_NAMES & touched), (
            f"the drain must not decide completion, but touches "
            f"{sorted(FORBIDDEN_NAMES & touched)}"
        )
        assert "TranscriptLostError" not in touched, (
            "the terminal error has exactly one raise site, and it is the gate"
        )

        # The check must be able to SEE each forbidden form, or it is not a
        # check. Each of these once slipped past a version of this test.
        for snippet, name in [
            ("def f(self):\n    self._session_complete = True", "_session_complete"),
            ("def f(self):\n    setattr(self, '_session_complete', True)", "_session_complete"),
            ("def f(self):\n    self._session_complete, x = True, 1", "_session_complete"),
            (
                "def f(self):\n    exc = TranscriptLostError('x')\n    raise exc",
                "TranscriptLostError",
            ),
            ("def f(self):\n    raise TranscriptLostError('x')", "TranscriptLostError"),
            ("def f(self):\n    self._finish_session(o)", "_finish_session"),
        ]:
            probe = ast.parse(snippet)
            seen: set[str] = set()
            for node in ast.walk(probe):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
                    targets = [node.target]
                for t in targets:
                    for sub in ast.walk(t):
                        if isinstance(sub, ast.Attribute):
                            seen.add(sub.attr)
                        elif isinstance(sub, ast.Name):
                            seen.add(sub.id)
                if isinstance(node, ast.Call):
                    fn = node.func
                    if isinstance(fn, ast.Attribute):
                        seen.add(fn.attr)
                    elif isinstance(fn, ast.Name):
                        seen.add(fn.id)
                        if fn.id in ("setattr", "getattr"):
                            for arg in node.args:
                                if isinstance(arg, ast.Constant) and isinstance(
                                    arg.value, str
                                ):
                                    seen.add(arg.value)
                if isinstance(node, ast.Raise) and node.exc is not None:
                    for sub in ast.walk(node.exc):
                        if isinstance(sub, ast.Name):
                            seen.add(sub.id)
                        elif isinstance(sub, ast.Attribute):
                            seen.add(sub.attr)
            assert name in seen, (
                f"the detector cannot see {name!r} in {snippet!r}, so this "
                "invariant does not constrain what it claims"
            )

    @pytest.mark.asyncio
    async def test_server_close_during_the_drain_still_surfaces_its_error(self):
        """
        Returning None on a closed socket must keep the fall-through path that
        inspects the receive task's exception: a transport error during the
        drain is an interruption, not a conclusion.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(frame(1600))
            await asyncio.sleep(0.05)
            stream.end_input()
            await asyncio.sleep(0.05)
            feed_ws_error(ws)

        task = asyncio.create_task(scenario())
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task
        assert not stream._session_complete


class TestGateStillHasOneHome:
    """
    The structural invariants the earlier rounds bought, re-asserted against
    the new taxonomy branch and the new drain: adding a case must not add an
    exit.
    """

    @pytest.mark.asyncio
    async def test_the_new_branch_added_no_second_completion_site(self):
        import inspect

        module_src = inspect.getsource(
            __import__("livekit.plugins.voxist.stream", fromlist=["stream"])
        )
        assert module_src.count("_session_complete = True") == 1
        assert module_src.count("raise TranscriptLostError(") == 1

    @pytest.mark.asyncio
    async def test_every_gate_call_is_a_statement_in_run_attempt(self):
        """
        The gate's verdict may never be used as a VALUE (assigned, or its
        return inspected): the AST invariant in test_stream keys on the call
        being a bare statement immediately before each return, and a
        refactor that "captured the result" would silently disarm it.
        """
        import ast
        import inspect
        import textwrap

        fn = ast.parse(
            textwrap.dedent(inspect.getsource(VoxistSTTStream._run_attempt))
        ).body[0]

        calls = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_finish_session"
        ]
        assert len(calls) == 4, (
            "one per exit: no audio, exhausted input, drain, and the "
            "exchange's fall-through"
        )
        statements = [
            node.value
            for node in ast.walk(fn)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ]
        for call in calls:
            assert call in statements, (
                "a _finish_session call whose value is used escapes the "
                "structural check in test_stream"
            )


class TestNoDrainWatchdogTask:
    """
    The drain polls; it must not spawn a third child task. A watchdog would
    need the same cancellation care as send and recv, for no payoff.
    """

    @pytest.mark.asyncio
    async def test_the_drain_creates_no_tasks(self):
        import inspect

        src = inspect.getsource(VoxistSTTStream._await_engine_finalization)
        assert "create_task" not in src
