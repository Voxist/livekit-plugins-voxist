"""VoxistTTS against a loopback stand-in for voxist-tts ``/v1/tts/{voice}/stream/text``.

The fake follows ``voxist-server/src/routes/ws.rs`` and
``docs/tts-websocket-contract.md``: config first, text chunks, ``end``; binary
PCM16 frames, ``{"error", "code", "severity"}`` and ``{"done": true}`` back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError
from livekit.agents.types import APIConnectOptions

from livekit.plugins.voxist import VoxistTTS
from livekit.plugins.voxist.tts import CLOSE_TIMEOUT, PROTOCOL_VERSION, SAMPLE_RATE

# A hang must fail the test, not stall CI.
pytestmark = pytest.mark.timeout(20)

NO_RETRY = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5.0)
FAST_RETRY = APIConnectOptions(max_retry=2, retry_interval=0.0, timeout=5.0)
SECRET = "s3cr3t-value-never-logged"
CF = {"CF-Access-Client-Id": "id.access", "CF-Access-Client-Secret": SECRET}
FRAME = b"\x01\x00" * (SAMPLE_RATE * 80 // 1000)  # 80 ms of PCM16


class FakeVoxistTTS:
    """Records every session; ``behaviour`` decides how a session is answered."""

    def __init__(self) -> None:
        self.sessions: list[dict[str, Any]] = []
        self.handshake_status: list[int] = []  # statuses to answer before upgrading
        self.behaviour: Callable[[web.WebSocketResponse, dict[str, Any]], Awaitable[None]]
        self.behaviour = self.speak
        self.closed = asyncio.Event()
        self.autoclose = True

    async def speak(self, ws: web.WebSocketResponse, session: dict[str, Any]) -> None:
        """Two frames per text chunk, then done after end."""
        frames = 0
        async for msg in ws:
            data = json.loads(msg.data)
            if "text" in data:
                session["texts"].append(data["text"])
                for _ in range(2):
                    await ws.send_bytes(FRAME)
                    frames += 1
            if data.get("end"):
                await ws.send_str(json.dumps({
                    "done": True, "request_id": "r1", "utterance_id": "u1",
                    "duration_s": frames * 0.08, "num_frames": frames,
                    "resolved_voice_id": session["voice"],
                }))
                return

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if self.handshake_status:
            return web.Response(status=self.handshake_status.pop(0))
        ws = web.WebSocketResponse(autoclose=self.autoclose)
        await ws.prepare(request)
        session: dict[str, Any] = {
            "voice": request.match_info["voice"],
            "headers": dict(request.headers),
            "texts": [],
        }
        self.sessions.append(session)
        first = await ws.receive()
        session["config"] = json.loads(first.data)
        try:
            await self.behaviour(ws, session)
        finally:
            await ws.close()
            session["closed_at"] = time.monotonic()
            self.closed.set()
        return ws


@pytest.fixture
async def fake():
    server = FakeVoxistTTS()
    app = web.Application()
    app.router.add_get("/v1/tts/{voice}/stream/text", server.handle)
    test_server = TestServer(app, host="127.0.0.1")
    await test_server.start_server()
    server.url = str(test_server.make_url("")).rstrip("/")
    yield server
    await test_server.close()


def make_tts(fake: FakeVoxistTTS, **kw: Any) -> VoxistTTS:
    opts: dict[str, Any] = {
        "base_url": fake.url, "voice_id": "ada", "language": "fr", "headers": CF,
    }
    opts.update(kw)
    return VoxistTTS(**opts)


async def run_stream(tts: VoxistTTS, chunks: list[str], conn=NO_RETRY) -> list[Any]:
    stream = tts.stream(conn_options=conn)
    for chunk in chunks:
        stream.push_text(chunk)
    stream.end_input()
    events = [ev async for ev in stream]
    await stream.aclose()
    return events


def pcm(events: list[Any]) -> bytes:
    return b"".join(bytes(ev.frame.data) for ev in events)


async def test_streams_text_and_audio_with_the_documented_handshake(fake):
    tts = make_tts(fake)
    events = await run_stream(tts, ["Bonjour, ", "je vous ", "écoute."])
    await tts.aclose()

    (session,) = fake.sessions
    assert session["config"] == {"format": "pcm_s16le", "language": "fr",
                                 "version": PROTOCOL_VERSION}
    assert session["texts"] == ["Bonjour, ", "je vous ", "écoute."]
    assert session["headers"]["CF-Access-Client-Id"] == "id.access"
    assert session["headers"]["CF-Access-Client-Secret"] == SECRET
    assert len(pcm(events)) >= 6 * len(FRAME)
    assert {ev.frame.sample_rate for ev in events} == {SAMPLE_RATE}
    assert events[-1].is_final


async def test_first_audio_arrives_before_the_input_ends(fake):
    """Audio for the first chunk is emitted while later text is still to come."""
    tts = make_tts(fake)
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text("Bonjour. ")
    first = await asyncio.wait_for(stream.__anext__(), 2.0)
    assert len(bytes(first.frame.data)) > 0
    stream.push_text("Au revoir.")
    stream.end_input()
    rest = [ev async for ev in stream]
    await stream.aclose()
    await tts.aclose()
    assert fake.sessions[0]["texts"] == ["Bonjour. ", "Au revoir."]
    assert rest[-1].is_final


async def test_https_base_url_becomes_wss_and_voice_is_percent_encoded():
    tts = VoxistTTS(base_url="https://tts.example.com/", voice_id="a/b c", language="de")
    assert tts._opts.url == "wss://tts.example.com/v1/tts/a%2Fb%20c/stream/text"
    await tts.aclose()


@pytest.mark.parametrize("url", ["http://127.0.0.1:1", "ws://localhost:1", "http://[::1]:1"])
def test_plain_http_is_allowed_to_loopback_only(url):
    assert VoxistTTS(base_url=url, voice_id="ada", language="fr")._opts.ws_base.startswith("ws://")


@pytest.mark.parametrize("kw", [
    {"voice_id": ""}, {"voice_id": "jobs"}, {"voice_id": " ada"}, {"voice_id": "."},
    {"voice_id": ".."}, {"language": ""}, {"base_url": "ftp://x"},
    {"base_url": "https://x/?q=1"}, {"first_audio_timeout": 0},
    {"base_url": "http://tts.example.com"}, {"base_url": "ws://tts.example.com"},
])
def test_bad_configuration_is_refused(kw):
    opts: dict[str, Any] = {"base_url": "https://x", "voice_id": "ada", "language": "fr"}
    opts.update(kw)
    with pytest.raises(ValueError):
        VoxistTTS(**opts)


async def test_chunked_synthesize_sends_text_then_end(fake):
    tts = make_tts(fake)
    events = [ev async for ev in tts.synthesize("Merci.", conn_options=NO_RETRY)]
    await tts.aclose()
    assert fake.sessions[0]["texts"] == ["Merci."]
    assert len(pcm(events)) >= 2 * len(FRAME)


async def test_empty_input_speaks_nothing_and_closes_the_socket(fake):
    tts = make_tts(fake)
    events = await run_stream(tts, [])
    await tts.aclose()
    assert events == []
    await asyncio.wait_for(fake.closed.wait(), 2.0)
    assert fake.sessions[0]["texts"] == []


async def test_fatal_error_frame_fails_without_retry(fake):
    async def refuse(ws, session):
        await ws.receive()
        await ws.send_str(json.dumps({"error": "Voice 'ada' not found", "code": "voice_not_found",
                                      "severity": "fatal", "request_id": "r9"}))

    fake.behaviour = refuse
    tts = make_tts(fake)
    with pytest.raises(APIStatusError) as info:
        await run_stream(tts, ["Bonjour."], conn=FAST_RETRY)
    await tts.aclose()
    assert info.value.status_code == 404
    assert info.value.request_id == "r9"
    assert not info.value.retryable
    assert len(fake.sessions) == 1  # not replayed


async def test_advisory_does_not_end_the_session(fake):
    async def advise(ws, session):
        await ws.send_str(json.dumps({"error": "ignored", "code": "chunk_ignored",
                                      "severity": "advisory", "request_id": "r1"}))
        await fake.speak(ws, session)

    fake.behaviour = advise
    tts = make_tts(fake)
    events = await run_stream(tts, ["Bonjour."])
    await tts.aclose()
    assert len(pcm(events)) >= 2 * len(FRAME)


async def test_close_before_done_is_an_error(fake):
    async def drop(ws, session):
        await ws.receive()
        await ws.send_bytes(FRAME)

    fake.behaviour = drop
    tts = make_tts(fake)
    with pytest.raises(APIConnectionError):
        await run_stream(tts, ["Bonjour."])
    await tts.aclose()


async def test_no_first_audio_in_time_is_a_timeout(fake):
    async def mute(ws, session):
        async for _ in ws:
            pass

    fake.behaviour = mute
    tts = make_tts(fake, first_audio_timeout=0.3)
    started = time.monotonic()
    with pytest.raises(APITimeoutError):
        await run_stream(tts, ["Bonjour."])
    assert time.monotonic() - started < 2.0
    await tts.aclose()


async def test_silence_after_end_is_a_timeout(fake):
    async def stall(ws, session):
        await ws.receive()  # text
        await ws.send_bytes(FRAME)
        async for _ in ws:
            pass

    fake.behaviour = stall
    tts = make_tts(fake, idle_timeout=0.3)
    with pytest.raises(APITimeoutError):
        await run_stream(tts, ["Bonjour."])
    await tts.aclose()


async def test_barge_in_closes_the_socket_promptly(fake):
    async def long_speech(ws, session):
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                return
            for _ in range(200):  # 16 s of audio, paced
                await ws.send_bytes(FRAME)
                await asyncio.sleep(0.02)

    fake.behaviour = long_speech
    tts = make_tts(fake)
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text("Une très longue réponse.")
    await asyncio.wait_for(stream.__anext__(), 2.0)
    interrupted = time.monotonic()
    await stream.aclose()
    await asyncio.wait_for(fake.closed.wait(), 2.0)
    assert fake.sessions[0]["closed_at"] - interrupted < 1.5
    await tts.aclose()


async def test_handshake_5xx_is_retried_locally_then_succeeds(fake):
    fake.handshake_status = [503]
    tts = make_tts(fake)
    events = await run_stream(tts, ["Bonjour."], conn=FAST_RETRY)
    await tts.aclose()
    assert len(pcm(events)) > 0


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_refusal_is_not_retried(fake, status):
    fake.handshake_status = [status, status, status]
    tts = make_tts(fake)
    with pytest.raises(APIStatusError) as info:
        await run_stream(tts, ["Bonjour."], conn=FAST_RETRY)
    await tts.aclose()
    assert info.value.status_code == status
    assert fake.handshake_status == [status, status]  # one attempt only


async def test_redirect_is_never_followed_and_credentials_stay_home(fake, caplog):
    seen: list[dict[str, str]] = []

    async def elsewhere(request: web.Request) -> web.Response:
        seen.append(dict(request.headers))
        return web.Response(text="login")

    other = web.Application()
    other.router.add_get("/login", elsewhere)
    other_server = TestServer(other, host="127.0.0.1")
    await other_server.start_server()

    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound(str(other_server.make_url("/login")))

    app = web.Application()
    app.router.add_get("/v1/tts/{voice}/stream/text", redirect)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    tts = VoxistTTS(base_url=str(server.make_url("")), voice_id="ada", language="fr", headers=CF)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(APIStatusError) as info:
        await run_stream(tts, ["Bonjour."], conn=FAST_RETRY)
    await tts.aclose()
    await server.close()
    await other_server.close()
    assert seen == []
    assert info.value.status_code == 302
    assert not info.value.retryable
    assert SECRET not in str(info.value)
    assert SECRET not in caplog.text


async def test_closed_tts_refuses_new_streams(fake):
    tts = make_tts(fake)
    await tts.aclose()
    with pytest.raises(APIConnectionError):
        await run_stream(tts, ["Bonjour."])


def _refuse_with(code: str, *, after_audio: bool = False):
    async def refuse(ws, session):
        msg = await ws.receive()
        session["texts"].append(json.loads(msg.data).get("text"))
        if after_audio:
            await ws.send_bytes(FRAME)
        await ws.send_str(json.dumps({"error": "x", "code": code, "severity": "fatal",
                                      "request_id": "r"}))
    return refuse


@pytest.mark.parametrize("code", ["capacity_exceeded", "backend_error", "session_timeout"])
async def test_a_503_class_refusal_before_any_audio_is_retried_with_the_same_text(fake, code):
    refuse = _refuse_with(code)

    async def first_refuses_then_speaks(ws, session):
        if len(fake.sessions) == 1:
            await refuse(ws, session)
        else:
            await fake.speak(ws, session)

    fake.behaviour = first_refuses_then_speaks
    tts = make_tts(fake)
    events = await run_stream(tts, ["Bonjour."], conn=FAST_RETRY)
    await tts.aclose()
    assert [s["texts"] for s in fake.sessions] == [["Bonjour."], ["Bonjour."]]
    assert len(pcm(events)) > 0


async def test_a_503_class_refusal_after_audio_is_not_retried(fake):
    fake.behaviour = _refuse_with("backend_error", after_audio=True)
    tts = make_tts(fake)
    with pytest.raises(APIStatusError) as info:
        await run_stream(tts, ["Bonjour."], conn=FAST_RETRY)
    await tts.aclose()
    assert info.value.status_code == 502 and not info.value.retryable
    assert len(fake.sessions) == 1


async def test_a_handshake_that_hangs_times_out_and_the_last_attempt_is_final():
    accepted: list[asyncio.StreamWriter] = []

    async def silent(reader, writer):
        accepted.append(writer)
        await asyncio.sleep(30)

    server = await asyncio.start_server(silent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    tts = VoxistTTS(base_url=f"http://127.0.0.1:{port}", voice_id="ada", language="fr",
                    connect_timeout=0.2)
    started = time.monotonic()
    with pytest.raises(APITimeoutError) as info:
        await run_stream(tts, ["Bonjour."], conn=FAST_RETRY)
    elapsed = time.monotonic() - started
    await tts.aclose()
    for writer in accepted:
        writer.close()
    server.close()
    assert len(accepted) == 3  # max_retry=2: three dials
    assert not info.value.retryable
    assert elapsed < 2.0


async def test_the_final_dial_error_is_not_retryable(fake):
    fake.handshake_status = [503, 503, 503]
    tts = make_tts(fake)
    with pytest.raises(APIStatusError) as info:
        await run_stream(tts, ["Bonjour."], conn=FAST_RETRY)
    await tts.aclose()
    assert info.value.status_code == 503 and not info.value.retryable
    assert fake.handshake_status == []


async def test_done_whose_length_disagrees_with_the_audio_is_logged(fake, caplog):
    async def short(ws, session):
        await ws.receive()  # text
        await ws.receive()  # end
        await ws.send_bytes(FRAME)
        await ws.send_str(json.dumps({"done": True, "duration_s": 5.0}))

    fake.behaviour = short
    tts = make_tts(fake)
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
        await run_stream(tts, ["Bonjour."])
    await tts.aclose()
    assert "audio length differs from done" in caplog.text


async def test_a_server_that_never_answers_the_close_costs_close_timeout_at_most(fake):
    fake.autoclose = False

    async def deaf(ws, session):
        await fake.speak(ws, session)
        await asyncio.sleep(10)  # never reads our close frame

    fake.behaviour = deaf
    tts = make_tts(fake)
    started = time.monotonic()
    await run_stream(tts, ["Bonjour."])
    assert time.monotonic() - started < CLOSE_TIMEOUT + 1.0
    await tts.aclose()


async def test_after_a_text_limit_nothing_more_is_sent_and_done_is_logged(fake, caplog):
    async def limited(ws, session):
        msg = await ws.receive()
        session["texts"].append(json.loads(msg.data)["text"])
        await ws.send_bytes(FRAME)
        await ws.send_str(json.dumps({"error": "limit", "code": "text_limit_reached",
                                      "severity": "advisory", "request_id": "r"}))
        session["after"] = []
        try:
            while True:
                later = await asyncio.wait_for(ws.receive(), 0.4)
                session["after"].append(later.data)
        except asyncio.TimeoutError:
            pass
        await ws.send_str(json.dumps({"done": True, "duration_s": 0.08,
                                      "limit_reached": True, "stop_reason": "limit"}))

    fake.behaviour = limited
    tts = make_tts(fake)
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text("Premier. ")
    await asyncio.wait_for(stream.__anext__(), 2.0)
    await asyncio.sleep(0.1)  # the advisory has arrived
    stream.push_text("Deuxième. ")
    stream.push_text("Troisième.")
    stream.end_input()
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
        [ev async for ev in stream]
    await stream.aclose()
    await tts.aclose()
    assert fake.sessions[0]["texts"] == ["Premier. "]
    assert fake.sessions[0]["after"] == []  # no text, no end
    assert "did not speak all the text" in caplog.text


async def test_the_first_audio_clock_restarts_with_each_text(fake):
    """An LLM still writing its sentence is not dead air: the clock runs from the
    latest text, so slow text with audio only after end still succeeds."""
    async def on_end(ws, session):
        async for msg in ws:
            data = json.loads(msg.data)
            if "text" in data:
                session["texts"].append(data["text"])
            if data.get("end"):
                await ws.send_bytes(FRAME)
                await ws.send_str(json.dumps({"done": True, "duration_s": 0.08}))
                return

    fake.behaviour = on_end
    tts = make_tts(fake, first_audio_timeout=0.5)
    stream = tts.stream(conn_options=NO_RETRY)
    for word in ("Une ", "phrase ", "lente."):
        stream.push_text(word)
        await asyncio.sleep(0.3)
    stream.end_input()
    events = [ev async for ev in stream]
    await stream.aclose()
    await tts.aclose()
    assert fake.sessions[0]["texts"] == ["Une ", "phrase ", "lente."]
    assert len(pcm(events)) > 0


async def test_check_connection_passes_a_healthy_socket_and_closes_it(fake):
    async def wait(ws, session):
        async for _ in ws:
            pass

    fake.behaviour = wait
    tts = make_tts(fake)
    await tts.check_connection(quiet_s=0.2)
    await asyncio.wait_for(fake.closed.wait(), 2.0)
    await tts.aclose()
    assert fake.sessions[0]["config"]["language"] == "fr"
    assert fake.sessions[0]["texts"] == []


async def test_check_connection_reports_a_fatal_frame(fake):
    async def refuse(ws, session):
        await ws.send_str(json.dumps({"error": "no", "code": "voice_not_found",
                                      "severity": "fatal", "request_id": "r"}))

    fake.behaviour = refuse
    tts = make_tts(fake)
    with pytest.raises(APIStatusError) as info:
        await tts.check_connection(quiet_s=1.0)
    await tts.aclose()
    assert info.value.status_code == 404 and info.value.body == {"code": "voice_not_found"}


async def test_check_connection_reports_an_access_refusal(fake):
    fake.handshake_status = [403]
    tts = make_tts(fake)
    with pytest.raises(APIStatusError) as info:
        await tts.check_connection()
    await tts.aclose()
    assert info.value.status_code == 403
