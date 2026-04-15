"""Tests for Feishu interactive card approval buttons."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Feishu mock so FeishuAdapter can be imported without lark-oapi
# ---------------------------------------------------------------------------
def _ensure_feishu_mocks():
    """Provide stubs for lark-oapi / aiohttp.web so the import succeeds."""
    if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
        mod = MagicMock()
        for name in (
            "lark_oapi", "lark_oapi.api.im.v1",
            "lark_oapi.event", "lark_oapi.event.callback_type",
        ):
            sys.modules.setdefault(name, mod)
    if importlib.util.find_spec("aiohttp") is None and "aiohttp" not in sys.modules:
        aio = MagicMock()
        sys.modules.setdefault("aiohttp", aio)
        sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig
import gateway.platforms.feishu as feishu_module
from gateway.platforms.feishu import FeishuAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter() -> FeishuAdapter:
    """Create a FeishuAdapter with mocked internals."""
    config = PlatformConfig(enabled=True)
    adapter = FeishuAdapter(config)
    adapter._client = MagicMock()
    return adapter


def _make_card_action_data(
    action_value: dict,
    chat_id: str = "oc_12345",
    open_id: str = "ou_user1",
    token: str = "tok_abc",
) -> SimpleNamespace:
    """Create a mock Feishu card action callback data object."""
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(open_chat_id=chat_id),
            operator=SimpleNamespace(open_id=open_id),
            action=SimpleNamespace(
                tag="button",
                value=action_value,
            ),
        ),
    )


def _close_submitted_coro(coro, _loop):
    """Close scheduled coroutines in sync-handler tests to avoid unawaited warnings."""
    coro.close()
    return SimpleNamespace(add_done_callback=lambda *_args, **_kwargs: None)


# ===========================================================================
# send_exec_approval — interactive card with buttons
# ===========================================================================

class TestFeishuExecApproval:
    """Test send_exec_approval sends an interactive card."""

    @pytest.mark.asyncio
    async def test_sends_interactive_card(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_001"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            result = await adapter.send_exec_approval(
                chat_id="oc_12345",
                command="rm -rf /important",
                session_key="agent:main:feishu:group:oc_12345",
                description="dangerous deletion",
            )

        assert result.success is True
        assert result.message_id == "msg_001"

        mock_send.assert_called_once()
        kwargs = mock_send.call_args[1]
        assert kwargs["chat_id"] == "oc_12345"
        assert kwargs["msg_type"] == "interactive"

        # Verify card payload contains the command and buttons
        card = json.loads(kwargs["payload"])
        assert card["header"]["template"] == "orange"
        assert "rm -rf /important" in card["elements"][0]["content"]
        assert "dangerous deletion" in card["elements"][0]["content"]

        # Check buttons
        actions = card["elements"][1]["actions"]
        assert len(actions) == 4
        action_names = [a["value"]["hermes_action"] for a in actions]
        assert action_names == [
            "approve_once", "approve_session", "approve_always", "deny"
        ]

    @pytest.mark.asyncio
    async def test_stores_approval_state(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_002"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await adapter.send_exec_approval(
                chat_id="oc_12345",
                command="echo test",
                session_key="my-session-key",
            )

        assert len(adapter._approval_state) == 1
        approval_id = list(adapter._approval_state.keys())[0]
        state = adapter._approval_state[approval_id]
        assert state["session_key"] == "my-session-key"
        assert state["message_id"] == "msg_002"
        assert state["chat_id"] == "oc_12345"

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._client = None
        result = await adapter.send_exec_approval(
            chat_id="oc_12345", command="ls", session_key="s"
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_truncates_long_command(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_003"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_send:
            long_cmd = "x" * 5000
            await adapter.send_exec_approval(
                chat_id="oc_12345", command=long_cmd, session_key="s"
            )

        card = json.loads(mock_send.call_args[1]["payload"])
        content = card["elements"][0]["content"]
        assert "..." in content
        assert len(content) < 5000

    @pytest.mark.asyncio
    async def test_multiple_approvals_get_unique_ids(self):
        adapter = _make_adapter()

        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="msg_x"),
        )
        with patch.object(
            adapter, "_feishu_send_with_retry", new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await adapter.send_exec_approval(
                chat_id="oc_1", command="cmd1", session_key="s1"
            )
            await adapter.send_exec_approval(
                chat_id="oc_2", command="cmd2", session_key="s2"
            )

        assert len(adapter._approval_state) == 2
        ids = list(adapter._approval_state.keys())
        assert ids[0] != ids[1]


# ===========================================================================
# _resolve_approval — approval state pop + gateway resolution
# ===========================================================================

class TestResolveApproval:
    """Test _resolve_approval pops state and calls resolve_gateway_approval."""

    @pytest.mark.asyncio
    async def test_resolves_once(self):
        adapter = _make_adapter()
        adapter._approval_state[1] = {
            "session_key": "agent:main:feishu:group:oc_12345",
            "message_id": "msg_001",
            "chat_id": "oc_12345",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._resolve_approval(1, "once", "Norbert")

        mock_resolve.assert_called_once_with("agent:main:feishu:group:oc_12345", "once")
        assert 1 not in adapter._approval_state

    @pytest.mark.asyncio
    async def test_resolves_deny(self):
        adapter = _make_adapter()
        adapter._approval_state[2] = {
            "session_key": "some-session",
            "message_id": "msg_002",
            "chat_id": "oc_12345",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._resolve_approval(2, "deny", "Alice")

        mock_resolve.assert_called_once_with("some-session", "deny")

    @pytest.mark.asyncio
    async def test_resolves_session(self):
        adapter = _make_adapter()
        adapter._approval_state[3] = {
            "session_key": "sess-3",
            "message_id": "msg_003",
            "chat_id": "oc_99",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._resolve_approval(3, "session", "Bob")

        mock_resolve.assert_called_once_with("sess-3", "session")

    @pytest.mark.asyncio
    async def test_resolves_always(self):
        adapter = _make_adapter()
        adapter._approval_state[4] = {
            "session_key": "sess-4",
            "message_id": "msg_004",
            "chat_id": "oc_55",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._resolve_approval(4, "always", "Carol")

        mock_resolve.assert_called_once_with("sess-4", "always")

    @pytest.mark.asyncio
    async def test_already_resolved_drops_silently(self):
        adapter = _make_adapter()

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._resolve_approval(99, "once", "Nobody")

        mock_resolve.assert_not_called()

# ===========================================================================
# _handle_card_action_event — non-approval card actions
# ===========================================================================

class TestNonApprovalCardAction:
    """Non-approval card actions should still route as synthetic commands."""

    @pytest.mark.asyncio
    async def test_routes_as_synthetic_command(self):
        adapter = _make_adapter()

        data = _make_card_action_data(
            action_value={"custom_action": "something_else"},
            token="tok_normal",
        )

        with (
            patch.object(
                adapter, "_resolve_sender_profile", new_callable=AsyncMock,
                return_value={"user_id": "ou_u", "user_name": "Dave", "user_id_alt": None},
            ),
            patch.object(adapter, "get_chat_info", new_callable=AsyncMock, return_value={"name": "Test Chat"}),
            patch.object(adapter, "_handle_message_with_guards", new_callable=AsyncMock) as mock_handle,
        ):
            await adapter._handle_card_action_event(data)

        mock_handle.assert_called_once()
        event = mock_handle.call_args[0][0]
        assert "/card button" in event.text


# ===========================================================================
# _on_card_action_trigger — inline card response for approval actions
# ===========================================================================

class _FakeCallBackCard:
    def __init__(self):
        self.type = None
        self.data = None


class _FakeP2Response:
    def __init__(self):
        self.card = None


@pytest.fixture(autouse=False)
def _patch_callback_card_types(monkeypatch):
    """Provide real-ish P2CardActionTriggerResponse / CallBackCard for tests."""
    monkeypatch.setattr(feishu_module, "P2CardActionTriggerResponse", _FakeP2Response)
    monkeypatch.setattr(feishu_module, "CallBackCard", _FakeCallBackCard)


class TestCardActionCallbackResponse:
    """Test that _on_card_action_trigger returns updated card inline."""

    def test_drops_action_when_loop_not_ready(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = None
        data = _make_card_action_data({"hermes_action": "approve_once", "approval_id": 1})

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_returns_card_for_approve_action(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 1},
            open_id="ou_bob",
        )
        adapter._sender_name_cache["ou_bob"] = ("Bob", 9999999999)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is not None
        assert response.card.type == "raw"
        card = response.card.data
        assert card["header"]["template"] == "green"
        assert "Approved once" in card["header"]["title"]["content"]
        assert "Bob" in card["elements"][0]["content"]

    def test_returns_card_for_deny_action(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data(
            {"hermes_action": "deny", "approval_id": 2},
        )

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response.card is not None
        card = response.card.data
        assert card["header"]["template"] == "red"
        assert "Denied" in card["header"]["title"]["content"]

    def test_ignores_missing_approval_id(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data({"hermes_action": "approve_once"})

        with patch("asyncio.run_coroutine_threadsafe") as mock_submit:
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None
        mock_submit.assert_not_called()

    def test_no_card_for_non_approval_action(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data({"some_other": "value"})

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        assert response is not None
        assert response.card is None

    def test_falls_back_to_open_id_when_name_not_cached(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data(
            {"hermes_action": "approve_session", "approval_id": 3},
            open_id="ou_unknown",
        )

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        card = response.card.data
        assert "ou_unknown" in card["elements"][0]["content"]

    def test_ignores_expired_cached_name(self, _patch_callback_card_types):
        adapter = _make_adapter()
        adapter._loop = MagicMock()
        adapter._loop.is_closed = MagicMock(return_value=False)
        data = _make_card_action_data(
            {"hermes_action": "approve_once", "approval_id": 4},
            open_id="ou_expired",
        )
        adapter._sender_name_cache["ou_expired"] = ("Old Name", 1)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_close_submitted_coro):
            response = adapter._on_card_action_trigger(data)

        card = response.card.data
        assert "Old Name" not in card["elements"][0]["content"]
        assert "ou_expired" in card["elements"][0]["content"]


# ===========================================================================
# _run_official_feishu_ws_client — CARD monkey-patch
# ===========================================================================


class _FakeHeader:
    """Minimal protobuf-like header for testing."""
    def __init__(self, key: str = "", value: str = ""):
        self.key = key
        self.value = value


class _FakeHeaders(list):
    """Minimal protobuf-like repeated headers container."""
    def add(self):
        h = _FakeHeader()
        self.append(h)
        return h


class _FakeFrame:
    """Minimal protobuf-like Frame for testing _patched_handle_data_frame."""
    def __init__(self, headers=None, payload=b"{}"):
        self.headers = headers if headers is not None else _FakeHeaders()
        self.payload = payload

    def SerializeToString(self):
        return self.payload


def _make_card_frame(
    msg_type: str = "card",
    msg_id: str = "msg_card_001",
    trace_id: str = "trace_001",
    sum_: str = "1",
    seq: str = "0",
    payload: bytes = b'{"test": "data"}',
) -> _FakeFrame:
    """Build a fake WS frame with CARD type headers."""
    headers = _FakeHeaders()
    for k, v in [
        ("type", msg_type),
        ("message_id", msg_id),
        ("trace_id", trace_id),
        ("sum", sum_),
        ("seq", seq),
    ]:
        headers.append(_FakeHeader(k, v))
    return _FakeFrame(headers=headers, payload=payload)


def _ensure_ws_mocks():
    """Provide deeper lark_oapi.ws.* stubs needed by _run_official_feishu_ws_client."""
    import enum as _enum

    class _MockMessageType(_enum.Enum):
        EVENT = "event"
        CARD = "card"
        PING = "ping"
        PONG = "pong"

    class _MockFrameType(_enum.Enum):
        CONTROL = 0
        DATA = 1

    class _MockResponse:
        def __init__(self, code=None):
            self.code = code
            self.data = None

    ws_client_mod = MagicMock()
    ws_client_mod.websockets = MagicMock()

    ws_enum_mod = MagicMock()
    ws_enum_mod.MessageType = _MockMessageType
    ws_enum_mod.FrameType = _MockFrameType

    ws_const_mod = MagicMock()
    ws_const_mod.HEADER_TYPE = "type"
    ws_const_mod.HEADER_MESSAGE_ID = "message_id"
    ws_const_mod.HEADER_TRACE_ID = "trace_id"
    ws_const_mod.HEADER_SUM = "sum"
    ws_const_mod.HEADER_SEQ = "seq"
    ws_const_mod.HEADER_BIZ_RT = "biz_rt"

    ws_model_mod = MagicMock()
    ws_model_mod.Response = _MockResponse

    core_const_mod = MagicMock()
    core_const_mod.UTF_8 = "utf-8"

    core_json_mod = MagicMock()
    core_json_mod.JSON = MagicMock()
    core_json_mod.JSON.marshal = MagicMock(side_effect=lambda x: json.dumps({"code": 200}))

    mods = {
        "lark_oapi.ws": MagicMock(),
        "lark_oapi.ws.client": ws_client_mod,
        "lark_oapi.ws.enum": ws_enum_mod,
        "lark_oapi.ws.const": ws_const_mod,
        "lark_oapi.ws.model": ws_model_mod,
        "lark_oapi.core": MagicMock(),
        "lark_oapi.core.const": core_const_mod,
        "lark_oapi.core.json": core_json_mod,
    }
    return mods, _MockMessageType


def _run_ws_client_with_patch():
    """Execute _run_official_feishu_ws_client to install the monkey-patch,
    returning (ws_client, original_handle, MessageType)."""
    ws_mods, msg_type_cls = _ensure_ws_mocks()

    ws_client = MagicMock()
    original_handle = AsyncMock()
    ws_client._handle_data_frame = original_handle
    ws_client._event_handler = MagicMock()
    ws_client._event_handler.do_without_validation = MagicMock(return_value=None)
    ws_client._combine = MagicMock()
    ws_client._write_message = AsyncMock()
    ws_client.start = MagicMock()

    adapter = MagicMock()
    adapter._ws_reconnect_nonce = 0
    adapter._ws_reconnect_interval = 10
    adapter._ws_ping_interval = None
    adapter._ws_ping_timeout = None

    with patch.dict("sys.modules", ws_mods):
        feishu_module._run_official_feishu_ws_client(ws_client, adapter)

    return ws_client, original_handle, msg_type_cls


class TestWSCardMonkeyPatch:
    """Test the _handle_data_frame monkey-patch for CARD messages."""

    def test_patch_replaces_handle_data_frame(self):
        """After _run_official_feishu_ws_client, _handle_data_frame should be patched."""
        ws_client, original_handle, _ = _run_ws_client_with_patch()
        ws_client.start.assert_called_once()
        assert ws_client._handle_data_frame is not original_handle

    @pytest.mark.asyncio
    async def test_card_frame_calls_event_handler(self):
        """A CARD frame should be routed through do_without_validation."""
        ws_client, original_handle, _ = _run_ws_client_with_patch()
        patched = ws_client._handle_data_frame

        frame = _make_card_frame()
        await patched(frame)

        ws_client._event_handler.do_without_validation.assert_called_once_with(
            b'{"test": "data"}'
        )
        ws_client._write_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_event_frame_delegates_to_original(self):
        """A non-CARD frame should delegate to the original handler."""
        ws_client, original_handle, _ = _run_ws_client_with_patch()
        patched = ws_client._handle_data_frame

        frame = _make_card_frame(msg_type="event")
        await patched(frame)

        original_handle.assert_called_once_with(frame)
        ws_client._event_handler.do_without_validation.assert_not_called()

    @pytest.mark.asyncio
    async def test_card_frame_handler_error_returns_500(self):
        """If the card handler raises, the response should still be written."""
        ws_client, _, _ = _run_ws_client_with_patch()
        ws_client._event_handler.do_without_validation = MagicMock(
            side_effect=RuntimeError("handler boom")
        )
        patched = ws_client._handle_data_frame

        frame = _make_card_frame()
        await patched(frame)

        ws_client._write_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_card_frame_multipart_combines(self):
        """Multi-part CARD frames should be combined via _combine."""
        combined_payload = b'{"combined": true}'
        ws_client, _, _ = _run_ws_client_with_patch()
        ws_client._combine = MagicMock(return_value=combined_payload)
        patched = ws_client._handle_data_frame

        original_payload = b'{"test": "data"}'
        frame = _make_card_frame(sum_="3", seq="1", payload=original_payload)
        await patched(frame)

        ws_client._combine.assert_called_once_with("msg_card_001", 3, 1, original_payload)
        ws_client._event_handler.do_without_validation.assert_called_once_with(combined_payload)

    @pytest.mark.asyncio
    async def test_card_frame_multipart_incomplete_returns_early(self):
        """If _combine returns None (waiting for more parts), handler returns early."""
        ws_client, _, _ = _run_ws_client_with_patch()
        ws_client._combine = MagicMock(return_value=None)
        patched = ws_client._handle_data_frame

        frame = _make_card_frame(sum_="3", seq="1")
        await patched(frame)

        ws_client._event_handler.do_without_validation.assert_not_called()
        ws_client._write_message.assert_not_called()
