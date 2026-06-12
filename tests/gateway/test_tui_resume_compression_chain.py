"""Regression test for #44640 — session.resume must resolve the compression chain.

When context compression forks a new child session, the TUI gateway's
``session.resume`` handler must call ``resolve_resume_session_id()`` so
that messages are loaded from the descendant session, not the stale parent.

``web_server.py`` and ``cli_commands_mixin.py`` already do this; this test
pins the behavior for ``tui_gateway/server.py``.
"""

import ast
import inspect


def _get_session_resume_source():
    """Return the source of the session.resume handler from tui_gateway/server.py."""
    import tui_gateway.server as srv

    # The handler is registered via @method("session.resume"). Walk module-level
    # AST to find the decorated function.
    source = inspect.getsource(srv)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Name)
                    and dec.func.id == "method"
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)
                    and dec.args[0].value == "session.resume"
                ):
                    return ast.get_source_segment(source, node)
    raise AssertionError("session.resume handler not found in tui_gateway/server.py")


class TestTuiResumeResolvesCompressionChain:
    """Ensure the TUI gateway session.resume handler resolves compressed sessions."""

    def test_session_resume_calls_resolve_resume_session_id(self):
        """The handler must call resolve_resume_session_id before reopen_session."""
        handler_src = _get_session_resume_source()

        assert "resolve_resume_session_id" in handler_src, (
            "session.resume handler in tui_gateway/server.py must call "
            "resolve_resume_session_id() to follow the compression chain. "
            "See #44640 and the equivalent calls in web_server.py / cli_commands_mixin.py."
        )

    def test_resolve_before_reopen(self):
        """resolve_resume_session_id must appear before reopen_session."""
        handler_src = _get_session_resume_source()

        resolve_pos = handler_src.find("resolve_resume_session_id")
        reopen_pos = handler_src.find("reopen_session")

        assert resolve_pos != -1, (
            "resolve_resume_session_id() not found in session.resume handler"
        )
        assert reopen_pos != -1, (
            "reopen_session() not found in session.resume handler"
        )
        assert resolve_pos < reopen_pos, (
            "resolve_resume_session_id() must be called BEFORE reopen_session() "
            "so the resolved target is used for loading messages."
        )

    def test_resolve_result_is_checked_before_use(self):
        """The resolve call result must be checked (truthy + differs) before
        overwriting target, to handle None / empty returns safely."""
        handler_src = _get_session_resume_source()

        # Verify via AST that the resolved value is used in a conditional,
        # not assigned directly to target without a guard.
        tree = ast.parse(handler_src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.BoolOp)
                and isinstance(node.test.op, ast.And)
            ):
                # Check if any value in the BoolOp references 'resolved'
                src_segment = ast.get_source_segment(handler_src, node.test)
                if src_segment and "resolved" in src_segment:
                    return  # Found a guarded conditional using 'resolved'

        raise AssertionError(
            "resolve_resume_session_id() result must be guarded with a conditional "
            "check before overwriting target — to handle None / DB inconsistency."
        )
