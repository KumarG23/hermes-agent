"""Exact, turn-local persistence evidence for the Runs API.

References stay on the agent, never in model context. Compression copies,
loaded history and uncommitted rows cannot acquire another object's identity.
"""


def committed_pair(agent, final_message, result):
    if (
        getattr(agent, "_run_binding_persistence_failed", False)
        or result.get("failed")
        or result.get("interrupted")
        or not result.get("completed")
        or result.get("response_transformed")
    ):
        return None
    user = getattr(agent, "_run_binding_user", None)
    if not isinstance(user, dict) or not isinstance(final_message, dict):
        return None
    if (
        user.get("role") != "user"
        or final_message.get("role") != "assistant"
        or final_message.get("tool_calls")
        or not result.get("final_response")
        or final_message.get("content") != result["final_response"]
    ):
        return None
    rows = getattr(agent, "_run_binding_rows", {})
    pair = []
    for message in (user, final_message):
        entry = rows.get(id(message))
        if (
            not entry
            or entry[0] is not message
            or entry[1] != agent.session_id
            or type(entry[2]) is not int
            or entry[2] <= 0
        ):
            return None
        pair.append(entry[2])
    if pair[0] == pair[1]:
        return None
    return pair
