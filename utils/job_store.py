"""
Persistent job state for the Streamlit UI.

Stored in a separate module so Python's import cache keeps this dict alive
across Streamlit script reruns.  app.py is re-executed on every user
interaction; module-level variables defined there get reset each time.
Variables defined here are initialised ONCE (on first import) and survive
every subsequent rerun, which is exactly what the background worker needs.

The worker thread writes here; the Streamlit main thread reads here.
Both operations are safe in CPython because dict-key assignment and
list.append() are atomic under the GIL.
"""

_JOB: dict = {
    "running":      False,
    "active_step":  "",       # step key currently executing
    "progress":     [],       # list of {"step": str, "message": str}
    "result":       None,     # final report text when complete
    "citations":    [],       # source URLs extracted during research
    "error":        None,     # traceback string on failure
    "topic":        "",       # topic being researched
    "start_ts":     None,     # float from time.time() at launch
}
