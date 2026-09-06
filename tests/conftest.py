"""Fine sessione dei test Qt: le viste QWebEngineView vanno cancellate
davvero prima che la QApplication venga distrutta all'uscita di Python.

pytest-qt chiude i widget con `deleteLater`, ma per l'ultimo test nessun
event loop gira più: le pagine web arrivano vive allo smontaggio del
profilo ("Release of profile requested but WebEnginePage still not
deleted") e il processo va in SIGTRAP/SIGSEGV dentro QtQuick. Qui si
consegnano le cancellazioni differite e si fa girare il loop finché
Qt non ha finito di smontare i renderer."""


def pytest_sessionfinish(session, exitstatus):
    try:
        from PySide6.QtCore import QCoreApplication, QEvent
    except ImportError:
        return
    app = QCoreApplication.instance()
    if app is None:
        return
    for _ in range(10):
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
